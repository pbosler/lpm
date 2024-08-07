#ifndef LPM_VTK_IO_IMPL_HPP
#define LPM_VTK_IO_IMPL_HPP

#include "LpmConfig.h"

#ifdef LPM_USE_VTK

#include "lpm_assert.hpp"
#include "vtk/lpm_vtk_io.hpp"

namespace Lpm {

template <typename SeedType>
VtkPolymeshInterface<SeedType>::VtkPolymeshInterface(
    const PolyMesh2d<SeedType>& pm)
    : mesh_(pm) {
  polydata_ = vtkSmartPointer<vtkPolyData>::New();

  const auto pts = make_points();
  const auto polys = make_cells();
  const auto cell_area = make_cell_area();

  polydata_->SetPoints(pts);
  polydata_->SetPolys(polys);
  polydata_->GetCellData()->AddArray(cell_area);
  this->add_vector_point_data(pm.vertices.lag_crds.view,"lag_crds");
  this->add_vector_cell_data(pm.faces.lag_crds.view,"lag_crds");
}

template <typename SeedType>
VtkPolymeshInterface<SeedType>::VtkPolymeshInterface(
    const PolyMesh2d<SeedType>& pm,
    const typename scalar_view_type::HostMirror height_field) : mesh_(pm) {
  polydata_ = vtkSmartPointer<vtkPolyData>::New();

  const auto pts = make_points(height_field);
  const auto polys = make_cells();
  const auto cell_area = make_cell_area();

  polydata_->SetPoints(pts);
  polydata_->SetPolys(polys);
  polydata_->GetCellData()->AddArray(cell_area);
  this->add_vector_point_data(pm.vertices.lag_crds.view,"lag_crds");
  this->add_vector_cell_data(pm.faces.lag_crds.view,"lag_crds");
}

template <typename SeedType>
vtkSmartPointer<vtkPoints> VtkPolymeshInterface<SeedType>::make_points() const {
  auto result = vtkSmartPointer<vtkPoints>::New();
  const auto vert_crd_view = mesh_.vertices.phys_crds._hostview;
  Real crds[3];
  for (Index i = 0; i < mesh_.n_vertices_host(); ++i) {
    for (Short j = 0; j < SeedType::geo::ndim; ++j) {
      crds[j] = vert_crd_view(i, j);
    }
    result->InsertNextPoint(crds[0], crds[1],
                            (SeedType::geo::ndim == 3 ? crds[2] : 0));
  }
  return result;
}

template <typename SeedType>
vtkSmartPointer<vtkPoints> VtkPolymeshInterface<SeedType>::make_points(
    const typename scalar_view_type::HostMirror height_field) const {
  auto result = vtkSmartPointer<vtkPoints>::New();
  const auto vert_crd_view = mesh_.vertices.phys_crds._hostview;
  Real crds[SeedType::geo::ndim];
  if constexpr (std::is_same<typename SeedType::geo, PlaneGeometry>::value) {
    for (Index i = 0; i < mesh_.n_vertices_host(); ++i) {
      for (Short j = 0; j < SeedType::geo::ndim; ++j) {
        crds[j] = vert_crd_view(i, j);
      }
      result->InsertNextPoint(crds[0], crds[1], height_field(i));
    }
  }
  else {
    if constexpr (std::is_same<typename SeedType::geo, SphereGeometry>::value) {
      for (Index i=0; i<mesh_.n_vertices_host(); ++i) {
        for (Short j=0; j<SeedType::geo::ndim; ++j) {
          crds[j] = (1 + height_field(i)) * vert_crd_view(i,j);
        }
      }
    }
  }
  return result;
}

template <typename SeedType>
vtkSmartPointer<vtkCellArray> VtkPolymeshInterface<SeedType>::make_cells()
    const {
  auto result = vtkSmartPointer<vtkCellArray>::New();
  for (Index i = 0; i < mesh_.n_faces_host(); ++i) {
    if (!mesh_.faces.has_kids_host(i)) {
      result->InsertNextCell(SeedType::nfaceverts);
      for (Short j = 0; j < SeedType::nfaceverts; ++j) {
        result->InsertCellPoint(mesh_.faces.vert_host(i, j));
      }
    }
  }
  return result;
}

template <typename SeedType>
void VtkPolymeshInterface<SeedType>::add_tracers(
    const std::vector<scalar_view_type>& vert_tracers,
    const std::vector<scalar_view_type>& face_tracers) {
  LPM_REQUIRE(vert_tracers.size() == face_tracers.size());

  for (Short k = 0; k < vert_tracers.size(); ++k) {
    add_scalar_point_data(vert_tracers[k]);
    add_scalar_cell_data(face_tracers[k]);
  }
}

template <typename SeedType>
vtkSmartPointer<vtkDoubleArray> VtkPolymeshInterface<SeedType>::make_cell_area()
    const {
  auto result = vtkSmartPointer<vtkDoubleArray>::New();
  result->SetName("area");
  result->SetNumberOfComponents(1);
  result->SetNumberOfTuples(mesh_.faces.n_leaves_host());
  Index ctr = 0;
  for (Index i = 0; i < mesh_.n_faces_host(); ++i) {
    if (!mesh_.faces.has_kids_host(i)) {
      result->InsertTuple1(ctr++, mesh_.faces.area_host(i));
    }
  }
  LPM_ASSERT(ctr == mesh_.faces.n_leaves_host());
  return result;
}

template <typename SeedType>
void VtkPolymeshInterface<SeedType>::update_positions() {
  const auto newpts = make_points();
  polydata_->SetPoints(newpts);
}

template <typename SeedType>
void VtkPolymeshInterface<SeedType>::write(const std::string& ofname) {
  this->writer_ = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer_->SetInputData(this->polydata_);
  writer_->SetFileName(ofname.c_str());
  writer_->Write();
}

template <typename SeedType>
template <typename ViewType>
void VtkPolymeshInterface<SeedType>::add_scalar_point_data(
    const ViewType s, const std::string& name) {
  typename ViewType::HostMirror h_scalars;
  //   if (ViewType::is_hostspace) {
  //     h_scalars = s;
  //   }
  //   else {
  h_scalars = ko::create_mirror_view(s);
  ko::deep_copy(h_scalars, s);
  //   }
  auto points_scalar = vtkSmartPointer<vtkDoubleArray>::New();
  points_scalar->SetName((name.empty() ? s.label().c_str() : name.c_str()));
  points_scalar->SetNumberOfComponents(1);
  points_scalar->SetNumberOfTuples(mesh_.n_vertices_host());
  for (Index i = 0; i < mesh_.n_vertices_host(); ++i) {
    points_scalar->InsertTuple1(i, h_scalars(i));
  }
  polydata_->GetPointData()->AddArray(points_scalar);
}

template <typename SeedType>
template <typename ViewType>
void VtkPolymeshInterface<SeedType>::add_vector_point_data(
    const ViewType v, const std::string& name) {
  typename ViewType::HostMirror h_vectors;
  //   if (ViewType::is_hostspace) {
  //     h_vectors = v;
  //   }
  //   else {
  h_vectors = ko::create_mirror_view(v);
  ko::deep_copy(h_vectors, v);
  //   }
  auto point_vectors = vtkSmartPointer<vtkDoubleArray>::New();
  point_vectors->SetName((name.empty() ? v.label().c_str() : name.c_str()));
  point_vectors->SetNumberOfTuples(mesh_.n_vertices_host());
  point_vectors->SetNumberOfComponents(v.extent(1));
  if constexpr (SeedType::geo::ndim == 3) {
    for (Index i = 0; i < mesh_.n_vertices_host(); ++i) {
      point_vectors->InsertTuple3(i, h_vectors(i, 0), h_vectors(i, 1),
                                  h_vectors(i, 2));
    }
  } else {
    for (Index i = 0; i < mesh_.n_vertices_host(); ++i) {
      point_vectors->InsertTuple2(i, h_vectors(i, 0), h_vectors(i, 1));
    }
  }
  polydata_->GetPointData()->AddArray(point_vectors);
}

template <typename SeedType>
template <typename ViewType>
void VtkPolymeshInterface<SeedType>::add_scalar_cell_data(
    const ViewType s, const std::string& name) {
  typename ViewType::HostMirror h_scalars;
  //   if (ViewType::is_hostspace) {
  //     h_scalars = s;
  //   }
  //   else {
  h_scalars = ko::create_mirror_view(s);
  ko::deep_copy(h_scalars, s);
  //   }
  auto cells_scalar = vtkSmartPointer<vtkDoubleArray>::New();
  cells_scalar->SetName((name.empty() ? s.label().c_str() : name.c_str()));
  cells_scalar->SetNumberOfComponents(1);
  cells_scalar->SetNumberOfTuples(mesh_.faces.n_leaves_host());
  Index ctr = 0;
  for (Index i = 0; i < mesh_.n_faces_host(); ++i) {
    if (!mesh_.faces.has_kids_host(i)) {
      cells_scalar->InsertTuple1(ctr++, h_scalars(i));
    }
  }
  LPM_ASSERT(ctr == mesh_.faces.n_leaves_host());
  polydata_->GetCellData()->AddArray(cells_scalar);
}

template <typename SeedType>
template <typename ViewType>
void VtkPolymeshInterface<SeedType>::add_vector_cell_data(
    const ViewType v, const std::string& name) {
  typename ViewType::HostMirror h_vectors;
  //   if (ViewType::is_hostspace) {
  //     h_vectors = v;
  //   }
  //   else {
  h_vectors = ko::create_mirror_view(v);
  ko::deep_copy(h_vectors, v);
  //   }
  auto cell_vectors = vtkSmartPointer<vtkDoubleArray>::New();
  cell_vectors->SetName((name.empty() ? v.label().c_str() : name.c_str()));
  cell_vectors->SetNumberOfTuples(mesh_.faces.n_leaves_host());
  cell_vectors->SetNumberOfComponents(v.extent(1));
  Index ctr = 0;
  if constexpr (SeedType::geo::ndim == 3) {
    for (Index i = 0; i < mesh_.n_faces_host(); ++i) {
      if (!mesh_.faces.has_kids_host(i)) {
        cell_vectors->InsertTuple3(ctr++, h_vectors(i, 0), h_vectors(i, 1),
                                   h_vectors(i, 2));
      }
    }
  } else {
    for (Index i = 0; i < mesh_.n_faces_host(); ++i) {
      if (!mesh_.faces.has_kids_host(i)) {
        cell_vectors->InsertTuple2(ctr++, h_vectors(i, 0), h_vectors(i, 1));
      }
    }
  }
  LPM_ASSERT(ctr == mesh_.faces.n_leaves_host());
  polydata_->GetCellData()->AddArray(cell_vectors);
}

#ifdef LPM_ENABLE_DFS

template <typename VT>
void VtkGridInterface::add_scalar_point_data(const VT& s, const std::string& name) {
  auto h_scalars = Kokkos::create_mirror_view(s);
  Kokkos::deep_copy(h_scalars, s);
  auto vtk_scalar = vtkSmartPointer<vtkDoubleArray>::New();
  vtk_scalar->SetName((name.empty() ? s.label().c_str() : name.c_str()));
  vtk_scalar->SetNumberOfComponents(1);
  vtk_scalar->SetNumberOfTuples(grid_.size() + grid_.nlat);
  Index vtk_idx = 0;
  for (Index i=0; i<grid_.nlat; ++i) {
    for (Index j=0; j<grid_.nlon; ++j) {
      vtk_scalar->InsertTuple1(vtk_idx++, h_scalars(i*grid_.nlon + j));
    }
    vtk_scalar->InsertTuple1(vtk_idx++, h_scalars(i*grid_.nlon));
  }
  vtk_grid_->GetPointData()->AddArray(vtk_scalar);
}

template <typename VT>
void VtkGridInterface::add_vector_point_data(const VT& v, const std::string& name) {
  auto h_vecs = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h_vecs, v);
  auto vtk_vector = vtkSmartPointer<vtkDoubleArray>::New();
  vtk_vector->SetName((name.empty() ? v.label().c_str() : name.c_str()));
  vtk_vector->SetNumberOfComponents(3);
  vtk_vector->SetNumberOfTuples(grid_.size() + grid_.nlat);
  Index vtk_idx = 0;
  for (Index i=0; i<grid_.nlat; ++i) {
    for (Index j=0; j<grid_.nlon; ++j) {
      const auto mvec = Kokkos::subview(h_vecs, i*grid_.nlon + j, Kokkos::ALL);
      vtk_vector->InsertTuple3(vtk_idx++, mvec[0], mvec[1], mvec[2]);
    }
    const auto mvec = Kokkos::subview(h_vecs, i*grid_.nlon, Kokkos::ALL);
    vtk_vector->InsertTuple3(vtk_idx++, mvec[0], mvec[1], mvec[2]);
  }
  vtk_grid_->GetPointData()->AddArray(vtk_vector);
}
#endif

}  // namespace Lpm

#endif  // LPM_USE_VTK
#endif
