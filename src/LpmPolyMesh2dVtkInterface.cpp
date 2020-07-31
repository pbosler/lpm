#include "LpmPolyMesh2dVtkInterface.hpp"

namespace Lpm {

template <typename SeedType>
Polymesh2dVtkInterface<SeedType>::Polymesh2dVtkInterface(const std::shared_ptr<PolyMesh2d<SeedType>>& pm) :
  mesh(pm) {

  polydata = vtkSmartPointer<vtkPolyData>::New();

  const auto pts = make_points();
  const auto polys = make_cells();
  const auto ca = make_cell_area();
  polydata->SetPoints(pts);
  polydata->SetPolys(polys);
  polydata->GetCellData()->AddArray(ca->GetAbstractArray(0));
}

template <typename SeedType>
vtkSmartPointer<vtkPoints> Polymesh2dVtkInterface<SeedType>::make_points() const {
  auto result = vtkSmartPointer<vtkPoints>::New();
  const auto vx = mesh->physVerts.getHostCrdView();
  Real crds[SeedType::geo];
  for (Index i=0; i<mesh->nvertsHost(); ++i) {
    for (Short j=0; j<SeedType::geo::ndim; ++j) {
      crds[j] = vx(i,j);
    }
    result->InsertNextPoint(crds[0], crds[1], (SeedType::ndim == 3 ? crds[2] : 0.0));
  }
  return result;
}

template <typename SeedType>
vtkSmartPointer<vtkCellArray> Polymesh2dVtkInterface<SeedType>::make_cells() const {
  auto result = vtkSmartPointer<vtkCellArray>::New();
  for (Index i=0; i<mesh->nfacesHost(); ++i) {
    if (!mesh->faces.hasKidsHost(i)) {
      result->InsertNextCell(SeedType::nfaceverts);
      for (Int j=0; j<SeedType::nfaceverts; ++j) {
        result->InsertCellPoint(mesh->faces.getVertHost(i,j));
      }
    }
  }
  return result;
}

template <typename SeedType>
vtkSmartPointer<vtkDoubleArray> Polymesh2dVtkInterface<SeedType>::make_cell_area() const {
  auto result = vtkSmartPointer<vtkDoubleArray>::New();
  result->SetName("area");
  result->SetNumberOfComponents(1);
  result->SetNumberOfTuples(mesh->faces.nLeavesHost());
  for (Index i=0; i<mesh->nfacesHost(); ++i) {
    if (!mesh->faces.hasKidsHost()) {
      result->InsertTuple1(i,mesh->faces->getAreaHost(i));
    }
  }
  return result;
}

template <typename SeedType>
void Polymesh2dVtkInterface<SeedType>::updatePositions() {
  const auto newpts = make_points();
  polydata->SetPoints(newpts);
}

template <typename SeedType>
void Polymesh2dVtkInterface<SeedType>::updateAreas() {
  const auto newa = make_cell_area();
  polydata->GetCellData()->GetAbstractArray(0) = newa->GetAbstractArray();
}

template <typename SeedType> template <typename ViewType>
void Polymesh2dVtkInterface<SeedType>::addScalarPointData(const ViewType& s, const std::string& name) {
  auto pdata_host = ko::create_mirror_view(s);
  ko::deep_copy(pdata_host, s);

  auto pd = vtkSmartPointer<vtkDoubleArray>::New();
  pd->SetName((name.empty() ? s.label().c_str() : name.c_str()));
  pd->SetNumberOfComponents(1);
  pd->SetNumberOfTuples(mesh->nvertsHost());
  for (Index i=0; i<mesh->nvertsHost(); ++i) {
    pd->InsertTuple1(pdata_host(i));
  }
  polydata->GetPointData()->AddArray(pd);
}

template <typename SeedType> template <typename ViewType>
void Polymesh2dVtkInterface<SeedType>::addVectorPointData(const ViewType& v, const std::string& name) {
  auto pdata_host = ko::create_mirror_view(v);
  ko::deep_copy(pdata_host, v);

  auto pd = vtkSmartPointer<vtkDoubleArray>::New();
  pd->SetName((name.empty() ? v.label().c_str() : name.c_str()));
  pd->SetNumberOfComponents(v.extent(1));
  pd->SetNumberOfTuples(mesh->nvertsHost());
  for (Index i=0; i<mesh->nvertsHost(); ++i) {
    for (Short j=0; j<v.extent(1); ++j) {
      pd->InsertTuple3(pdata_host(i,0), pdata_host(i,1), (SeedType::geo::ndim == 3 ? pdata_host(i,2) : 0.0));
    }
  }
  polydata->GetPointData()->AddArray(pd);
}

template <typename SeedType> template <typename ViewType>
void Polymesh2dVtkInterface<SeedType>::addScalarCellData(const ViewType& s, const std::string& name) {
  auto cdata_host = ko::create_mirror_view(s);
  ko::deep_copy(cdata_host, s);

  auto cd = vtkSmartPointer<vtkDoubleArray>::New();
  cd->SetName((name.empty() ? s.label().c_str() : name.c_str()));
  cd->SetNumberOfComponents(mesh->faces.nLeavesHost());
  cd->SetNumberOfTuples(1);
  for (Index i=0; i<mesh->nfacesHost(); ++i) {
    if (!mesh->faces.hasKidsHost(i)) {
      cd->InsertTuple1(cdata_host(i));
    }
  }
  polydata->GetCellData()->AddArray(cd);
}

template <typename SeedType> template <typename ViewType>
void Polymesh2dVtkInterface<SeedType>::addVectorCellData(const ViewType& v, const std::string& name) {
  auto cdata_host = ko::create_mirror_view(v);
  ko::deep_copy(cdata_host, v);

  auto cd = vtkSmartPointer<vtkDoubleArray>::New();
  cd->SetName((name.empty() ? v.label().c_str() : name.c_str()));
  cd->SetNumberOfComponents(v.extent(1));
  cd->SetNumberOfTuples(mesh->faces.nLeavesHost());
  for (Index i=0; i<mesh->nfacesHost(); ++i) {
    if (!mesh->faces.hasKidsHost(i)) {
      cd->InsertTuple3(cdata_host(i,0),cdata_host(i,1), (SeedType::geo::ndim == 3 ? cdata_host(i,2) : 0.0));
    }
  }
  polydata->GetCellData()->AddArray(cd);
}

}