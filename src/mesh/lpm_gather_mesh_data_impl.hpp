#ifndef LPM_GATHER_MESH_DATA_IMPL_HPP
#define LPM_GATHER_MESH_DATA_IMPL_HPP

#include "mesh/lpm_gather_mesh_data.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_planar_grid.hpp"

namespace Lpm {

/** Gathers face leaf coordinates from the full array of face coordinates
  (which includes parent faces).  Use this to avoid duplicate coordinate
  entries for divided faces.
*/
template <typename SeedType>
struct GatherFaceLeafCrds {
  scalar_view_type x;
  scalar_view_type y;
  scalar_view_type z;
  typename SeedType::geo::crd_view_type face_crds;
  index_view_type face_leaf_idx;
  mask_view_type face_mask;
  Index vert_offset;

  GatherFaceLeafCrds(const scalar_view_type sx,
    const scalar_view_type sy,
    const typename SeedType::geo::crd_view_type fcrds,
    const index_view_type fleaves,
    const mask_view_type fmask,
    const Index offset) :
    x(sx),
    y(sy),
    face_crds(fcrds),
    face_leaf_idx(fleaves),
    face_mask(fmask),
    vert_offset(offset) {}

  GatherFaceLeafCrds(const scalar_view_type sx,
    const scalar_view_type sy,
    const scalar_view_type sz,
    const typename SeedType::geo::crd_view_type fcrds,
    const index_view_type fleaves,
    const mask_view_type fmask,
    const Index offset) :
    x(sx),
    y(sy),
    z(sz),
    face_crds(fcrds),
    face_leaf_idx(fleaves),
    face_mask(fmask),
    vert_offset(offset) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!face_mask(i)) {
      x(vert_offset + face_leaf_idx(i)) = face_crds(i,0);
      y(vert_offset + face_leaf_idx(i)) = face_crds(i,1);
      if constexpr (SeedType::geo::ndim == 3) {
        z(vert_offset + face_leaf_idx(i)) = face_crds(i,2);
      }
    }
  }
};

/**  Gathers scalar field data from vertices and faces so that only
  vertices and leaf face values are included in the output array.

  Use this functor with external interpolation libraries.
*/
struct GatherScalarFaceData {
  scalar_view_type data;
  scalar_view_type face_data;
  index_view_type face_leaf_idx;
  mask_view_type face_mask;
  Index vert_offset;

  GatherScalarFaceData(scalar_view_type sdata,
    const scalar_view_type fdata,
    const index_view_type fleaves,
    const mask_view_type fmask,
    const Index offset) :
    data(sdata),
    face_data(fdata),
    face_leaf_idx(fleaves),
    face_mask(fmask),
    vert_offset(offset) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!face_mask(i)) {
      data(vert_offset + face_leaf_idx(i)) = face_data(i);
    }
  }
};

/**  Gathers vector field data from vertices and faces so that only
  vertices and leaf face values are included in the output array.

  Use this functor with external interpolation libraries.
*/
struct GatherVectorFaceData {
  Kokkos::View<Real**> data;
  Kokkos::View<Real**> face_data;
  index_view_type face_leaf_idx;
  mask_view_type face_mask;
  Index vert_offset;

  GatherVectorFaceData(Kokkos::View<Real**> sdata,
    const Kokkos::View<Real**> fdata,
    const index_view_type fleaves,
    const mask_view_type fmask,
    const Index offset) :
    data(sdata),
    face_data(fdata),
    face_leaf_idx(fleaves),
    face_mask(fmask),
    vert_offset(offset) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!face_mask(i)) {
      for (int j=0; j<data.extent(1); ++j) {
        data(vert_offset + face_leaf_idx(i), j) = face_data(i,j);
      }
    }
  }
};

template <typename SeedType>
void GatherMeshData<SeedType>::init_scalar_field(const std::string& name,
  const scalar_view_type& view) {
  scalar_fields.emplace(name, view);
  h_scalar_fields.emplace(name, Kokkos::create_mirror_view(view));
}

template <typename SeedType>
GatherMeshData<SeedType>::GatherMeshData(
  const std::shared_ptr<PolyMesh2d<SeedType>> pm) :
  mesh(pm),
  x("gathered_x",
    pm->n_vertices_host() + pm->faces.n_leaves_host()),
  y("gathered_y",
    pm->n_vertices_host() + pm->faces.n_leaves_host())
   {
      if constexpr (SeedType::geo::ndim == 3) {
        z = scalar_view_type("gathered_z",
    pm->n_vertices_host() + pm->faces.n_leaves_host());
      }
      gather_coordinates<SeedType::geo::ndim>();
      h_x = Kokkos::create_mirror_view(x);
      h_y = Kokkos::create_mirror_view(y);
      if constexpr (SeedType::geo::ndim == 3) {
        h_z = Kokkos::create_mirror_view(z);
      }
    }

template <typename SeedType>
GatherMeshData<SeedType>::GatherMeshData(const PlanarGrid& grid) :
  x("gathered_x", grid.size()),
  y("gathered_y", grid.size()) {

  h_x = Kokkos::create_mirror_view(x);
  h_y = Kokkos::create_mirror_view(y);
  auto xx = x;
  auto yy = y;
  auto gxy = grid.pts;
  Kokkos::parallel_for(grid.size(),
    KOKKOS_LAMBDA (const Index i) {
      xx(i) = gxy(i,0);
      yy(i) = gxy(i,1);
    });
}

template <typename SeedType>
void GatherMeshData<SeedType>::update_host() const {
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  if constexpr (SeedType::geo::ndim == 3) {
    Kokkos::deep_copy(h_z, z);
  }
  for (const auto& sf : scalar_fields) {
    Kokkos::deep_copy(h_scalar_fields.at(sf.first), sf.second);
  }
  for (const auto& vf : vector_fields) {
    Kokkos::deep_copy(h_vector_fields.at(vf.first), vf.second);
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::update_device() const {
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  if constexpr (SeedType::geo::ndim == 3) {
    Kokkos::deep_copy(z, h_z);
  }
  for (const auto& sf : scalar_fields) {
    Kokkos::deep_copy(sf.second, h_scalar_fields.at(sf.first));
  }
  for (const auto& vf : vector_fields) {
    Kokkos::deep_copy(vf.second, h_vector_fields.at(vf.first));
  }
}

template <typename SeedType>  template <int ndim>
typename std::enable_if<ndim==3, void>::type
GatherMeshData<SeedType>::gather_coordinates() {
  auto xx = x;
  auto yy = y;
  auto zz = z;
  auto mesh_vert_xyz = mesh->vertices.phys_crds->crds;
  Kokkos::parallel_for(mesh->n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      xx(i) = mesh_vert_xyz(i,0);
      yy(i) = mesh_vert_xyz(i,1);
      zz(i) = mesh_vert_xyz(i,2);
    });
  const auto vert_offset = mesh->n_vertices_host();
  const auto mesh_face_xyz = mesh->faces.phys_crds->crds;
  const auto face_mask = mesh->faces.mask;
  const auto face_leaf_idx = mesh->faces.leaf_idx;
  Kokkos::parallel_for(mesh->n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      if (!face_mask(i)) {
        x(vert_offset + face_leaf_idx(i)) = mesh_face_xyz(i, 0);
        y(vert_offset + face_leaf_idx(i)) = mesh_face_xyz(i, 1);
        z(vert_offset + face_leaf_idx(i)) = mesh_face_xyz(i, 2);
      }
    });
}

template <typename SeedType> template <int ndim>
typename std::enable_if<ndim==2, void>::type
GatherMeshData<SeedType>::gather_coordinates() {
  auto xx = x;
  auto yy = y;
  auto mesh_vert_xyz = mesh->vertices.phys_crds->crds;
  Kokkos::parallel_for(mesh->n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      xx(i) = mesh_vert_xyz(i,0);
      yy(i) = mesh_vert_xyz(i,1);
    });
  const auto vert_offset = mesh->n_vertices_host();
  const auto mesh_face_xyz = mesh->faces.phys_crds->crds;
  const auto face_mask = mesh->faces.mask;
  const auto face_leaf_idx = mesh->faces.leaf_idx;
  Kokkos::parallel_for(mesh->n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      if (!face_mask(i)) {
        x(vert_offset + face_leaf_idx(i)) = mesh_face_xyz(i, 0);
        y(vert_offset + face_leaf_idx(i)) = mesh_face_xyz(i, 1);
      }
    });
}

template <typename SeedType>
void GatherMeshData<SeedType>::init_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields) {
    for (const auto& sf : vert_fields) {

      LPM_ASSERT(face_fields.find(sf.first) != face_fields.end());

      scalar_fields.emplace(sf.first, scalar_view_type(sf.first,
        mesh->n_vertices_host() + mesh->faces.n_leaves_host()));
      h_scalar_fields.emplace(sf.first, Kokkos::create_mirror_view(
        scalar_fields.at(sf.first)));
    }
  }

template <typename SeedType>
void GatherMeshData<SeedType>::init_vector_fields(
    const std::map<std::string,
      VectorField<typename SeedType::geo, VertexField>>& vert_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, FaceField>>& face_fields) {

    for (const auto& vf : vert_fields) {

      LPM_ASSERT(face_fields.find(vf.first) != face_fields.end());

      vector_fields.emplace(vf.first,
        typename SeedType::geo::vec_view_type(vf.first,
          mesh->n_vertices_host() + mesh->faces.n_leaves_host()));

      h_vector_fields.emplace(vf.first,
        Kokkos::create_mirror_view(vector_fields.at(vf.first)));
    }
  }

template <typename SeedType>
void GatherMeshData<SeedType>::gather_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields) {

    for (const auto& sf : vert_fields) {
      auto vert_vals = Kokkos::subview(scalar_fields.at(sf.first),
        std::make_pair(0, mesh->n_vertices_host()));
      Kokkos::deep_copy(vert_vals,
        Kokkos::subview(sf.second.view, std::make_pair(0, mesh->n_vertices_host())));
      Kokkos::parallel_for(mesh->n_faces_host(),
        GatherScalarFaceData(scalar_fields.at(sf.first),
          face_fields.at(sf.first).view,
          mesh->faces.leaf_idx,
          mesh->faces.mask,
          mesh->n_vertices_host()));
    }
  }

template <typename SeedType>
void GatherMeshData<SeedType>::gather_vector_fields(
    const std::map<std::string,
      VectorField<typename SeedType::geo, VertexField>>& vert_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, FaceField>>& face_fields) {

      for (const auto& vf : vert_fields) {
        auto vert_vals = Kokkos::subview(vector_fields.at(vf.first),
          std::make_pair(0, mesh->n_vertices_host()), Kokkos::ALL);
        Kokkos::deep_copy(vert_vals,
          Kokkos::subview(vf.second.view,
            std::make_pair(0, mesh->n_vertices_host(), Kokkos::ALL)));
        Kokkos::parallel_for(mesh->n_faces_host(),
          GatherVectorFaceData(vector_fields.at(vf.first),
            face_fields.at(vf.first).view,
            mesh->faces.leaf_idx,
            mesh->faces.mask,
            mesh->n_vertices_host()));
      }
  }


template <typename SeedType>
std::string GatherMeshData<SeedType>::info_string(const Int tab_lev, const bool verbose) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_lev);
  ss << tabstr << "GatherMeshData<" << SeedType::id_string() << "> info: "
     << " n() = " << n() << "\n";
  tabstr += "\t";
  if (!scalar_fields.empty()) {
    ss << tabstr << "scalar fields : ";
    for (const auto& sf : scalar_fields) {
      ss << sf.first << " ";
    }
    ss << "\n";
  }
  if (!vector_fields.empty()) {
    ss << tabstr << "vector fields : ";
    for (const auto& vf : vector_fields) {
      ss << vf.first << " ";
    }
    ss << "\n";
  }
  if (verbose) {
    ss << tabstr << "crds:\n";
    for (Index i=0; i<n(); ++i) {
      ss << tabstr << "( ";
      ss << h_x(i) << " " << h_y(i) << " ";
      if constexpr (SeedType::geo::ndim == 3) {
        ss << h_z(i) << " ";
      }
      ss << ")\n";
      ss << "----------------------------\n";
    }
    if (!scalar_fields.empty()) {
      for (const auto& sf : scalar_fields) {
        ss << tabstr << sf.first << " vals:\n";
        for (Index i=0; i<n(); ++i) {
          ss << h_scalar_fields.at(sf.first)(i) << " ";
        }
        ss << "\n";
      }
      ss << "----------------------------\n";
    }
    if (!vector_fields.empty()) {
      for (const auto& vf : vector_fields) {
        ss << tabstr << vf.first << " vals: \n";
        for (Index i=0; i<n(); ++i) {
          ss << "( ";
          for (Int j=0; j<SeedType::geo::ndim; ++j) {
            ss << h_vector_fields.at(vf.first)(i,j) << " ";
          }
          ss << ")\n";
        }
      }
      ss << "----------------------------\n";
    }
  }
  return ss.str();
}

}

#endif
