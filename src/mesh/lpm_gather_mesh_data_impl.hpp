#ifndef LPM_GATHER_MESH_DATA_IMPL_HPP
#define LPM_GATHER_MESH_DATA_IMPL_HPP

#include "lpm_planar_grid.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {


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

  GatherScalarFaceData(scalar_view_type sdata, const scalar_view_type fdata,
                       const index_view_type fleaves,
                       const mask_view_type fmask, const Index offset)
      : data(sdata),
        face_data(fdata),
        face_leaf_idx(fleaves),
        face_mask(fmask),
        vert_offset(offset) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
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
                       const mask_view_type fmask, const Index offset)
      : data(sdata),
        face_data(fdata),
        face_leaf_idx(fleaves),
        face_mask(fmask),
        vert_offset(offset) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!face_mask(i)) {
      for (int j = 0; j < data.extent(1); ++j) {
        data(vert_offset + face_leaf_idx(i), j) = face_data(i, j);
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
void GatherMeshData<SeedType>::unpack_coordinates() {
  LPM_ASSERT(!unpacked);
  this->template unpack_helper<SeedType::geo::ndim>();
}

template <typename SeedType>
template <int ndim>
typename std::enable_if<ndim == 3, void>::type
GatherMeshData<SeedType>::unpack_helper() {

  x = scalar_view_type("gathered_x",
                       mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  y = scalar_view_type("gathered_y",
                       mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  lag_x = scalar_view_type("gathered_lag_x",
                           mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  lag_y = scalar_view_type("gathered_lag_y",
                           mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  z = scalar_view_type("gathered_z",
                       mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  lag_z = scalar_view_type("gathered_lag_z",
                           mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  h_x = Kokkos::create_mirror_view(x);
  h_y = Kokkos::create_mirror_view(y);
  h_lag_x = Kokkos::create_mirror_view(lag_x);
  h_lag_y = Kokkos::create_mirror_view(lag_y);
  h_z = Kokkos::create_mirror_view(z);
  h_lag_z = Kokkos::create_mirror_view(lag_z);
  Kokkos::parallel_for(
      mesh.n_vertices_host() + mesh.faces.n_leaves_host(),
      KOKKOS_LAMBDA(const Index i) {
        x(i) = phys_crds(i, 0);
        y(i) = phys_crds(i, 1);
        z(i) = phys_crds(i, 2);
        lag_x(i) = lag_crds(i, 0);
        lag_y(i) = lag_crds(i, 1);
        lag_z(i) = lag_crds(i, 2);
      });
  unpacked = true;
}

template <typename SeedType>
template <int ndim>
typename std::enable_if<ndim == 2, void>::type
GatherMeshData<SeedType>::unpack_helper() {
  x = scalar_view_type("gathered_x",
                       mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  y = scalar_view_type("gathered_y",
                       mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  lag_x = scalar_view_type("gathered_lag_x",
                           mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  lag_y = scalar_view_type("gathered_lag_y",
                           mesh.n_vertices_host() + mesh.faces.n_leaves_host());
  h_x = Kokkos::create_mirror_view(x);
  h_y = Kokkos::create_mirror_view(y);
  h_lag_x = Kokkos::create_mirror_view(lag_x);
  h_lag_y = Kokkos::create_mirror_view(lag_y);
  Kokkos::parallel_for(
      mesh.n_vertices_host() + mesh.faces.n_leaves_host(),
      KOKKOS_LAMBDA(const Index i) {
        x(i) = phys_crds(i, 0);
        y(i) = phys_crds(i, 1);
        lag_x(i) = lag_crds(i, 0);
        lag_y(i) = lag_crds(i, 1);
      });
  unpacked = true;
}

template <typename SeedType>
GatherMeshData<SeedType>::GatherMeshData(const PolyMesh2d<SeedType>& pm)
    : mesh(pm),
      phys_crds("gathered_phys_crds",
                pm.n_vertices_host() + pm.faces.n_leaves_host()),
      lag_crds("gathered_phys_crds",
               pm.n_vertices_host() + pm.faces.n_leaves_host()),
      unpacked(false) {
  h_phys_crds = Kokkos::create_mirror_view(phys_crds);
  h_lag_crds = Kokkos::create_mirror_view(lag_crds);
  gather_coordinates();
}

template <typename SeedType>
void GatherMeshData<SeedType>::update_host() const {
  if (unpacked) {
    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_lag_x, lag_x);
    Kokkos::deep_copy(h_y, y);
    Kokkos::deep_copy(h_lag_y, lag_y);
    if constexpr (SeedType::geo::ndim == 3) {
      Kokkos::deep_copy(h_z, z);
      Kokkos::deep_copy(h_lag_z, lag_z);
    }
  }
  Kokkos::deep_copy(h_phys_crds, phys_crds);
  Kokkos::deep_copy(h_lag_crds, lag_crds);

  for (const auto& sf : scalar_fields) {
    Kokkos::deep_copy(h_scalar_fields.at(sf.first), sf.second);
  }
  for (const auto& vf : vector_fields) {
    Kokkos::deep_copy(h_vector_fields.at(vf.first), vf.second);
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::update_device() const {
  if (unpacked) {
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(lag_x, h_lag_x);
    Kokkos::deep_copy(lag_y, h_lag_y);
    if constexpr (SeedType::geo::ndim == 3) {
      Kokkos::deep_copy(z, h_z);
      Kokkos::deep_copy(lag_z, h_lag_z);
    }
  }
  Kokkos::deep_copy(phys_crds, h_phys_crds);
  Kokkos::deep_copy(lag_crds, h_lag_crds);
  for (const auto& sf : scalar_fields) {
    Kokkos::deep_copy(sf.second, h_scalar_fields.at(sf.first));
  }
  for (const auto& vf : vector_fields) {
    Kokkos::deep_copy(vf.second, h_vector_fields.at(vf.first));
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::gather_coordinates() {
  const auto mesh_vert_xyz = mesh.vertices.phys_crds.view;
  const auto mesh_vert_lag_xyz = mesh.vertices.lag_crds.view;
  const Kokkos::MDRangePolicy<Kokkos::Rank<2>> vert_policy(
      {0, 0}, {mesh.n_vertices_host(), SeedType::geo::ndim});
  Kokkos::parallel_for(
      vert_policy, KOKKOS_LAMBDA(const Index i, const Int j) {
        phys_crds(i, j) = mesh_vert_xyz(i, j);
        lag_crds(i, j) = mesh_vert_lag_xyz(i, j);
      });
  const auto vert_offset = mesh.n_vertices_host();
  const auto mesh_face_xyz = mesh.faces.phys_crds.view;
  const auto mesh_face_lag_xyz = mesh.faces.lag_crds.view;
  const auto face_mask = mesh.faces.mask;
  const auto face_leaf_idx = mesh.faces.leaf_idx;
  const Kokkos::MDRangePolicy<Kokkos::Rank<2>> face_policy(
      {0, 0}, {mesh.n_faces_host(), SeedType::geo::ndim});
  Kokkos::parallel_for(
      face_policy, KOKKOS_LAMBDA(const Index i, const Int j) {
        if (!face_mask(i)) {
          phys_crds(vert_offset + face_leaf_idx(i), j) = mesh_face_xyz(i, j);
          lag_crds(vert_offset + face_leaf_idx(i), j) = mesh_face_lag_xyz(i, j);
        }
      });
}

template <typename SeedType>
void GatherMeshData<SeedType>::gather_coordinates(
  typename SeedType::geo::crd_view_type passive_x,
  typename SeedType::geo::crd_view_type active_x) {
  const Kokkos::MDRangePolicy<Kokkos::Rank<2>> vert_policy(
    {0, 0}, {mesh.n_vertices_host(), SeedType::geo::ndim});
  Kokkos::parallel_for(vert_policy,
    KOKKOS_LAMBDA (const Index i, const int j) {
      phys_crds(i,j) = passive_x(i,j);
    });

  const auto vert_offset = mesh.n_vertices_host();
  const auto face_mask = mesh.faces.mask;
  const auto face_leaf_idx = mesh.faces.leaf_idx;
  const Kokkos::MDRangePolicy<Kokkos::Rank<2>> face_policy(
      {0, 0}, {mesh.n_faces_host(), SeedType::geo::ndim});
  Kokkos::parallel_for(
      face_policy, KOKKOS_LAMBDA(const Index i, const Int j) {
        if (!face_mask(i)) {
          phys_crds(vert_offset + face_leaf_idx(i), j) = active_x(i, j);
        }
      });
}

template <typename SeedType>
void GatherMeshData<SeedType>::init_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields) {
  for (const auto& sf : vert_fields) {
    LPM_ASSERT(face_fields.find(sf.first) != face_fields.end());

    scalar_fields.emplace(
        sf.first, scalar_view_type(sf.first, mesh.n_vertices_host() +
                                                 mesh.faces.n_leaves_host()));
    h_scalar_fields.emplace(
        sf.first, Kokkos::create_mirror_view(scalar_fields.at(sf.first)));
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::init_vector_fields(
    const std::map<std::string, VectorField<typename SeedType::geo,
                                            VertexField>>& vert_fields,
    const std::map<std::string, VectorField<typename SeedType::geo, FaceField>>&
        face_fields) {
  for (const auto& vf : vert_fields) {
    LPM_ASSERT(face_fields.find(vf.first) != face_fields.end());

    vector_fields.emplace(
        vf.first,
        typename SeedType::geo::vec_view_type(
            vf.first, mesh.n_vertices_host() + mesh.faces.n_leaves_host()));

    h_vector_fields.emplace(
        vf.first, Kokkos::create_mirror_view(vector_fields.at(vf.first)));
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::gather_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields) {
  for (const auto& sf : vert_fields) {
    auto vert_vals = Kokkos::subview(scalar_fields.at(sf.first),
                                     std::make_pair(0, mesh.n_vertices_host()));
    Kokkos::deep_copy(
        vert_vals, Kokkos::subview(sf.second.view,
                                   std::make_pair(0, mesh.n_vertices_host())));
    Kokkos::parallel_for(
        mesh.n_faces_host(),
        GatherScalarFaceData(scalar_fields.at(sf.first),
                             face_fields.at(sf.first).view, mesh.faces.leaf_idx,
                             mesh.faces.mask, mesh.n_vertices_host()));
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::gather_scalar_fields(
  const std::map<std::string, scalar_view_type>& passive_fields,
  const std::map<std::string, scalar_view_type>& active_fields) {

  for (const auto& sf : passive_fields) {
    auto vert_vals = Kokkos::subview(scalar_fields.at(sf.first),
      std::make_pair(0, mesh.n_vertices_host()));

    Kokkos::deep_copy(vert_vals, Kokkos::subview(sf.second,
      std::make_pair(0, mesh.n_vertices_host())));

    Kokkos::parallel_for(mesh.n_faces_host(),
      GatherScalarFaceData(scalar_fields.at(sf.first),
        active_fields.at(sf.first), mesh.faces.leaf_idx,
        mesh.faces.mask, mesh.n_vertices_host()));
  }
}

template <typename SeedType>
void GatherMeshData<SeedType>::gather_vector_fields(
    const std::map<std::string, VectorField<typename SeedType::geo,
                                            VertexField>>& vert_fields,
    const std::map<std::string, VectorField<typename SeedType::geo, FaceField>>&
        face_fields) {
  for (const auto& vf : vert_fields) {
    auto vert_vals =
        Kokkos::subview(vector_fields.at(vf.first),
                        std::make_pair(0, mesh.n_vertices_host()), Kokkos::ALL);
    Kokkos::deep_copy(
        vert_vals, Kokkos::subview(
                       vf.second.view,
                       std::make_pair(0, mesh.n_vertices_host()), Kokkos::ALL));
    Kokkos::parallel_for(
        mesh.n_faces_host(),
        GatherVectorFaceData(vector_fields.at(vf.first),
                             face_fields.at(vf.first).view, mesh.faces.leaf_idx,
                             mesh.faces.mask, mesh.n_vertices_host()));
  }
}

template <typename SeedType>
std::string GatherMeshData<SeedType>::info_string(const Int tab_lev,
                                                  const bool verbose) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_lev);
  ss << tabstr << "GatherMeshData<" << SeedType::id_string() << "> info: "
     << " n() = " << n() << "\n";
  tabstr += "\t";
  if (!scalar_fields.empty()) {
    ss << tabstr << "scalar fields : ";
    for (const auto& sf : scalar_fields) {
      typename Kokkos::MinMax<Real>::value_type min_max;
      auto vals = sf.second;

      Kokkos::parallel_reduce(n(), KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
        if (vals(i) < mm.min_val) mm.min_val = vals(i);
        if (vals(i) > mm.max_val) mm.max_val = vals(i);
      }, Kokkos::MinMax<Real>(min_max));
      ss << sf.first << " (min, max) = (" << min_max.min_val << ", " << min_max.max_val << ") ";
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
    for (Index i = 0; i < n(); ++i) {
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
        for (Index i = 0; i < n(); ++i) {
          ss << h_scalar_fields.at(sf.first)(i) << " ";
        }
        ss << "\n";
      }
      ss << "----------------------------\n";
    }
    if (!vector_fields.empty()) {
      for (const auto& vf : vector_fields) {
        ss << tabstr << vf.first << " vals: \n";
        for (Index i = 0; i < n(); ++i) {
          ss << "( ";
          for (Int j = 0; j < SeedType::geo::ndim; ++j) {
            ss << h_vector_fields.at(vf.first)(i, j) << " ";
          }
          ss << ")\n";
        }
      }
      ss << "----------------------------\n";
    }
  }
  return ss.str();
}

}  // namespace Lpm

#endif
