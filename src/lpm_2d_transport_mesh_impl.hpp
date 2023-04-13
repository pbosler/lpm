#include <sstream>

#include "lpm_2d_transport_mesh.hpp"
#include "util/lpm_string_util.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

namespace Lpm {

template <typename SeedType>
template <typename VelocityType>
void TransportMesh2d<SeedType>::initialize_velocity() {
  static_assert(
      std::is_same<typename SeedType::geo, typename VelocityType::geo>::value,
      "Geometry types must match.");

  Kokkos::parallel_for(
      this->vertices.nh(),
      VelocityKernel<VelocityType>(this->velocity_verts.view,
                                   this->vertices.phys_crds.view, 0));
  Kokkos::parallel_for(this->faces.nh(), VelocityKernel<VelocityType>(
                                             this->velocity_faces.view,
                                             this->faces.phys_crds.view, 0));
}

template <typename SeedType>
template <typename VelocityType>
void TransportMesh2d<SeedType>::set_velocity(const Real t,
                                             const Index vert_start_idx,
                                             const Index face_start_idx) {
  static_assert(
      std::is_same<typename SeedType::geo, typename VelocityType::geo>::value,
      "Geometry types must match.");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(vert_start_idx, this->vertices.nh()),
      VelocityKernel<VelocityType>(this->velocity_verts.view,
                                   this->vertices.phys_crds.view, t));
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(face_start_idx, this->faces.nh()),
      VelocityKernel<VelocityType>(this->velocity_faces.view,
                                   this->faces.phys_crds.view, t));
}

template <typename SeedType>
template <typename ICType>
void TransportMesh2d<SeedType>::initialize_tracer(const ICType& tracer_ic) {
  static_assert(
      std::is_same<typename SeedType::geo, typename ICType::geo>::value,
      "Geometry types must match.");

  tracer_verts.emplace(
      tracer_ic.name(),
      ScalarField<VertexField>(tracer_ic.name(),
                               this->velocity_verts.view.extent(0)));
  tracer_faces.emplace(
      tracer_ic.name(),
      ScalarField<FaceField>(tracer_ic.name(),
                             this->velocity_faces.view.extent(0)));

  auto vertview = tracer_verts.at(tracer_ic.name()).view;
  auto faceview = tracer_faces.at(tracer_ic.name()).view;
  const auto vlags = this->vertices.lag_crds.view;
  const auto flags = this->faces.lag_crds.view;
  Kokkos::parallel_for(
      this->vertices.nh(), KOKKOS_LAMBDA(const Index i) {
        const auto crd = Kokkos::subview(vlags, i, Kokkos::ALL);
        vertview(i) = tracer_ic(crd);
      });
  Kokkos::parallel_for(
      this->faces.nh(), KOKKOS_LAMBDA(const Index i) {
        const auto crd = Kokkos::subview(flags, i, Kokkos::ALL);
        faceview(i) = tracer_ic(crd);
      });
}

template <typename SeedType>
template <typename ICType>
void TransportMesh2d<SeedType>::set_tracer_from_lag_crds(const ICType& tracer_ic,
                                                  const Index vert_start_idx,
                                                  const Index face_start_idx) {
  static_assert(
      std::is_same<typename SeedType::geo, typename ICType::geo>::value,
      "Geometry types must match.");

  auto vertview = tracer_verts.at(tracer_ic.name()).view;
  auto faceview = tracer_faces.at(tracer_ic.name()).view;
  const auto vlag_crds = this->vertices.lag_crds.view;
  const auto flag_crds = this->faces.lag_crds.view;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(vert_start_idx, this->vertices.nh()),
      KOKKOS_LAMBDA(const Index i) {
        const auto crd = Kokkos::subview(vlag_crds, i, Kokkos::ALL);
        vertview(i) = tracer_ic(crd);
      });
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(face_start_idx, this->faces.nh()),
      KOKKOS_LAMBDA(const Index i) {
        const auto crd = Kokkos::subview(flag_crds, i, Kokkos::ALL);
        faceview(i) = tracer_ic(crd);
      });
}

template <typename SeedType>
void TransportMesh2d<SeedType>::allocate_scalar_tracer(
    const std::string name) {
  tracer_verts.emplace(name, ScalarField<VertexField>(
                                 name, this->velocity_verts.view.extent(0)));
  tracer_faces.emplace(
      name, ScalarField<FaceField>(name, this->velocity_faces.view.extent(0)));
}

template <typename SeedType>
std::string TransportMesh2d<SeedType>::info_string(const int tab_lev) const {
  const std::string label =
      "\nTransportMesh2d<" + SeedType::id_string() + "> info:\n";
  const int indent = tab_lev + 1;
  const bool dump_all = false;
  const std::string tabstr = indent_string(tab_lev);
  std::ostringstream ss;
  ss << label;
  ss << this->PolyMesh2d<SeedType>::info_string(std::string(), indent,
                                                dump_all);
  ss << tabstr << "ntracers = " << ntracers() << "\n";
  for (const auto& tracer : this->tracer_verts) {
    ss << tabstr << "\t" << tracer.first << "\n";
  }
  return ss.str();
}

template <typename SeedType>
void TransportMesh2d<SeedType>::update_device() const {
  PolyMesh2d<SeedType>::update_device();
  for (const auto& f : tracer_verts) {
    f.second.update_device();
  }
  for (const auto& f : tracer_faces) {
    f.second.update_device();
  }
}

template <typename SeedType>
void TransportMesh2d<SeedType>::update_host() const {
  PolyMesh2d<SeedType>::update_host();
  for (const auto& f : tracer_verts) {
    f.second.update_host();
  }
  for (const auto& f : tracer_faces) {
    f.second.update_host();
  }
}

#ifdef LPM_USE_VTK
template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(
    const TransportMesh2d<SeedType>& tm) {
  VtkPolymeshInterface<SeedType> vtk(tm);
  vtk.add_vector_point_data(tm.velocity_verts.view, "velocity");
  vtk.add_vector_cell_data(tm.velocity_faces.view, "velocity");
  for (const auto& tic : tm.tracer_verts) {
    vtk.add_scalar_point_data(tic.second.view, tic.first);
    vtk.add_scalar_cell_data(tm.tracer_faces.at(tic.first).view, tic.first);
  }
  return vtk;
}
#endif

}  // namespace Lpm
