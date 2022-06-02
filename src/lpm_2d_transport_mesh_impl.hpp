#include "lpm_2d_transport_mesh.hpp"
#include "util/lpm_string_util.hpp"
#include <sstream>
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

namespace Lpm {

template <typename SeedType> template <typename VelocityType>
void TransportMesh2d<SeedType>::initialize_velocity() {
  static_assert(std::is_same<typename SeedType::geo, typename VelocityType::geo>::value,
    "Geometry types must match.");

  Kokkos::parallel_for(this->vertices.nh(),
    VelocityKernel<VelocityType>(this->velocity_verts.view, this->vertices.phys_crds->crds, 0)
  );
  Kokkos::parallel_for(this->faces.nh(),
    VelocityKernel<VelocityType>(this->velocity_faces.view, this->faces.phys_crds->crds, 0)
  );
}

template <typename SeedType> template <typename ICType>
void TransportMesh2d<SeedType>::initialize_tracer(const ICType& tracer_ic) {
  static_assert(std::is_same<typename SeedType::geo, typename ICType::geo>::value,
    "Geometry types must match.");

  tracer_verts.emplace(tracer_ic.name(),
    ScalarField<VertexField>(tracer_ic.name(), this->velocity_verts.view.extent(0)));
  tracer_faces.emplace(tracer_ic.name(),
    ScalarField<FaceField>(tracer_ic.name(), this->velocity_faces.view.extent(0)));

  auto vertview = tracer_verts.at(tracer_ic.name()).view;
  auto faceview = tracer_faces.at(tracer_ic.name()).view;
  Kokkos::parallel_for(this->vertices.nh(), KOKKOS_LAMBDA (const Index i) {
    const auto crd = Kokkos::subview(this->vertices.lag_crds->crds, i, Kokkos::ALL);
    vertview(i) = tracer_ic(crd);
  });
  Kokkos::parallel_for(this->faces.nh(), KOKKOS_LAMBDA (const Index i) {
    const auto crd = Kokkos::subview(this->faces.lag_crds->crds, i, Kokkos::ALL);
    faceview(i) = tracer_ic(crd);
  });
}

template <typename SeedType>
std::string TransportMesh2d<SeedType>::info_string(const int tab_lev) const {
  const std::string label = "\nTransportMesh2d<" + SeedType::id_string() + "> info:\n";
  const int indent = tab_lev + 1;
  const bool dump_all = false;
  const std::string tabstr = indent_string(tab_lev);
  std::ostringstream ss;
  ss << label;
  ss << this->PolyMesh2d<SeedType>::info_string(std::string(), indent, dump_all);
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
VtkPolymeshInterface<SeedType> vtk_interface(const std::shared_ptr<TransportMesh2d<SeedType>> tm) {
  std::shared_ptr<PolyMesh2d<SeedType>> base_ptr(tm);
  VtkPolymeshInterface<SeedType> vtk(base_ptr);
  vtk.add_vector_point_data(tm->velocity_verts.view, "velocity");
  vtk.add_vector_cell_data(tm->velocity_faces.view, "velocity");
  for (const auto& tic : tm->tracer_verts) {
    vtk.add_scalar_point_data(tic.second.view, tic.first);
    vtk.add_scalar_cell_data(tm->tracer_faces.at(tic.first).view, tic.first);
  }
  return vtk;
}
#endif

} // namespace lpm
