#ifndef LPM_INCOMPRESSIBLE2D_IMPL_HPP
#define LPM_INCOMPRESSIBLE2D_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_kernels.hpp"
#include "lpm_velocity_gallery.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

namespace Lpm {

template <typename SeedType>
Incompressible2D<SeedType>::Incompressible2D(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis, const Real velocity_eps, const std::vector<std::string>& tracers) :
  rel_vort_passive("relative_vorticity", mesh_params.nmaxverts),
  rel_vort_active("relative_vorticity", mesh_params.nmaxfaces),
  abs_vort_passive("absolute_vorticity", mesh_params.nmaxverts),
  abs_vort_active("absolute_vorticity", mesh_params.nmaxfaces),
  stream_fn_passive("stream_function", mesh_params.nmaxverts),
  stream_fn_active("stream_function", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  mesh(mesh_params),
  coriolis(coriolis),
  t(0),
  eps(velocity_eps)
{
  for (int k=0; k<tracers.size(); ++k) {
    tracer_passive.emplace(tracers[k], ScalarField<VertexField>(tracers[k], mesh_params.nmaxverts));
    tracer_active.emplace(tracers[k], ScalarField<FaceField>(tracers[k], mesh_params.nmaxfaces));
  }
}

template <typename SeedType>
void Incompressible2D<SeedType>::update_host() {
  mesh.update_host();
  rel_vort_passive.update_host();
  rel_vort_active.update_host();
  abs_vort_passive.update_host();
  abs_vort_active.update_host();
  stream_fn_passive.update_host();
  stream_fn_active.update_host();
  velocity_passive.update_host();
  velocity_active.update_host();
  for (const auto& tracer : tracer_passive) {
    tracer.second.update_host();
    tracer_active.at(tracer.first).update_host();
  }
}

template <typename SeedType>
void Incompressible2D<SeedType>::update_device() {
  mesh.update_device();
  rel_vort_passive.update_device();
  rel_vort_active.update_device();
  abs_vort_passive.update_device();
  abs_vort_active.update_device();
  stream_fn_passive.update_device();
  stream_fn_active.update_device();
  velocity_passive.update_device();
  velocity_active.update_device();
  for (const auto& tracer : tracer_passive) {
    tracer.second.update_device();
    tracer_active.at(tracer.first).update_device();
  }
}

template <typename SeedType> template <typename VorticityType>
void Incompressible2D<SeedType>::init_vorticity(const VorticityType& vorticity) {
  auto crds = mesh.vertices.phys_crds.view;
  auto relvort_view = rel_vort_passive.view;
  auto absvort_view = abs_vort_passive.view;
  auto cor = coriolis;
  Kokkos::parallel_for("Incompressible2D::init_vorticity, passive",
    mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real zeta = vorticity(mcrd);
      relvort_view(i) = zeta;
      absvort_view(i) = zeta + cor.f(mcrd);
    });
  crds = mesh.faces.phys_crds.view;
  relvort_view = rel_vort_active.view;
  absvort_view = abs_vort_active.view;
  Kokkos::parallel_for("Incompressible2D::init_vorticity, active",
    mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real zeta = vorticity(mcrd);
      relvort_view(i) = zeta;
      absvort_view(i) = zeta + cor.f(mcrd);
    });
}

template <typename SeedType>
void Incompressible2D<SeedType>::init_direct_sums() {
  Kokkos::TeamPolicy<> passive_policy(mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::TeamPolicy<> active_policy(mesh.n_faces_host(), Kokkos::AUTO());

  Kokkos::parallel_for("Incompressible2D::init_direct_sums, passive",
    passive_policy,
    Incompressible2DPassiveSums<geo>(velocity_passive.view,
      stream_fn_passive.view,
      mesh.vertices.phys_crds.view,
      mesh.faces.phys_crds.view,
      rel_vort_active.view,
      mesh.faces.area,
      mesh.faces.mask,
      eps,
      mesh.n_faces_host()));

  Kokkos::parallel_for("Incompressible2D::init_direct_sums, active",
    active_policy,
    Incompressible2DActiveSums<geo>(velocity_active.view,
      stream_fn_active.view,
      mesh.faces.phys_crds.view,
      rel_vort_active.view,
      mesh.faces.area,
      mesh.faces.mask,
      eps,
      mesh.n_faces_host()));
}

template <typename SeedType> template <typename VelocityType>
void Incompressible2D<SeedType>::init_velocity() {
  static_assert(
    std::is_same<typename SeedType::geo, typename VelocityType::geo>::value,
    "geometry types must match.");

  Kokkos::parallel_for(
      this->mesh.n_vertices_host(),
      VelocityKernel<VelocityType>(this->velocity_passive.view,
                                   this->mesh.vertices.phys_crds.view, t));
  Kokkos::parallel_for(
      this->mesh.n_faces_host(),
      VelocityKernel<VelocityType>(this->velocity_active.view,
                                   this->mesh.faces.phys_crds.view, t));
}

template <typename SeedType> template <typename SolverType>
void Incompressible2D<SeedType>::advance_timestep(SolverType& solver) {
  solver.advance_timestep_impl();
  t = solver.t_idx * solver.dt;
}

template <typename SeedType>
Int Incompressible2D<SeedType>::n_tracers() const {
  LPM_ASSERT(tracer_active.size() == tracer_passive.size());
  return tracer_active.size();
}

#ifdef LPM_USE_VTK
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const Incompressible2D<SeedType>& ic2d) {
    VtkPolymeshInterface<SeedType> vtk(ic2d.mesh);
    vtk.add_scalar_point_data(ic2d.rel_vort_passive.view);
    vtk.add_scalar_point_data(ic2d.stream_fn_passive.view);
    vtk.add_vector_point_data(ic2d.velocity_passive.view);
    vtk.add_scalar_cell_data(ic2d.rel_vort_active.view);
    vtk.add_scalar_cell_data(ic2d.stream_fn_active.view);
    vtk.add_vector_cell_data(ic2d.velocity_active.view);
    for (const auto& tracer : ic2d.tracer_passive) {
      vtk.add_scalar_point_data(tracer.second.view, tracer.first);
      vtk.add_scalar_cell_data(ic2d.tracer_active.at(tracer.first).view, tracer.first);
    }
    return vtk;
  }
#endif

} // namespace Lpm

#endif
