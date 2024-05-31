#ifndef LPM_INCOMPRESSIBLE2D_IMPL_HPP
#define LPM_INCOMPRESSIBLE2D_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_kernels.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "mesh/lpm_bivar_remesh_impl.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

namespace Lpm {

template <typename SeedType>
Incompressible2D<SeedType>::Incompressible2D(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis, const Real velocity_eps) : //, const std::vector<std::string>& tracers) :
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
  logger = lpm_logger();
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
  return init_vorticity(vorticity,
    0, mesh.n_vertices_host(),
    0, mesh.n_faces_host());
}

template <typename SeedType>
Real Incompressible2D<SeedType>::total_vorticity() const {
  Real total_vort;
  const auto zeta_view = rel_vort_active.view;
  const auto area_view = mesh.faces.area;
  const auto mask_view = mesh.faces.mask;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i, Real& sum) {
      sum += (mask_view(i) ? 0 : zeta_view(i)*area_view(i));
    }, total_vort);
  return total_vort;
}

template <typename SeedType> template <typename VorticityType>
void Incompressible2D<SeedType>::init_vorticity(const VorticityType& vorticity,
  const Index vert_start_idx, const Index vert_end_idx,
  const Index face_start_idx, const Index face_end_idx) {

  // passive particles
  auto crds = mesh.vertices.phys_crds.view;
  auto relvort_view = rel_vort_passive.view;
  auto absvort_view = abs_vort_passive.view;
  auto cor = coriolis;
  Kokkos::parallel_for("Incompressible2D::init_vorticity, passive",
    Kokkos::RangePolicy(vert_start_idx, vert_end_idx),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real zeta = vorticity(mcrd);
      relvort_view(i) = zeta;
      absvort_view(i) = zeta + cor.f(mcrd);
    });
  // active particles
  crds = mesh.faces.phys_crds.view;
  relvort_view = rel_vort_active.view;
  absvort_view = abs_vort_active.view;
  Kokkos::parallel_for("Incompressible2D::init_vorticity, active",
    Kokkos::RangePolicy(face_start_idx, face_end_idx),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real zeta = vorticity(mcrd);
      relvort_view(i) = zeta;
      absvort_view(i) = zeta + cor.f(mcrd);
    });
}

template <typename SeedType> template <typename TracerType>
void Incompressible2D<SeedType>::init_tracer(const TracerType& tracer, const std::string& tname) {
  static_assert(std::is_same<typename SeedType::geo,
      typename TracerType::geo>::value, "Geometry types must match.");

  const std::string name = (tname.empty() ? tracer.name() : tname);
  tracer_passive.emplace(name,
    ScalarField<VertexField>(name, velocity_passive.view.extent(0)));
  tracer_active.emplace(name,
    ScalarField<FaceField>(name, velocity_active.view.extent(0)));

  if constexpr (std::is_same<TracerType,
    FtleTracer<typename SeedType::geo>>::value) {
    Kokkos::deep_copy(tracer_passive.at(name).view, 1);
    Kokkos::deep_copy(tracer_active.at(name).view, 1);
  }
  else {
    auto tracer_view = tracer_passive.at(name).view;
    auto lag_crd_view = mesh.vertices.lag_crds.view;
    Kokkos::parallel_for(mesh.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mcrd = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
        tracer_view(i) = tracer(mcrd);
      });

    tracer_view = tracer_active.at(name).view;
    lag_crd_view = mesh.faces.lag_crds.view;
    Kokkos::parallel_for(mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mcrd = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
        tracer_view(i) = tracer(mcrd);
      });
  }
}

template <typename SeedType> template <typename TracerType>
void Incompressible2D<SeedType>::allocate_tracer(const TracerType& tracer, const std::string& tname) {
static_assert(std::is_same<typename SeedType::geo,
      typename TracerType::geo>::value, "Geometry types must match.");

  const std::string name = (tname.empty() ? tracer.name() : tname);
  tracer_passive.emplace(name,
    ScalarField<VertexField>(name, velocity_passive.view.extent(0)));
  tracer_active.emplace(name,
    ScalarField<FaceField>(name, velocity_active.view.extent(0)));
}

template <typename SeedType>
void Incompressible2D<SeedType>::allocate_tracer(const std::string& name) {
  tracer_passive.emplace(name, ScalarField<VertexField>(name, velocity_passive.view.extent(0)));
  tracer_active.emplace(name, ScalarField<FaceField>(name, velocity_active.view.extent(0)));
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

template <typename SeedType>
std::string Incompressible2D<SeedType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  const std::string label = "Incompressible2D<" + SeedType::id_string() + "> info:\n";
  ss << mesh.info_string(label, tab_level);
  ss << rel_vort_active.info_string(tab_level+1);
  ss << velocity_active.info_string(tab_level+1);
  return ss.str();
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

template <typename SeedType>
BivarRemesh<SeedType> bivar_remesh(Incompressible2D<SeedType>& new_ic2d,
  const Incompressible2D<SeedType>& old_ic2d) {

  using passive_scalar_field_map = std::map<std::string, ScalarField<VertexField>>;
  using active_scalar_field_map = std::map<std::string, ScalarField<FaceField>>;
  using passive_vector_field_map = std::map<std::string, VectorField<PlaneGeometry, VertexField>>;
  using active_vector_field_map = std::map<std::string, VectorField<PlaneGeometry, FaceField>>;

  passive_scalar_field_map passive_scalars_old;
  passive_scalars_old.emplace("relative_vorticity", old_ic2d.rel_vort_passive);
  passive_scalars_old.emplace("absolute_vorticity", old_ic2d.abs_vort_passive);
  passive_scalars_old.emplace("stream_function", old_ic2d.stream_fn_passive);
  for (const auto& t : old_ic2d.tracer_passive) {
    passive_scalars_old.emplace(t.first, t.second);
  }

  passive_scalar_field_map passive_scalars_new;
  passive_scalars_new.emplace("relative_vorticity", new_ic2d.rel_vort_passive);
  passive_scalars_new.emplace("absolute_vorticity", new_ic2d.abs_vort_passive);
  passive_scalars_new.emplace("stream_function", new_ic2d.stream_fn_passive);
  for (const auto& t : new_ic2d.tracer_passive) {
    passive_scalars_new.emplace(t.first, t.second);
  }

  active_scalar_field_map active_scalars_old;
  active_scalars_old.emplace("relative_vorticity", old_ic2d.rel_vort_active);
  active_scalars_old.emplace("absolute_vorticity", old_ic2d.abs_vort_active);
  active_scalars_old.emplace("stream_function", old_ic2d.stream_fn_active);
  for (const auto& t : old_ic2d.tracer_active) {
    active_scalars_old.emplace(t.first, t.second);
  }
  active_scalar_field_map active_scalars_new;
  active_scalars_new.emplace("relative_vorticity", new_ic2d.rel_vort_active);
  active_scalars_new.emplace("absolute_vorticity", new_ic2d.abs_vort_active);
  active_scalars_new.emplace("stream_function", new_ic2d.stream_fn_active);
  for (const auto& t : new_ic2d.tracer_active) {
    active_scalars_new.emplace(t.first, t.second);
  }

  passive_vector_field_map passive_vectors_old;
  passive_vectors_old.emplace("velocity", old_ic2d.velocity_passive);

  active_vector_field_map active_vectors_old;
  active_vectors_old.emplace("velocity", old_ic2d.velocity_active);

  passive_vector_field_map passive_vectors_new;
  passive_vectors_new.emplace("velocity", new_ic2d.velocity_passive);

  active_vector_field_map active_vectors_new;
  active_vectors_new.emplace("velocity", new_ic2d.velocity_active);

  BivarRemesh<SeedType> bivar(new_ic2d.mesh,
                           passive_scalars_new,
                           active_scalars_new,
                           passive_vectors_new,
                           active_vectors_new,
                           old_ic2d.mesh,
                           passive_scalars_old,
                           active_scalars_old,
                           passive_vectors_old,
                           active_vectors_old);

  return bivar;
}

template <typename SeedType>
CompadreRemesh<SeedType> compadre_remesh(Incompressible2D<SeedType>& new_ic2d,
  const Incompressible2D<SeedType>& old_ic2d, const gmls::Params& gmls_params) {

  using passive_scalar_field_map = std::map<std::string, ScalarField<VertexField>>;
  using active_scalar_field_map = std::map<std::string, ScalarField<FaceField>>;
  using passive_vector_field_map = std::map<std::string, VectorField<typename SeedType::geo, VertexField>>;
  using active_vector_field_map = std::map<std::string, VectorField<typename SeedType::geo, FaceField>>;

  passive_scalar_field_map passive_scalars_old;
  passive_scalars_old.emplace("relative_vorticity", old_ic2d.rel_vort_passive);
  passive_scalars_old.emplace("absolute_vorticity", old_ic2d.abs_vort_passive);
  passive_scalars_old.emplace("stream_function", old_ic2d.stream_fn_passive);
  for (const auto& t : old_ic2d.tracer_passive) {
    passive_scalars_old.emplace(t.first, t.second);
  }

  passive_scalar_field_map passive_scalars_new;
  passive_scalars_new.emplace("relative_vorticity", new_ic2d.rel_vort_passive);
  passive_scalars_new.emplace("absolute_vorticity", new_ic2d.abs_vort_passive);
  passive_scalars_new.emplace("stream_function", new_ic2d.stream_fn_passive);
  for (const auto& t : new_ic2d.tracer_passive) {
    passive_scalars_new.emplace(t.first, t.second);
  }

  active_scalar_field_map active_scalars_old;
  active_scalars_old.emplace("relative_vorticity", old_ic2d.rel_vort_active);
  active_scalars_old.emplace("absolute_vorticity", old_ic2d.abs_vort_active);
  active_scalars_old.emplace("stream_function", old_ic2d.stream_fn_active);
  for (const auto& t : old_ic2d.tracer_active) {
    active_scalars_old.emplace(t.first, t.second);
  }
  active_scalar_field_map active_scalars_new;
  active_scalars_new.emplace("relative_vorticity", new_ic2d.rel_vort_active);
  active_scalars_new.emplace("absolute_vorticity", new_ic2d.abs_vort_active);
  active_scalars_new.emplace("stream_function", new_ic2d.stream_fn_active);
  for (const auto& t : new_ic2d.tracer_active) {
    active_scalars_new.emplace(t.first, t.second);
  }

  passive_vector_field_map passive_vectors_old;
  passive_vectors_old.emplace("velocity", old_ic2d.velocity_passive);

  active_vector_field_map active_vectors_old;
  active_vectors_old.emplace("velocity", old_ic2d.velocity_active);

  passive_vector_field_map passive_vectors_new;
  passive_vectors_new.emplace("velocity", new_ic2d.velocity_passive);

  active_vector_field_map active_vectors_new;
  active_vectors_new.emplace("velocity", new_ic2d.velocity_active);

  return CompadreRemesh<SeedType>(new_ic2d.mesh,
    passive_scalars_new,
    active_scalars_new,
    passive_vectors_new,
    active_vectors_new,
    old_ic2d.mesh,
    passive_scalars_old,
    active_scalars_old,
    passive_vectors_old,
    active_vectors_old,
    gmls_params);
}

} // namespace Lpm

#endif
