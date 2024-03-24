#ifndef LPM_SWE_IMPL_HPP
#define LPM_SWE_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "lpm_swe.hpp"

namespace Lpm {

template <typename InitialCondition>
    struct SWEInitializeProblem {
      view_2d<Real>  lag_crds;
      scalar_view_type rel_vort;
      scalar_view_type pot_vort;
      scalar_view_type divergence;
      scalar_view_type surface_height;
      scalar_view_type depth;
      view_2d<Real> velocity;
      InitialCondition initial;

      SWEInitializeProblem(view_2d<Real> lx, scalar_view_type zeta, scalar_view_type Q, scalar_view_type sigma, scalar_view_type s,
        scalar_view_type h, view_2d<Real> u, const InitialCondition ic) :
        lag_crds(lx),
        rel_vort(zeta),
        pot_vort(Q),
        divergence(sigma),
        surface_height(s),
        depth(h),
        velocity(u),
        initial(ic)
        {}

      KOKKOS_INLINE_FUNCTION
      void operator() (const Index i) const {
        const auto mcrd = Kokkos::subview(lag_crds, i, Kokkos::ALL);
        const auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
        rel_vort(i) = initial.zeta0(mcrd);
        divergence(i) = initial.sigma0(mcrd);
        const Real s = initial.sfc0(mcrd);
        surface_height(i) = s;
        depth(i) = s - initial.bottom_height(mcrd);
        if constexpr (std::is_same<typename InitialCondition::geo, PlaneGeometry>::value) {
            pot_vort(i) = (rel_vort(i) + coriolis_f<typename InitialCondition::geo>(initial.f0, initial.beta, mcrd[1])) / depth(i);
        }
        else {
            pot_vort(i) = (rel_vort(i) + coriolis_f<typename InitialCondition::geo>(initial.Omega, mcrd[2])) /  depth(i);
        }
        initial.u0(mvel, mcrd);
      }
};


template <typename SeedType>
SWE<SeedType>::SWE(const PolyMeshParameters<SeedType>& mesh_params, const Real Omg) :
  rel_vort_passive("relative_vorticity", mesh_params.nmaxverts),
  rel_vort_active("relative_vorticity", mesh_params.nmaxfaces),
  pot_vort_passive("potential_vorticity", mesh_params.nmaxverts),
  pot_vort_active("potential_vorticity", mesh_params.nmaxfaces),
  div_passive("divergence", mesh_params.nmaxverts),
  div_active("divergence", mesh_params.nmaxfaces),
  surf_passive("surface_height", mesh_params.nmaxverts),
  surf_active("surface_height", mesh_params.nmaxfaces),
  surf_lap_passive("surface_laplacian", mesh_params.nmaxverts),
  surf_lap_active("surface_laplaican", mesh_params.nmaxfaces),
  bottom_passive("bottom_height", mesh_params.nmaxverts),
  bottom_active("bottom_height", mesh_params.nmaxfaces),
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  double_dot_passive("double_dot", mesh_params.nmaxverts),
  double_dot_active("double_dot", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  Omega(Omg),
  g(1),
  t(0) {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
    "spherical geometry required");
    }

template <typename SeedType>
SWE<SeedType>::SWE(const PolyMeshParameters<SeedType>& mesh_params, const Real f, const Real b) :
  rel_vort_passive("relative_vorticity", mesh_params.nmaxverts),
  rel_vort_active("relative_vorticity", mesh_params.nmaxfaces),
  pot_vort_passive("potential_vorticity", mesh_params.nmaxverts),
  pot_vort_active("potential_vorticity", mesh_params.nmaxfaces),
  div_passive("divergence", mesh_params.nmaxverts),
  div_active("divergence", mesh_params.nmaxfaces),
  surf_passive("surface_height", mesh_params.nmaxverts),
  surf_active("surface_height", mesh_params.nmaxfaces),
  surf_lap_passive("surface_laplacian", mesh_params.nmaxverts),
  surf_lap_active("surface_laplaican", mesh_params.nmaxfaces),
  bottom_passive("bottom_height", mesh_params.nmaxverts),
  bottom_active("bottom_height", mesh_params.nmaxfaces),
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  double_dot_passive("double_dot", mesh_params.nmaxverts),
  double_dot_active("double_dot", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  f0(f),
  beta(b),
  g(1),
  t(0) {
  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "planar geometry required");
  }


template <typename SeedType>
void SWE<SeedType>::update_host() {
  rel_vort_passive.update_host();
  rel_vort_active.update_host();
  pot_vort_passive.update_host();
  pot_vort_active.update_host();
  div_passive.update_host();
  div_active.update_host();
  surf_passive.update_host();
  surf_active.update_host();
  surf_lap_passive.update_host();
  surf_lap_active.update_host();
  bottom_passive.update_host();
  bottom_active.update_host();
  depth_passive.update_host();
  depth_active.update_host();
  velocity_passive.update_host();
  velocity_active.update_host();
  double_dot_passive.update_host();
  double_dot_active.update_host();
  mass_active.update_host();
  mesh.update_host();
}

template <typename SeedType>
void SWE<SeedType>::update_device() {
  rel_vort_passive.update_device();
  rel_vort_active.update_device();
  pot_vort_passive.update_device();
  pot_vort_active.update_device();
  div_passive.update_device();
  div_active.update_device();
  surf_passive.update_device();
  surf_active.update_device();
  surf_lap_passive.update_device();
  surf_lap_active.update_device();
  bottom_passive.update_device();
  bottom_active.update_device();
  depth_passive.update_device();
  depth_active.update_device();
  velocity_passive.update_device();
  velocity_active.update_device();
  double_dot_passive.update_device();
  double_dot_active.update_device();
  mass_active.update_device();
  mesh.update_device();
}

template <typename SeedType> template <typename InitialCondition>
void SWE<SeedType>::init_swe_problem(const InitialCondition& ic) {
  Kokkos::parallel_for("init_swe_problem (passive)", mesh.n_vertices_host(),
    SWEInitializeProblem(mesh.vertices.lag_crds.view,
    rel_vort_passive.view,
    pot_vort_passive.view,
    div_passive.view,
    surf_passive.view,
    depth_passive.view,
    velocity_passive.view,
    ic));

  Kokkos::parallel_for("init_swe_problem (active)", mesh.n_faces_host(),
    SWEInitializeProblem(mesh.faces.lag_crds.view,
      rel_vort_active.view,
      pot_vort_active.view,
      div_active.view,
      surf_active.view,
      depth_active.view,
      velocity_active.view,
      ic));
}

template <typename SeedType> template <typename TopoType>
void SWE<SeedType>::set_bottom_topography(const TopoType& topo) {
  auto crds = mesh.vertices.phys_crds.view;
  auto topo_view = bottom_passive.view;
  Kokkos::parallel_for(mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      topo_view(i) = topo(mcrd);
    });

  crds = mesh.faces.phys_crds.view;
  topo_view = bottom_active.view;
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      topo_view(i) = topo(mcrd);
    });
}

template <typename SeedType> template <typename BottomType, typename SurfaceType>
void SWE<SeedType>::init_surface(const BottomType& topo, const SurfaceType& sfc) {
  auto crds = mesh.vertices.phys_crds.view;
  auto topo_view = bottom_passive.view;
  auto sfc_view = surf_passive.view;
  auto depth_view = depth_passive.view;
  Kokkos::parallel_for(mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real b = topo(mcrd);
      const Real s = sfc(mcrd);
      topo_view(i) = b;
      sfc_view(i) = s;
      depth_view(i) = s - b;
    });

  crds = mesh.faces.phys_crds.view;
  topo_view = bottom_active.view;
  sfc_view = surf_active.view;
  depth_view = depth_active.view;
  auto mass_view = mass_active.view;
  auto area_view = mesh.faces.area;
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real b = topo(mcrd);
      const Real s = sfc(mcrd);
      topo_view(i) = b;
      sfc_view(i) = s;
      depth_view(i) = s - b;
      mass_view(i) = (s - b) * area_view(i);
    });
}

template <typename SeedType> template <typename SolverType>
void SWE<SeedType>::advance_timestep(SolverType& solver) {
  solver.advance_timestep();
  t = solver.t_idx * solver.dt;
}

#ifdef LPM_USE_VTK
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const SWE<SeedType>& swe) {

  VtkPolymeshInterface<SeedType> vtk(swe.mesh, swe.surf_passive.hview);
  vtk.add_scalar_point_data(swe.rel_vort_passive.view);
  vtk.add_scalar_point_data(swe.pot_vort_passive.view);
  vtk.add_scalar_point_data(swe.div_passive.view);
  vtk.add_scalar_point_data(swe.surf_passive.view);
  vtk.add_scalar_point_data(swe.surf_lap_passive.view);
  vtk.add_scalar_point_data(swe.depth_passive.view);
  vtk.add_scalar_point_data(swe.double_dot_passive.view);
  vtk.add_scalar_point_data(swe.bottom_passive.view);
  vtk.add_vector_point_data(swe.velocity_passive.view);
  vtk.add_scalar_cell_data(swe.rel_vort_active.view);
  vtk.add_scalar_cell_data(swe.pot_vort_active.view);
  vtk.add_scalar_cell_data(swe.div_active.view);
  vtk.add_scalar_cell_data(swe.surf_active.view);
  vtk.add_scalar_cell_data(swe.depth_active.view);
  vtk.add_scalar_cell_data(swe.surf_lap_active.view);
  vtk.add_scalar_cell_data(swe.double_dot_active.view);
  vtk.add_scalar_cell_data(swe.bottom_active.view);
  vtk.add_vector_cell_data(swe.velocity_active.view);
  vtk.add_scalar_cell_data(swe.mass_active.view);
  return vtk;
  }
#endif

} // namespace Lpm

#endif
