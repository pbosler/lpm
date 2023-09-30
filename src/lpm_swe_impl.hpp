#ifndef LPM_SWE_IMPL_HPP
#define LPM_SWE_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "lpm_swe.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename Geo>
struct KineticEnergyFunctor {
  scalar_view_type depth;
  typename Geo::vec_view_type velocity;
  scalar_view_type area;
  mask_view_type mask;

  KineticEnergyFunctor(scalar_view_type h, typename Geo::vec_view_type v, scalar_view_type a, mask_view_type m) :
    depth(h), velocity(v), area(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i, Real& ke) const {
    if (not mask(i)) {
      const auto mvec = Kokkos::subview(velocity, i, Kokkos::ALL);
      ke += depth(i)*Geo::norm2(mvec)*area(i);
    }
  }
};

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
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  Omega(Omg),
  t(0),
  g(constants::GRAVITY) {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
    "spherical geometry required");
    coriolis.Omega = Omega;
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
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  f0(f),
  beta(b),
  t(0),
  g(constants::GRAVITY) {
  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "planar geometry required");
    coriolis.f0 = f;
    coriolis.beta = b;
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
  depth_passive.update_host();
  depth_active.update_host();
  velocity_passive.update_host();
  velocity_active.update_host();
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
  depth_passive.update_device();
  depth_active.update_device();
  velocity_passive.update_device();
  velocity_active.update_device();
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

  auto mass = mass_active.view;
  const auto area = mesh.faces.area;
  const auto h = depth_active.view;
  const auto mask = mesh.faces.mask;
  Kokkos::parallel_for("init mass", mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      if (not mask(i)) {
        mass(i) = h(i) * area(i);
      }
      else {
        mass(i) = 0;
      }
    });
}

template <typename SeedType>
Real SWE<SeedType>::total_mass() const {
  Real mass;
  const auto mass_view = mass_active.view;
  const auto mask = mesh.faces.mask;
  Kokkos::parallel_reduce("SWE total mass", mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i, Real& m) {
      if (not mask(i)) {
        m += mass_view(i);
      }
    }, mass);
  return mass;
}

template <typename SeedType>
Real SWE<SeedType>::total_energy() const {
  return kinetic_energy() + potential_energy();
}

template <typename SeedType>
Real SWE<SeedType>::total_enstrophy() const {
  Real ens;
  Kokkos::parallel_reduce("SWE enstrophy", mesh.n_faces_host(),
    ScalarSquareIntegralFunctor(rel_vort_active, mesh.faces.area, mesh.faces.mask), ens);
  return ens;
}

template <typename SeedType>
Real SWE<SeedType>::kinetic_energy() const {
  Real ke;
  Kokkos::parallel_reduce("SWE kinetic energy", mesh.n_faces_host(),
    KineticEnergyFunctor(velocity_active, mesh.faces.area, mesh.faces.mask), ke);
  return 0.5*ke;
}

template <typename SeedType>
Real SWE<SeedType>::potential_energy() const {
  Real pe;
  Kokkos::parallel_reduce("SWE potential energy", mesh.n_faces_host(),
    ScalarSquareIntegralFunctor(depth_active, mesh.faces.area, mesh.faces.mask),
    pe);
  return 0.5 * g * pe;
}


#ifdef LPM_USE_VTK
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const SWE<SeedType>& swe) {
  VtkPolymeshInterface<SeedType> vtk(swe.mesh);
  vtk.add_scalar_point_data(swe.rel_vort_passive.view);
  vtk.add_scalar_point_data(swe.pot_vort_passive.view);
  vtk.add_scalar_point_data(swe.div_passive.view);
  vtk.add_scalar_point_data(swe.surf_passive.view);
  vtk.add_scalar_point_data(swe.depth_passive.view);
  vtk.add_vector_point_data(swe.velocity_passive.view);
  vtk.add_scalar_cell_data(swe.rel_vort_active.view);
  vtk.add_scalar_cell_data(swe.pot_vort_active.view);
  vtk.add_scalar_cell_data(swe.div_active.view);
  vtk.add_scalar_cell_data(swe.surf_active.view);
  vtk.add_scalar_cell_data(swe.depth_active.view);
  vtk.add_vector_cell_data(swe.velocity_active.view);
  return vtk;
  }
#endif

} // namespace Lpm

#endif
