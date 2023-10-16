#ifndef LPM_SWE_IMPL_HPP
#define LPM_SWE_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_kernels.hpp"
#include "lpm_swe_tendencies.hpp"
#include "lpm_velocity_gallery.hpp"

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
std::string SWE<SeedType>::info_string(const int tab_level, const bool verbose) const {
  return mesh.info_string();
}

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
  surf_laplacian_passive("surface_laplacian", mesh_params.nmaxverts),
  surf_laplacian_active("surface_laplacian", mesh_params.nmaxfaces),
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  double_dot_passive("double_dot", mesh_params.nmaxverts),
  double_dot_active("double_dot", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  Omega(Omg),
  t(0),
  g(constants::GRAVITY),
  eps(0) {
    coriolis.Omega = Omega;
  }

template <typename SeedType>
SWE<SeedType>::SWE(const PolyMeshParameters<SeedType>& mesh_params) :
  rel_vort_passive("relative_vorticity", mesh_params.nmaxverts),
  rel_vort_active("relative_vorticity", mesh_params.nmaxfaces),
  pot_vort_passive("potential_vorticity", mesh_params.nmaxverts),
  pot_vort_active("potential_vorticity", mesh_params.nmaxfaces),
  div_passive("divergence", mesh_params.nmaxverts),
  div_active("divergence", mesh_params.nmaxfaces),
  surf_passive("surface_height", mesh_params.nmaxverts),
  surf_active("surface_height", mesh_params.nmaxfaces),
  surf_laplacian_passive("surface_laplacian", mesh_params.nmaxverts),
  surf_laplacian_active("surface_laplacian", mesh_params.nmaxfaces),
  depth_passive("depth", mesh_params.nmaxverts),
  depth_active("depth", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  double_dot_passive("double_dot", mesh_params.nmaxverts),
  double_dot_active("double_dot", mesh_params.nmaxfaces),
  mass_active("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  t(0),
  g(constants::GRAVITY),
  eps(0) {}


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
  surf_laplacian_passive.update_host();
  surf_laplacian_active.update_host();
  depth_passive.update_host();
  depth_active.update_host();
  velocity_passive.update_host();
  velocity_active.update_host();
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
  surf_laplacian_passive.update_device();
  surf_laplacian_active.update_device();
  depth_passive.update_device();
  depth_active.update_device();
  velocity_passive.update_device();
  velocity_active.update_device();
  mass_active.update_device();
  mesh.update_device();
}

template <typename SeedType> template <typename InitialCondition, typename SurfaceLaplacianType>
void SWE<SeedType>::init_swe_problem(const InitialCondition& ic, SurfaceLaplacianType& lap) {
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
  std::cout << "initializing velocity and double dot product.\n";
  init_velocity(); // to compute double dot
  std::cout << "initializing surface.\n";
  lap.update_src_data(mesh.vertices.phys_crds.view, mesh.faces.phys_crds.view,
    surf_passive.view, surf_active.view, mesh.faces.area);
  std::cout << "computing surface laplacian.\n";
  lap.compute();
}

template <typename SeedType>
void SWE<SeedType>::init_velocity() {
  Kokkos::TeamPolicy<> vert_policy(mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::parallel_for("initialize velocity (passive)", vert_policy,
    SWEVelocityPassive<typename SeedType::geo>(velocity_passive.view, double_dot_passive.view,
      mesh.vertices.phys_crds.view, mesh.faces.phys_crds.view,
      rel_vort_active.view, div_active.view, mesh.faces.area,
      mesh.faces.mask, eps, mesh.n_faces_host()));
  Kokkos::TeamPolicy<> face_policy(mesh.n_faces_host(), Kokkos::AUTO());
  Kokkos::parallel_for("initialize velocity (active)", face_policy,
    SWEVelocityActive<typename SeedType::geo>(velocity_active.view, double_dot_active.view,
      mesh.faces.phys_crds.view, rel_vort_active.view, div_active.view,
      mesh.faces.area, mesh.faces.mask, eps, mesh.n_faces_host()));
}

template <typename SeedType>
template <typename VelocityType>
void SWE<SeedType>::init_velocity_from_function() {
  static_assert(std::is_same<typename SeedType::geo,
    typename VelocityType::geo>::value,
    "Geometry types must match");

  Kokkos::parallel_for(this->mesh.n_vertices_host(),
    VelocityKernel<VelocityType>(this->velocity_passive.view,
      this->mesh.vertices.phys_crds.view, 0));
  Kokkos::parallel_for(mesh.n_faces_host(),
    VelocityKernel<VelocityType>(velocity_active.view,
      mesh.faces.phys_crds.view, 0));
}

template <typename SeedType>
template <typename VorticityType>
void SWE<SeedType>::init_vorticity(const VorticityType& vort_fn) {
  const auto vcrds = mesh.vertices.phys_crds.view;
  const auto fcrds = mesh.faces.phys_crds.view;
  const auto vzeta = rel_vort_passive.view;
  const auto fzeta = rel_vort_active.view;
  const auto vh = depth_passive.view;
  const auto fh = depth_active.view;
  const auto vq = pot_vort_passive.view;
  const auto fq = pot_vort_active.view;
  const auto cor = coriolis;
  Kokkos::parallel_for(mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(vcrds, i, Kokkos::ALL);
      const Real zeta = vort_fn(mcrd);
      const Real f = cor.f(mcrd[SeedType::geo::ndim-1]);
      vzeta(i) = zeta;
      vq(i) = (zeta + f) / vh(i);
    });
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(fcrds, i, Kokkos::ALL);
      const Real zeta = vort_fn(mcrd);
      const Real f = cor.f(mcrd[SeedType::geo::ndim-1]);
      fzeta(i) = zeta;
      fq(i) = (zeta + f) / fh(i);
    });
}

template <typename SeedType>
template <typename DivergenceType>
void SWE<SeedType>::init_divergence(const DivergenceType& div_fn) {
  const auto vcrds = mesh.vertices.phys_crds.view;
  const auto fcrds = mesh.faces.phys_crds.view;
  const auto vsigma = div_passive.view;
  const auto fsigma = div_active.view;
  Kokkos::parallel_for(mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(vcrds, i, Kokkos::ALL);
      vsigma(i) = div_fn(mcrd);
    });
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mcrd = Kokkos::subview(fcrds, i, Kokkos::ALL);
      fsigma(i) = div_fn(mcrd);
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
template <typename SurfaceLaplacianType, typename BottomType>
void SWE<SeedType>::init_surface(SurfaceLaplacianType& lap, const BottomType& topo) {
  Kokkos::parallel_for("init surface (passive)", mesh.n_vertices_host(),
    SurfaceUpdatePassive(surf_passive.view, depth_passive.view,
      mesh.vertices.phys_crds.view, topo));
  Kokkos::parallel_for("init surface (active)", mesh.n_faces_host(),
    SurfaceUpdateActive(surf_active.view, depth_active.view, mass_active.view,
      mesh.faces.area, mesh.faces.mask, mesh.faces.phys_crds.view, topo));
  lap.update_src_data(mesh.vertices.phys_crds.view, mesh.faces.phys_crds.view,
    surf_passive.view, surf_active.view, mesh.faces.area);
  lap.compute();
}

template <typename SeedType>
Real SWE<SeedType>::total_energy() const {
  return kinetic_energy() + potential_energy();
}

template <typename SeedType>
Real SWE<SeedType>::total_enstrophy() const {
  Real ens;
  Kokkos::parallel_reduce("SWE enstrophy", mesh.n_faces_host(),
    ScalarSquareIntegralFunctor(rel_vort_active.view, mesh.faces.area, mesh.faces.mask), ens);
  return ens;
}

template <typename SeedType>
Real SWE<SeedType>::kinetic_energy() const {
  Real ke;
  Kokkos::parallel_reduce("SWE kinetic energy", mesh.n_faces_host(),
    KineticEnergyFunctor<typename SeedType::geo>(depth_active.view, velocity_active.view, mesh.faces.area, mesh.faces.mask), ke);
  return 0.5*ke;
}

template <typename SeedType>
Real SWE<SeedType>::potential_energy() const {
  Real pe;
  Kokkos::parallel_reduce("SWE potential energy", mesh.n_faces_host(),
    ScalarSquareIntegralFunctor(depth_active.view, mesh.faces.area, mesh.faces.mask),
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
  vtk.add_scalar_point_data(swe.surf_laplacian_passive.view);
  vtk.add_scalar_point_data(swe.depth_passive.view);
  vtk.add_vector_point_data(swe.velocity_passive.view);
  vtk.add_scalar_point_data(swe.double_dot_passive.view);
  vtk.add_scalar_cell_data(swe.rel_vort_active.view);
  vtk.add_scalar_cell_data(swe.pot_vort_active.view);
  vtk.add_scalar_cell_data(swe.div_active.view);
  vtk.add_scalar_cell_data(swe.surf_active.view);
  vtk.add_scalar_cell_data(swe.surf_laplacian_active.view);
  vtk.add_scalar_cell_data(swe.depth_active.view);
  vtk.add_vector_cell_data(swe.velocity_active.view);
  vtk.add_scalar_cell_data(swe.double_dot_active.view);
  vtk.add_scalar_cell_data(swe.mass_active.view);
  return vtk;
  }
#endif

} // namespace Lpm

#endif
