#ifndef LPM_HEADER_HPP
#define LPM_HEADER_HPP

#include "LpmConfig.h"
#include "lpm_swe_solver.hpp"
#include "lpm_swe_kernels.hpp"
#include "util/lpm_tuple.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

/** Reduction functor for separate source and target points;

  This functor is called in the inner loop of a parallel_for( parallel_reduce ) pattern.
*/
template <typename Geo>
struct SWEVelocityReducerSeparatePts {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  /** this value type collects the rhs terms for a swe velocity reduction

    values are packed as:

    value_type[0:ndim] = velocity components
    value_type[ndim:ndim*(1+ndim)] = ndim x ndim matrix for
      velocity gradient (row major storage)
  */
  using value_type = Kokkos::Tuple<Real, ndim*(1+ndim)>;
  Index i; /// index of tgt point in tgtx view
  crd_view tgtx; /// coordinates of tgt points that are all separated from
                 /// any source points
  crd_view srcx; /// coordinates of src points
  scalar_view_type rel_vort; /// relative vorticity of src points
  scalar_view_type divergence; /// divergence of src points
  scalar_view_type area; /// area of src points
  mask_view_type mask; /// mask to exclude divided src points
  Real eps; /// kernel regularization parameter

  KOKKOS_INLINE_FUNCTION
  SWEVelocityReducerSeparatePts(const Index i,
    const crd_view tx, const crd_view sx, const scalar_view_type zeta,
    const scalar_view_type sigma, const scalar_view_type a,
    const mask_view_type m, const Real eps) :
    i(i), tgtx(tx), srcx(sx), rel_vort(zeta), divergence(sigma),
    area(a), mask(m), eps(eps)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, value_type& rhs) const {
    if (not mask(j)) {
      const auto tgt_crd = Kokkos_subview(tgtx, i, Kokkos::ALL);
      const auto src_crd = Kokkos_subview(srcx, j, Kokkos::ALL);
      const Real src_zeta = rel_vort(j);
      const Real src_sigma = divergence(j);
      const Real src_area = area(j);

      Real vel_zeta[ndim];
      kzeta<Geo>(vel_zeta, tgt_crd, src_crd, src_zeta, src_area, eps);

      Real vel_sigma[ndim];
      ksigma<Geo>(vel_sigma, tgt_crd, src_crd, src_sigma, src_area, eps);

      Real gkz[ndim*ndim];
      const Real vort_str = src_zeta*src_area;
      grad_kzeta<Geo>(kgz, tgt_crd, src_crd, eps);

      Real gks[ndim*ndim];
      const real div_str = src_sigma*src_area;
      grad_ksigma<Geo>(gks, tgt_crd, src_crd, eps);

      for (Int i=0; i<ndim; ++i) {
        rhs[i] += vel_zeta[i] + vel_sigma[i];
      }
      for (Int i=0; i<ndim*ndim; ++i) {
        rhs[ndim + i] += gkz[i]*vort_str + gks[i]*div_str;
      }
    }
  }
};

/** Computes the velocity reduction at passive particles.

  This functor is called in the outer loop of a parallel_for(parallel_reduce) pattern.

  It computes velocity and the velocity gradient double dot product.
*/
template <typename Geo>
struct SWEVelocityPassive {
  static constexpr ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  vec_view velocity;  /// output: velocity at tgt pts
  scalar_view_type double_dot; /// output: double dot product of velocity tensor at output pts
  crd_view tgtx; /// input: coordinates of target pts
  crd_view srcx; /// input: coordinates of source pts
  scalar_view_type rel_vort; /// input: vorticity of source pts
  scalar_view_type divergence; /// input: divergence of sources pts
  scalar_view_type area; /// input: area of src points
  mask_view_type mask; /// input: mask of src points
  Real eps; /// input: kernel regularization parameter
  Index nsrc; /// number of source points

  SWEVelocityPassive(vec_view u, scalar_view_type dd, crd_view tx, crd_view sx,
    scalar_view_type zeta, scalar_view_type sigma, scalar_view_type area,
    mask_view_type m, const Real eps, const Index n) :
      velocity(u),
      double_dot(dd),
      tgtx(tx),
      srcx(sx),
      rel_vort(zeta),
      divergence(sigma),
      area(a),
      mask(m),
      eps(eps),
      nsrc(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();  // index of tgt
    Kokkos::Tuple<Real, ndim*(1+ndim)> rhs; // collected results of reductions
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nsrc),
      SWEVelocityReducerSeparatePts<Geo>(i, tgtx, srcx, rel_vort, divergence, area, mask, eps),
      rhs);
    auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    for (Int j=0; j<ndim; ++j) {
      mvel[j] = rhs[j];
    }
    double_dot[i] = 0;
    for (Int j=0; j<ndim; ++j) {
      for (Int k=0; k<ndim; ++k) {
        jk_idx = j*ndim + k;
        kj_idx = k*ndim + j;

        double_dot[i] += (j == k ? 1 : 2) * rhs[ndim + jk_idx] * rhs[ndim + kj_idx];
      }
    }
  }
};

template <typename Geo>
struct SWEVelocityActive {
  static constexpr ndim = Geo::ndim;
    using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  vec_view velocity;  /// output: velocity at tgt pts
  scalar_view_type double_dot; /// output: double dot product of velocity tensor at output pts
  crd_view srcx; /// input: coordinates of source pts
  scalar_view_type rel_vort; /// input: vorticity of source pts
  scalar_view_type divergence; /// input: divergence of sources pts
  scalar_view_type area; /// input: area of src points
  mask_view_type mask; /// input: mask of src points
  Real eps; /// input: kernel regularization parameter
  Index nsrc; /// number of source points

  SWEVelocityActive(vec_view u, scalar_view_type dd, crd_view sx,
    scalar_view_type zeta, scalar_view_type sigma, scalar_view_type area,
    mask_view_type m, const Real eps, const Index n) :
      velocity(u),
      double_dot(dd),
      srcx(sx),
      rel_vort(zeta),
      divergence(sigma),
      area(a),
      mask(m),
      eps(eps),
      nsrc(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();  // index of tgt
    Kokkos::Tuple<Real, ndim*(1+ndim)> rhs; // collected results of reductions

    // compute rhs velocity reductions
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nsrc),
      SWEVelocityReducerCollocatedPts<Geo>(i, srcx, rel_vort, divergence, area, mask, eps),
      rhs);

    // unpack results
    auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    for (Int j=0; j<ndim; ++j) {
      mvel[j] = rhs[j];
    }
    double_dot[i] = 0;
    for (Int j=0; j<ndim; ++j) {
      for (Int k=0; k<ndim; ++k) {
        jk_idx = j*ndim + k;
        kj_idx = k*ndim + j;

        double_dot[i] += (j == k ? 1 : 2) * rhs[ndim + jk_idx] * rhs[ndim + kj_idx];
      }
    }
  }
};

template <typename Geo>
struct SWEVelocityReducerCollocatedPts {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  /** this value type collects the rhs terms for a swe velocity reduction

    values are packed as:

    value_type[0:ndim] = velocity components
    value_type[ndim:ndim*(1+ndim)] = ndim x ndim matrix for
      velocity gradient (row major storage)
  */
  using value_type = Kokkos::Tuple<Real, ndim*(1+ndim)>;
  Index i; /// index of tgt point in tgtx view
  crd_view srcx; /// coordinates of src/tgt points
  scalar_view_type rel_vort; /// relative vorticity of src points
  scalar_view_type divergence; /// divergence of src points
  scalar_view_type area; /// area of src points
  mask_view_type mask; /// mask to exclude divided src points
  Real eps; /// kernel regularization parameter

  KOKKOS_INLINE_FUNCTION
  SWEVelocityReducerSeparatePts(const Index i,
    const crd_view sx, const scalar_view_type zeta,
    const scalar_view_type sigma, const scalar_view_type a,
    const mask_view_type m, const Real eps) :
    i(i), srcx(sx), rel_vort(zeta), divergence(sigma),
    area(a), mask(m), eps(eps)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, value_type& rhs) const {
    if ( (not mask(j) and (i != j)) ) {
      const auto tgt_crd = Kokkos_subview(srcx, i, Kokkos::ALL);
      const auto src_crd = Kokkos_subview(srcx, j, Kokkos::ALL);
      const Real src_zeta = rel_vort(j);
      const Real src_sigma = divergence(j);
      const Real src_area = area(j);

      Real vel_zeta[ndim];
      kzeta<Geo>(vel_zeta, tgt_crd, src_crd, src_zeta, src_area, eps);

      Real vel_sigma[ndim];
      ksigma<Geo>(vel_sigma, tgt_crd, src_crd, src_sigma, src_area, eps);

      Real gkz[ndim*ndim];
      const Real vort_str = src_zeta*src_area;
      grad_kzeta<Geo>(kgz, tgt_crd, src_crd, eps);

      Real gks[ndim*ndim];
      const real div_str = src_sigma*src_area;
      grad_ksigma<Geo>(gks, tgt_crd, src_crd, eps);

      for (Int i=0; i<ndim; ++i) {
        rhs[i] += vel_zeta[i] + vel_sigma[i];
      }
      for (Int i=0; i<ndim*ndim; ++i) {
        rhs[ndim + i] += gkz[i]*vort_str + gks[i]*div_str;
      }
    }
  }
};

template <typename SeedType>
SWERK4<SeedType>::SWERK4(SWE<SeedType>& swe, const Real dt) :
  swe(swe),
  dt(dt),
  x_passive(
    Kokkos::subview(swe.mesh.vertices.phys_crds.view,
    std::make_pair(0, mesh.n_vertices_host()), Kokkos::ALL)),
  x_active(
    Kokkos::subview(swe.mesh.faces.phys_crds.view,
    std::make_pair(0, mesh.n_faces_host()), Kokkos::ALL)),
  velocity_passive(
    Kokkos::subview(swe.velocity_passive,
    std::make_pair(0, swe.mesh.n_vertices_host()), Kokkos::ALL)),
  velocity_active(
    Kokkos::subview(swe.velocity_active,
    std::make_pair(0, swe.mesh.n_faces_host()), Kokkos::ALL)),
  rel_vort_passive(
    Kokkos::subview(swe.rel_vort_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  rel_vort_active(
    Kokkos::subview(swe.rel_vort_active.view, std::make_pair(0, mesh.n_faces_host()))),
  divergence_passive(
    Kokkos::subview(swe.divergence_passive.view, std::make_pair(0, swe.mesh.n_vertices_host()))),
  divergence_active(
    Kokkos::subview(swe.divergence_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
  depth_passive(Kokkos::subview(swe.depth_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  area_active(Kokkos::subview(swe.mesh.faces.area, std::make_pair(0, swe.mesh.n_faces_host()))),
  mass_active(Kokkos::subview(swe.mass_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
  surf_passive(Kokkos::subview(swe.surf_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  surf_active(Kokkos::subview(swe.surf_active, std::make_pair(0, swe.mesh.n_faces_host()))),
  depth_active(Kokkos::subview(sew.depth_active, std::make_pair(0, swe.mesh.n_faces_host()))),
  x_passive_1("rk4_xp1", swe.mesh.n_vertices_host()),
  x_passive_2("rk4_xp2", swe.mesh.n_vertices_host()),
  x_passive_3("rk4_xp3", swe.mesh.n_vertices_host()),
  x_passive_4("rk4_xp4", swe.mesh.n_vertices_host()),
  x_passive_work("rk4_xp_work", swe.mesh.n_vertices_host()),
  rel_vort_passive_1("rk4_zeta_p1", swe.mesh.n_vertices_host()),
  rel_vort_passive_2("rk4_zeta_p2", swe.mesh.n_vertices_host()),
  rel_vort_passive_3("rk4_zeta_p3", swe.mesh.n_vertices_host()),
  rel_vort_passive_4("rk4_zeta_p4", swe.mesh.n_vertices_host()),
  rel_vort_passive_work("rk4_zeta_pwork", swe.mesh.n_vertices_host()),
  div_passive_1("rk4_div_p1", swe.mesh.n_vertices_host()),
  div_passive_2("rk4_div_p2", swe.mesh.n_vertices_host()),
  div_passive_3("rk4_div_p3", swe.mesh.n_vertices_host()),
  div_passive_4("rk4_div_p4", swe.mesh.n_vertices_host()),
  div_passive_work("rk4_div_pwork", swe.mesh.n_vertices_host()),
  depth_passive_1("rk4_depth_p1", swe.mesh.n_vertices_host()),
  depth_passive_2("rk4_depth_p2", swe.mesh.n_vertices_host()),
  depth_passive_3("rk4_depth_p3", swe.mesh.n_vertices_host()),
  depth_passive_4("rk4_depth_p4", swe.mesh.n_vertices_host()),
  depth_passive_work("rk4_depth_pwork", swe.mesh.n_vertices_host()),
  double_dot_passive("rk4_ddot_p", swe.mesh.n_vertices_host()),
  surf_laplacian_passive("rk4_surf_lap_p", swe.mesh.n_vertices_host()),
  x_active_1("rk4_xa1", swe.mesh.n_vertices_host()),
  x_active_2("rk4_xa2", swe.mesh.n_vertices_host()),
  x_active_3("rk4_xa3", swe.mesh.n_vertices_host()),
  x_active_4("rk4_xa4", swe.mesh.n_vertices_host()),
  x_active_work("rk4_xa_work", swe.mesh.n_vertices_host()),
  rel_vort_active_1("rk4_zeta_a1", swe.mesh.n_vertices_host()),
  rel_vort_active_2("rk4_zeta_a2", swe.mesh.n_vertices_host()),
  rel_vort_active_3("rk4_zeta_a3", swe.mesh.n_vertices_host()),
  rel_vort_active_4("rk4_zeta_a4", swe.mesh.n_vertices_host()),
  rel_vort_active_work("rk4_zeta_awork", swe.mesh.n_vertices_host()),
  div_active_1("rk4_div_a1", swe.mesh.n_vertices_host()),
  div_active_2("rk4_div_a2", swe.mesh.n_vertices_host()),
  div_active_3("rk4_div_a3", swe.mesh.n_vertices_host()),
  div_active_4("rk4_div_a4", swe.mesh.n_vertices_host()),
  div_active_work("rk4_div_awork", swe.mesh.n_vertices_host()),
  area_active_1("rk4_area_a1", swe.mesh.n_vertices_host()),
  area_active_2("rk4_area_a2", swe.mesh.n_vertices_host()),
  area_active_3("rk4_area_a3", swe.mesh.n_vertices_host()),
  area_active_4("rk4_area_a4", swe.mesh.n_vertices_host()),
  area_active_work("rk4_area_awork", swe.mesh.n_vertices_host()),
  double_dot_active("rk4_ddot_a", swe.mesh.n_vertices_host()),
  surf_laplacian_active("rk4_surf_lap_a", swe.mesh.n_vertices_host()) {}

template <typename SeedType> template <typename SurfaceLaplacianType, typename BottomType>
void SWERK4<SeedType>::advance_timestep(SurfaceLaplacianType& lap, const BottomType& topo) {
  Kokkos::TeamPolicy<> passive_policy(swe.mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::TeamPolicy<> active_policy(swe.mesh.n_faces_host(), Kokkos::AUTO());

  // rk stage 1: on input, velocity, double dot product, and surface laplacian
  // are already defined
  KokkosBlas::scal(x_passive_1, dt, velocity_passive);
  KokkosBlas::scal(x_active_1, dt, velocity_active);
  Kokkos::parallel_for("rk stage 1 passive", swe.mesh.n_vertices_host(),
    SWEPassiveTendencies<typename SeedType::geo>(rel_vort_passive_1,
      div_passive_1, depth_passive_1,
      x_passive, velocity_passive, rel_vort_passive, divergence_passive,
      depth_passive, double_dot_passive, surf_laplacian_passive, swe.coriolis,
      swe.g, dt));
  Kokkos::parallel_for("rk stage 1 active", swe.mesh.n_faces_host(),
    SWEActiveTendencies<typename SeedType::geo>(rel_vort_active_1,
      div_active_1, area_active_1,
      x_active, velocity_active, rel_vort_active, divergence_active,
      area_active, double_dot_active, surf_laplacian_active, coriolis,
      swe.g, dt));
  Kokkos::fence();

  // rk stage 2: set input
  KokkosBlas::update(1, x_passive, 0.5, x_passive_1, 0,  x_passive_work);
  KokkosBlas::update(1, rel_vort_passive, 0.5, rel_vort_passive_1, 0, rel_vort_passive_work);
  KokkosBlas::update(1, divergence_passive, 0.5, div_passive_1, 0, div_passive_work);
  KokkosBlas::update(1, depth_passive, 0.5, depth_passive_1, 0, depth_passive_work);

  KokkosBlas::update(1, x_active, 0.5, x_active_1, 0,  x_active_work);
  KokkosBlas::update(1, rel_vort_active, 0.5, rel_vort_active_1, 0, rel_vort_active_work);
  KokkosBlas::update(1, divergence_active, 0.5, div_active_1, 0, div_active_work);
  KokkosBlas::update(1, area_active, 0.5, area_active_1, 0, area_active_work);

  Kokkos::parallel_for("rk stage 2 surface update (passive)", swe.mesh.n_vertices_host(),
    SurfaceUpdatePassive(surf_passive, depth_passive_work, x_passive_work, topo));
  Kokkos::parallel_for("rk stage 2 surface update (active)", swe.mesh.n_faces_host(),
    SurfaceUpdateActive(surf_active, depth_active, mass_active, area_active_work,
      swe.mesh.faces.mask, x_active_work, topo));
}

} // namespace Lpm

#endif
