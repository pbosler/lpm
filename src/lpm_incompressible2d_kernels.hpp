#ifndef LPM_INCOMPRESSIBLE2D_KERNELS_HPP
#define LPM_INCOMPRESSIBLE2D_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

/** @brief Type containing the Green's function and Biot-Savart
  kernels for 2d incompressible flow.

  The generic template defines spherical kernels.
*/
template <typename Geo>
struct Incompressible2DKernels {
  using value_type           = Kokkos::Tuple<Real, Geo::ndim + 1>;
  static constexpr Int nvals = Geo::ndim + 1;

  /** Function computes the direct sum contributions of src_y on tgt_x.
      Results are a packed 4-tuple:

      result[0] = biot-savart, x-component
      result[1] = biot-savart, y-component
      result[2] = biot-savart, z-component
      result[3] = greens fn

      @param [in] tgt_x target point coordinates
      @param [in] src_y source point coordinates
      @param [in] eps kernel smoothing parameter
      @return packed 4-tuple
  */
  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static value_type kernel_vals(const XType& tgt_x,
                                                       const YType& src_y,
                                                       const Real& eps) {
    const Real arg = 1 - SphereGeometry::dot(tgt_x, src_y) + square(eps);
    Kokkos::Tuple<Real, 4> result;
    SphereGeometry::cross(result, tgt_x, src_y);
    for (Int k = 0; k < 3; ++k) {
      result[k] /= (-4 * constants::PI * arg);
    }
    result[3] = -log(arg) / (4 * constants::PI);
    return result;
  }
};

/** @brief Type containing the Green's function and Biot-Savart
  kernels for 2d incompressible flow.

  This specialized template defines planar kernels.
*/
template <>
struct Incompressible2DKernels<PlaneGeometry> {
  using value_type           = Kokkos::Tuple<Real, PlaneGeometry::ndim + 1>;
  static constexpr Int nvals = PlaneGeometry::ndim + 1;

  /** Function computes the direct sum contributions of src_y on tgt_x.
      Results are a packed 3-tuple:

      result[0] = biot-savart, x-component
      result[1] = biot-savart, y-component
      result[2] = greens fn

      @param [in] tgt_x target point coordinates
      @param [in] src_y source point coordinates
      @param [in] eps kernel smoothing parameter
      @return packed 4-tuple
  */
  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static value_type kernel_vals(const XType& tgt_x,
                                                       const YType& src_y,
                                                       const Real& eps) {
    const Real xmy[2] = {tgt_x[0] - src_y[0], tgt_x[1] - src_y[1]};
    const Real arg    = PlaneGeometry::norm2(xmy) + square(eps);
    const Real denom  = 2 * constants::PI * arg;
    Kokkos::Tuple<Real, 3> result;
    result[0] = -xmy[1] / denom;
    result[1] = xmy[0] / denom;
    result[2] = -log(arg) / (4 * constants::PI);
    return result;
  }
};

/** @brief Kokkos reducer concept for 2d incompressible flow
  (both planar and spherical).

  Each call to this functor computes the contribution of src_y onto tgt_x.
*/
template <typename Geo>
struct Incompressible2DReducer {
  using crd_view    = typename Geo::crd_view_type;
  using kernel_type = Incompressible2DKernels<Geo>;
  using value_type  = typename kernel_type::value_type;

  crd_view tgt_x;             /// target point coordinates
  Index i;                    /// target point index inside target views
  crd_view src_y;             /// source point coordinates
  scalar_view_type src_zeta;  /// source point relative vorticity
  scalar_view_type src_area;  /// source point area weights
  mask_view_type src_mask;    /// source point mask (to exclude divided panels)
  Real eps;                   /// kernel smoothing parameter
  bool collocated_src_tgt;  /// if true, the i = j contribution will be skipped

  KOKKOS_INLINE_FUNCTION
  Incompressible2DReducer(const crd_view tx, const Index i, const crd_view sy,
                          const scalar_view_type sz, const scalar_view_type sa,
                          const mask_view_type sm, const Real eps,
                          const bool collocated)
      : tgt_x(tx),
        i(i),
        src_y(sy),
        src_zeta(sz),
        src_area(sa),
        src_mask(sm),
        eps(eps),
        collocated_src_tgt(collocated) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& s) const {
    if (!collocated_src_tgt or i != j) {
      if (!src_mask(j)) {
        const auto xcrd       = Kokkos::subview(tgt_x, i, Kokkos::ALL);
        const auto ycrd       = Kokkos::subview(src_y, j, Kokkos::ALL);
        const Real Gamma_circ = src_zeta(j) * src_area(j);
        const value_type local_result =
            kernel_type::kernel_vals(xcrd, ycrd, eps);
        for (Int k = 0; k < kernel_type::nvals; ++k) {
          s[k] += local_result[k] * Gamma_circ;
        }
      }
    }
  }
};

/** @brief Direct sum functor for 2d incompressible flow on passive particles
  (both planar and spherical).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
template <typename Geo>
struct Incompressible2DPassiveSums {
  using crd_view    = typename Geo::crd_view_type;
  using vec_view    = typename Geo::vec_view_type;
  using kernel_type = Incompressible2DKernels<Geo>;
  using value_type  = typename kernel_type::value_type;

  vec_view passive_velocity;
  scalar_view_type passive_stream_fn;
  crd_view passive_x;
  crd_view active_y;
  scalar_view_type active_rel_vort;
  scalar_view_type active_area;
  mask_view_type active_mask;
  Real eps;
  Index n_active;

  Incompressible2DPassiveSums(vec_view pu, scalar_view_type ppsi,
                              const crd_view px, const crd_view ay,
                              const scalar_view_type az,
                              const scalar_view_type aa,
                              const mask_view_type am, const Real eps,
                              const Index na)
      : passive_velocity(pu),
        passive_stream_fn(ppsi),
        passive_x(px),
        active_y(ay),
        active_rel_vort(az),
        active_area(aa),
        active_mask(am),
        eps(eps),
        n_active(na) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    value_type sums;
    constexpr bool collocated = false;
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(thread_team, n_active),
        Incompressible2DReducer<Geo>(passive_x, i, active_y, active_rel_vort,
                                     active_area, active_mask, eps, collocated),
        sums);
    for (int k = 0; k < Geo::ndim; ++k) {
      passive_velocity(i, k) = sums[k];
    }
    passive_stream_fn(i) = sums[Geo::ndim];
  }
};

/** @brief Direct sum functor for 2d incompressible flow on active particles
  (both planar and spherical).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
template <typename Geo>
struct Incompressible2DActiveSums {
  using crd_view    = typename Geo::crd_view_type;
  using vec_view    = typename Geo::vec_view_type;
  using kernel_type = Incompressible2DKernels<Geo>;
  using value_type  = typename kernel_type::value_type;

  vec_view active_velocity;
  scalar_view_type active_stream_fn;
  crd_view active_xy;
  scalar_view_type active_rel_vort;
  scalar_view_type active_area;
  mask_view_type active_mask;
  Real eps;
  Index n_active;

  Incompressible2DActiveSums(vec_view au, scalar_view_type apsi,
                             const crd_view axy, const scalar_view_type az,
                             const scalar_view_type aa, const mask_view_type am,
                             const Real eps, const Index na)
      : active_velocity(au),
        active_stream_fn(apsi),
        active_xy(axy),
        active_rel_vort(az),
        active_area(aa),
        active_mask(am),
        eps(eps),
        n_active(na) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    value_type sums;
    const bool collocated = FloatingPoint<Real>::zero(eps);
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(thread_team, n_active),
        Incompressible2DReducer<Geo>(active_xy, i, active_xy, active_rel_vort,
                                     active_area, active_mask, eps, collocated),
        sums);
    for (int k = 0; k < Geo::ndim; ++k) {
      active_velocity(i, k) = sums[k];
    }
    active_stream_fn(i) = sums[Geo::ndim];
  }
};

/** @brief  RHS tendencies for 2d incompressible flow
  at active particles  (both planar and spherical problems)

  If U is the state vector, and the SWE are written as a dynamical
  system, dU/dt = F(U), this functor computes the RHS F, given U.

  This functor will be called from a Kokkos::parallel_for, with a range
  policy over all active particles.
*/
template <typename Geo>
struct Incompressible2DTendencies {
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using Coriolis =
      typename std::conditional<std::is_same<Geo, PlaneGeometry>::value,
                                CoriolisBetaPlane, CoriolisSphere>::type;

  scalar_view_type dzeta;  /// [out] vorticity derivative
  vec_view u;              /// [in] velocity
  Coriolis coriolis;       /// [in] Coriolis evaluations

  Incompressible2DTendencies(scalar_view_type dzeta, const vec_view u,
                             const Coriolis& coriolis)
      : dzeta(dzeta), u(u), coriolis(coriolis) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto ui = Kokkos::subview(u, i, Kokkos::ALL);
    dzeta(i)      = -coriolis.dfdt(ui);
  }
};

}  // namespace Lpm

#endif
