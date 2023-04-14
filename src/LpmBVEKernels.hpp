#ifndef LPM_BVE_KERNELS_HPP
#define LPM_BVE_KERNELS_HPP
#include <cmath>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

typedef typename SphereGeometry::crd_view_type crd_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef typename ko::TeamPolicy<>::member_type member_type;

/** Green's function kernel for the sphere.

  Ref: Kimura & Okamoto 1987.

*/
template <typename VecType>
KOKKOS_INLINE_FUNCTION void greensFn(Real& psi, const VecType& tgt_x,
                                     const VecType& src_x, const Real& src_vort,
                                     const Real src_area) {
  const Real circ = -src_vort * src_area;
  psi = std::log(1 - SphereGeometry::dot(tgt_x, src_x)) * circ / (4 * PI);
}

/**
    Biot-Savart Kernel for the sphere.
        Computes contribution of source vortex located at xx with circulation
   zeta(xx) * area(xx) to tgt location x.

        u(x) = cross(x, xx)*zeta(xx)*area(xx);

    Ref: Kimura & Okamoto 1987.
*/
template <typename VecType>
KOKKOS_INLINE_FUNCTION void biotSavart(ko::Tuple<Real, 3>& u,
                                       const VecType& tgt_x,
                                       const VecType& src_x,
                                       const Real& src_vort,
                                       const Real& src_area) {
  u = SphereGeometry::cross(tgt_x, src_x);
  const Real strength =
      -src_vort * src_area / (4 * PI * (1 - SphereGeometry::dot(src_x, tgt_x)));
  for (Short j = 0; j < 3; ++j) {
    u[j] *= strength;
  }
}

/** Stream function reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$
*/
struct StreamReduceDistinct {
  typedef Real value_type;  ///< required by kokkos for custom reducers
  Index i;                  ///< index of target point in tgtx view
  crd_view tgtx;  ///< view holding coordinates of target locations (usually,
                  ///< vertices)
  crd_view srcx;  ///< view holding coordinates of source locations (usually,
                  ///< face centers)
  scalar_view_type srcf;  ///< view holding RHS (vorticity) data
  scalar_view_type srca;  ///< view holding panel areas
  mask_view_type
      facemask;  ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  StreamReduceDistinct(const Index& ii, const crd_view& x, const crd_view& xx,
                       const scalar_view_type& f, const scalar_view_type& a,
                       const mask_view_type& fm)
      : i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& pot) const {
    Real potential = 0;
    if (!facemask(j)) {
      auto mytgt = ko::subview(tgtx, i, ko::ALL());
      auto mysrc = ko::subview(srcx, j, ko::ALL());
      greensFn(potential, mytgt, mysrc, srcf(j), srca(j));
    }
    pot += potential;
  }
};

/** Velocity reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$
*/
struct VelocityReduceDistinct {
  typedef ko::Tuple<Real, 3>
      value_type;  ///< required by kokkos for custom reducers
  Index i;         ///< index of target point in tgtx view
  crd_view tgtx;   ///< view holding coordinates of target locations (usually,
                   ///< vertices)
  crd_view srcx;   ///< view holding coordinates of source locations (usually,
                   ///< face centers)
  scalar_view_type srcf;  ///< view holding RHS (vorticity) data
  scalar_view_type srca;  ///< view holding panel areas
  mask_view_type
      facemask;  ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  VelocityReduceDistinct(const Index& ii, const crd_view& x, const crd_view& xx,
                         const scalar_view_type& f, const scalar_view_type& a,
                         const mask_view_type& fm)
      : i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& vel) const {
    ko::Tuple<Real, 3> u;
    if (!facemask(j)) {
      auto mytgt = ko::subview(tgtx, i, ko::ALL());
      auto mysrc = ko::subview(srcx, j, ko::ALL());
      biotSavart(u, mytgt, mysrc, srcf(j), srca(j));
    }
    vel += u;
  }
};

/** @brief Solves the BVE at panel vertices.
 @device
 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function
 and 1 for velocity
*/

struct BVEVertexSolve {
  scalar_view_type vertpsi;   ///< [output] stream function values
  vec_view vertu;             ///< [output] velocity values
  crd_view vertx;             ///< [input] target coordinates
  crd_view facex;             ///< [input] source coordinates
  scalar_view_type facevort;  ///< [input] source vorticity
  scalar_view_type facearea;  ///< [input] source area
  mask_view_type facemask;    ///< [input] source mask (prevent divided panels
                              ///< from contributing to sums)
  Index nf;                   ///< [input] number of sources

  BVEVertexSolve(scalar_view_type& psi, vec_view& u, const crd_view& vx,
                 const crd_view& fx, const scalar_view_type& zeta,
                 const scalar_view_type& a, const mask_view_type& fm,
                 const Index& nsrc)
      : vertx(vx),
        facex(fx),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        vertpsi(psi),
        vertu(u),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real psi;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        StreamReduceDistinct(i, vertx, facex, facevort, facearea, facemask),
        psi);
    vertpsi(i) = psi;
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        VelocityReduceDistinct(i, vertx, facex, facevort, facearea, facemask),
        u);
    for (Short j = 0; j < 3; ++j) {
      vertu(i, j) = u[j];
    }
  }
};

struct BVEVertexStreamFn {
  scalar_view_type psi;
  crd_view vertx;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEVertexStreamFn(scalar_view_type& p, const crd_view& vx, const crd_view& fx,
                    const scalar_view_type& zeta, const scalar_view_type& a,
                    const mask_view_type& fm, const Index& nsrc)
      : psi(p),
        vertx(vx),
        facex(fx),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index& i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        StreamReduceDistinct(i, vertx, facex, facevort, facearea, facemask), p);
    psi(i) = p;
  }
};

struct BVEVertexVelocity {
  vec_view vertvel;
  crd_view vertx;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEVertexVelocity(vec_view& u, const crd_view& vx, const crd_view& fx,
                    const scalar_view_type& zeta, const scalar_view_type& a,
                    const mask_view_type& fm, const Index& nsrc)
      : vertvel(u),
        vertx(vx),
        facex(fx),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        VelocityReduceDistinct(i, vertx, facex, facevort, facearea, facemask),
        u);
    for (Short j = 0; j < 3; ++j) {
      vertvel(i, j) = u[j];
    }
  }
};

struct StreamReduceCollocated {
  typedef Real value_type;  ///< required by kokkos for custom reducers
  Index i;                  ///< index of target coordinate vector
  crd_view srcx;            ///< collection of source coordinate vectors
  scalar_view_type srcf;    ///< source vorticity values
  scalar_view_type srca;    ///< panel areas
  mask_view_type mask;      ///< mask (excludes non-leaf faces)

  KOKKOS_INLINE_FUNCTION
  StreamReduceCollocated(const Index& ii, const crd_view& x,
                         const scalar_view_type& f, const scalar_view_type& a,
                         const mask_view_type& m)
      : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& pot) const {
    Real potential = 0;
    if (!mask(j) && i != j) {
      auto mtgt = ko::subview(srcx, i, ko::ALL());
      auto msrc = ko::subview(srcx, j, ko::ALL());
      greensFn(potential, mtgt, msrc, srcf(j), srca(j));
    }
    pot += potential;
  }
};

struct VelocityReduceCollocated {
  typedef ko::Tuple<Real, 3>
      value_type;         ///< required by kokkos for custom reducers
  Index i;                ///< index of target coordinate vector
  crd_view srcx;          ///< collection of source coordinates
  scalar_view_type srcf;  ///< source vorticity
  scalar_view_type srca;  ///< source areas
  mask_view_type mask;    ///< mask (excludes non-leaf sources)

  KOKKOS_INLINE_FUNCTION
  VelocityReduceCollocated(const Index& ii, const crd_view& x,
                           const scalar_view_type& f, const scalar_view_type& a,
                           const mask_view_type& m)
      : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& vel) const {
    ko::Tuple<Real, 3> u;
    if (!mask(j) && i != j) {
      auto mtgt = ko::subview(srcx, i, ko::ALL());
      auto msrc = ko::subview(srcx, j, ko::ALL());
      biotSavart(u, mtgt, msrc, srcf(j), srca(j));
    }
    vel += u;
  }
};

struct BVEFaceSolve {
  scalar_view_type facepsi;
  vec_view faceu;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceSolve(scalar_view_type& psi, vec_view& u, const crd_view& x,
               const scalar_view_type& zeta, const scalar_view_type& a,
               const mask_view_type& fm, const Index& nsrc)
      : facepsi(psi),
        faceu(u),
        facex(x),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real psi;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        StreamReduceCollocated(i, facex, facevort, facearea, facemask), psi);
    facepsi(i) = psi;
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        VelocityReduceCollocated(i, facex, facevort, facearea, facemask), u);
    for (Short j = 0; j < 3; ++j) {
      faceu(i, j) = u[j];
    }
  }
};

struct BVEFaceStreamFn {
  scalar_view_type psi;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceStreamFn(scalar_view_type& p, const crd_view& fx,
                  const scalar_view_type& zeta, const scalar_view_type& a,
                  const mask_view_type& fm, const Index& nsrc)
      : psi(p),
        facex(fx),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        StreamReduceCollocated(i, facex, facevort, facearea, facemask), p);
    psi(i) = p;
  }
};

struct BVEFaceVelocity {
  vec_view faceu;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceVelocity(vec_view& u, const crd_view& fx, const scalar_view_type& zeta,
                  const scalar_view_type& a, const mask_view_type& fm,
                  const Index& nsrc)
      : faceu(u),
        facex(fx),
        facevort(zeta),
        facearea(a),
        facemask(fm),
        nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        VelocityReduceCollocated(i, facex, facevort, facearea, facemask), u);
    for (Short j = 0; j < 3; ++j) {
      faceu(i, j) = u[j];
    }
  }
};

struct BVEVorticityTendency {
  scalar_view_type dzeta;
  vec_view vel;
  Real Omega;
  Real dt;

  BVEVorticityTendency(scalar_view_type& dvort, const vec_view& u,
                       const Real& timestep, const Real& rot)
      : dzeta(dvort), vel(u), dt(timestep), Omega(rot) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& i) const {
    dzeta(i) = -2.0 * Omega * vel(i, 2) * dt;
  }
};

}  // namespace Lpm
#endif
