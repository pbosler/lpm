#ifndef LPM_BVE_SPHERE_KERNELS_HPP
#define LPM_BVE_SPHERE_KERNELS_HPP
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_kokkos_defs.hpp"
#include "lpm_constants.hpp"
#include <cmath>

namespace Lpm {

typedef typename SphereGeometry::crd_view_type crd_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef typename ko::TeamPolicy<>::member_type member_type;

/** @brief Evaluates the Poisson equation's Green's function for SpherePoisson

  Returns \f$ g(x,y)f(y)A(y),\f$ where \f$ g(x,y) = -log(1-x\cdot y)/(4\pi) \f$.

  @param psi Output value --- potential response due to a source of strength src_f*src_area
  @param tgt_x Coordinate of target location
  @param src_x Source location
  @param src_f Source (e.g., vorticity) value
  @param src_area Area of source panel
*/
template <typename VecType> KOKKOS_INLINE_FUNCTION
void greens_fn(Real& psi, const VecType& tgt_x, const VecType& src_x,
  const Real& src_vort, const Real src_area) {
  const Real circ = -src_vort*src_area;
  psi = std::log(1-SphereGeometry::dot(tgt_x,src_x)) * circ / (4*constants::PI);
}

/** @brief Computes the spherical Biot-Savart kernel's contribution to velocity for a single source

  Returns \f$ K(x,y)f(y)A(y)\f$, where \f$K(x,y) = \nabla g(x,y)\times x = \frac{x \times y}{4\pi(1-x\cdot y)}\f$.

  @param psi Output value --- potential response due to a source of strength src_f*src_area
  @param tgt_x Coordinate of target location
  @param src_x Source location
  @param src_f Source (e.g., vorticity) value
  @param src_area Area of source panel
*/
template <typename VecType> KOKKOS_INLINE_FUNCTION
void biot_savart(ko::Tuple<Real,3>& u, const VecType& tgt_x,
  const VecType& src_x, const Real& src_vort, const Real& src_area) {
  u = SphereGeometry::cross(tgt_x, src_x);
  const Real strength = -src_vort * src_area /
    (4*constants::PI*(1-SphereGeometry::dot(src_x,tgt_x)));
  for (Short j=0; j<3; ++j) {
    u[j] *= strength;
  }
}

/** Stream function reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$
*/
struct StreamReduceDistinct {
  typedef Real value_type; ///< required by kokkos for custom reducers
  Index i; ///< index of target point in tgtx view
  crd_view tgtx; ///< view holding coordinates of target locations (usually, vertices)
  crd_view srcx; ///< view holding coordinates of source locations (usually, face centers)
  scalar_view_type srcf; ///< view holding RHS (vorticity) data
  scalar_view_type srca; ///< view holding panel areas
  mask_view_type facemask; ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  StreamReduceDistinct(const Index& ii, const crd_view& x, const crd_view& xx, const scalar_view_type& f,
     const scalar_view_type& a, const mask_view_type& fm) :
     i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& pot) const {
      Real potential = 0;
      if (!facemask(j)) {
          auto mytgt = ko::subview(tgtx,i,ko::ALL());
          auto mysrc = ko::subview(srcx,j,ko::ALL());
          greens_fn(potential, mytgt, mysrc, srcf(j), srca(j));
      }
      pot += potential;
  }
};

/** Velocity reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$
*/
struct VelocityReduceDistinct {
  typedef ko::Tuple<Real,3> value_type; ///< required by kokkos for custom reducers
  Index i; ///< index of target point in tgtx view
  crd_view tgtx; ///< view holding coordinates of target locations (usually, vertices)
  crd_view srcx; ///< view holding coordinates of source locations (usually, face centers)
  scalar_view_type srcf; ///< view holding RHS (vorticity) data
  scalar_view_type srca; ///< view holding panel areas
  mask_view_type facemask; ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  VelocityReduceDistinct(const Index& ii, const crd_view& x,
    const crd_view& xx, const scalar_view_type& f, const scalar_view_type& a, const mask_view_type& fm):
    i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& vel) const {
      ko::Tuple<Real,3> u;
      if (!facemask(j)) {
          auto mytgt = ko::subview(tgtx, i, ko::ALL());
          auto mysrc = ko::subview(srcx, j, ko::ALL());
          biot_savart(u, mytgt, mysrc, srcf(j), srca(j));
      }
      vel += u;
  }
};

/** @brief Solves the BVE for stream function and velocity at passive particlse.
 @device
 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function and 1 for velocity
*/
struct BVEVertexSolve {
  scalar_view_type vertpsi; ///< [output] stream function values
  vec_view vertu; ///< [output] velocity values
  crd_view vertx; ///< [input] target coordinates
  crd_view facex; ///< [input] source coordinates
  scalar_view_type facevort; ///< [input] source vorticity
  scalar_view_type facearea; ///< [input] source area
  mask_view_type facemask;///< [input] source mask (prevent divided panels from contributing to sums)
  Index nf; ///< [input] number of sources

  BVEVertexSolve(scalar_view_type& psi, vec_view& u, const crd_view& vx, const crd_view& fx,
    const scalar_view_type& zeta, const scalar_view_type& a, const mask_view_type& fm, const Index& nsrc) :
    vertx(vx), facex(fx), facevort(zeta), facearea(a), facemask(fm), vertpsi(psi), vertu(u), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real psi;
    ko::parallel_reduce(ko::TeamThreadRange(mbr,nf),
      StreamReduceDistinct(i, vertx, facex, facevort, facearea, facemask), psi);
    vertpsi(i) = psi;
    ko::Tuple<Real,3> u;
    ko::parallel_reduce(ko::TeamThreadRange(mbr,nf),
      VelocityReduceDistinct(i, vertx, facex, facevort, facearea, facemask), u);
    for (Short j=0; j<3; ++j) {
      vertu(i,j) = u[j];
    }
  }
};

/** @brief Solves Poisson equation for the stream function (but not velocity) at passive particlse.
 @device
 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function and 1 for velocity
*/
struct BVEVertexStreamFn {
  scalar_view_type psi;
  crd_view vertx;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEVertexStreamFn(scalar_view_type& p, const crd_view& vx, const crd_view& fx, const scalar_view_type& zeta,
    const scalar_view_type& a, const mask_view_type& fm, const Index& nsrc) : psi(p), vertx(vx), facex(fx),
    facevort(zeta), facearea(a), facemask(fm), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index& i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(ko::TeamThreadRange(mbr,nf),
      StreamReduceDistinct(i, vertx, facex, facevort, facearea, facemask),p);
    psi(i) = p;
  }
};

/** @brief Solves Poisson equation for velocity (but not the stream function) at passive particlse.
 @device
 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function and 1 for velocity
*/
struct BVEVertexVelocity {
  vec_view vertvel;
  crd_view vertx;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEVertexVelocity(vec_view& u, const crd_view& vx, const crd_view& fx, const scalar_view_type& zeta,
    const scalar_view_type& a, const mask_view_type& fm, const Index& nsrc) : vertvel(u), vertx(vx),
    facex(fx), facevort(zeta), facearea(a), facemask(fm), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    ko::Tuple<Real,3> u;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
      VelocityReduceDistinct(i, vertx, facex, facevort, facearea, facemask), u);
    for (Short j=0; j<3; ++j) {
      vertvel(i,j) = u[j];
    }
  }
};

/** @brief Reduction kernel for stream function at active particles.
 @device
 @par Parallel pattern:
 1 thread team per target site performs reduction for stream function
*/
struct StreamReduceCollocated {
  typedef Real value_type; ///< required by kokkos for custom reducers
  Index i; ///< index of target coordinate vector
  crd_view srcx; ///< collection of source coordinate vectors
  scalar_view_type srcf; ///< source vorticity values
  scalar_view_type srca; ///< panel areas
  mask_view_type mask; ///< mask (excludes non-leaf faces)

  KOKKOS_INLINE_FUNCTION
  StreamReduceCollocated(const Index& ii, const crd_view& x, const scalar_view_type& f, const scalar_view_type& a,
    const mask_view_type& m) : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& pot) const {
      Real potential = 0;
      if (!mask(j) && i != j) {
          auto mtgt = ko::subview(srcx, i, ko::ALL());
          auto msrc = ko::subview(srcx, j, ko::ALL());
          greens_fn(potential, mtgt, msrc, srcf(j), srca(j));
      }
      pot += potential;
  }
};


/** @brief Reduction kernel of BVE for velocity at active particles.
 @device
 @par Parallel pattern:
 1 thread team per target site performs reduction for velocity
*/
struct VelocityReduceCollocated {
  typedef ko::Tuple<Real,3> value_type; ///< required by kokkos for custom reducers
  Index i; ///< index of target coordinate vector
  crd_view srcx; ///< collection of source coordinates
  scalar_view_type srcf; ///< source vorticity
  scalar_view_type srca; ///< source areas
  mask_view_type mask; ///< mask (excludes non-leaf sources)

  KOKKOS_INLINE_FUNCTION
  VelocityReduceCollocated(const Index& ii, const crd_view& x, const scalar_view_type& f, const scalar_view_type& a,
    const mask_view_type& m) : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& vel) const {
      ko::Tuple<Real,3> u;
      if (!mask(j) && i != j) {
          auto mtgt = ko::subview(srcx, i, ko::ALL());
          auto msrc = ko::subview(srcx, j, ko::ALL());
          biot_savart(u, mtgt, msrc, srcf(j), srca(j));
      }
      vel += u;
  }
};

/** @brief Solves the BVE for both stream function and velocity at active particles.

@device
@par Parallel pattern:
1 thread team per target site performs 2 reductions: one for the stream function
and one for velocity.
*/
struct BVEFaceSolve {
  scalar_view_type facepsi;
  vec_view faceu;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceSolve(scalar_view_type& psi, vec_view& u, const crd_view& x,
    const scalar_view_type& zeta, const scalar_view_type& a, const mask_view_type& fm, const Index& nsrc) :
    facepsi(psi), faceu(u), facex(x), facevort(zeta), facearea(a), facemask(fm), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i=mbr.league_rank();
    Real psi;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), StreamReduceCollocated(i, facex, facevort, facearea, facemask), psi);
    facepsi(i) = psi;
    ko::Tuple<Real,3> u;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), VelocityReduceCollocated(i, facex, facevort, facearea, facemask), u);
    for (Short j=0; j<3; ++j) {
      faceu(i,j) = u[j];
    }
  }
};

/** @brief Solves the BVE for stream at active particles.

@device
@par Parallel pattern:
1 thread team per target site performs 2 reductions: one for the stream function
and one for velocity.
*/
struct BVEFaceStreamFn {
  scalar_view_type psi;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceStreamFn(scalar_view_type& p, const crd_view& fx, const scalar_view_type& zeta, const scalar_view_type& a,
    const mask_view_type& fm, const Index& nsrc) : psi(p), facex(fx), facevort(zeta),
    facearea(a), facemask(fm), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(ko::TeamThreadRange(mbr,nf),
      StreamReduceCollocated(i, facex, facevort, facearea, facemask),p);
    psi(i) = p;
  }
};

/** @brief Solves the BVE for velocity at active particles.

@device
@par Parallel pattern:
1 thread team per target site performs 2 reductions: one for the stream function
and one for velocity.
*/
struct BVEFaceVelocity {
  vec_view faceu;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facearea;
  mask_view_type facemask;
  Index nf;

  BVEFaceVelocity(vec_view& u, const crd_view& fx, const scalar_view_type& zeta, const scalar_view_type& a,
    const mask_view_type& fm, const Index& nsrc) : faceu(u), facex(fx), facevort(zeta), facearea(a), facemask(fm), nf(nsrc) {}

  KOKKOS_INLINE_FUNCTION
  void operator () (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    ko::Tuple<Real,3> u;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), VelocityReduceCollocated(i, facex, facevort, facearea, facemask), u);
    for (Short j=0; j<3; ++j) {
      faceu(i,j) = u[j];
    }
  }
};

/** @brief Kernel to compute the time derivative of vorticity  for the BVE.
  @device
*/
struct BVEVorticityTendency {
  scalar_view_type dzeta;
  vec_view vel;
  Real Omega;
  Real dt;

  BVEVorticityTendency(scalar_view_type& dvort, const vec_view& u, const Real& timestep, const Real& rot) :
    dzeta(dvort), vel(u), dt(timestep), Omega(rot) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    dzeta(i) = -2.0 * Omega * vel(i,2) * dt;
  }
};

}
#endif
