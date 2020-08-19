#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmPSE.hpp"

#include "Kokkos_Core.hpp"

namespace Lpm {

typedef typename ko::TeamPolicy<>::member_type member_type;

template <typename VecType> KOKKOS_INLINE_FUNCTION
void velocityKernel(ko::Tuple<Real,2>& u, const VecType& tgt_x,
  const VecType& src_x, const Real& src_vort, const Real& src_div, const Real& src_area) {

  Real sqdist = 0;
  for (Short i=0; i<2; ++i) {
    sqdist += square(tgt_x[i]-src_x[i]);
  }
  const Real denom = 2*PI*sqdist;
  const Real vort_str = src_vort * src_area / denom;
  const Real div_str = src_div * src_area /denom;

  u[0] = -(tgt_x[1] - src_x[1])*vort_str + (tgt_x[0] - src_x[0])*div_str;
  u[1] =  (tgt_x[0] - src_x[0])*vort_str + (tgt_x[1] - src_x[1])*div_str;
}

template <typename VecType> KOKKOS_INLINE_FUNCTION
void planeSweRhsPse(ko::Tuple<Real,7>& res, const VecType& tgt_x, const Real& tgt_s,
  const VecType& src_x, const Real& src_vort, const Real& src_div,
  const Real& src_area, const Real& src_s, const Real& pse_eps) {

  Real sqdist = 0;
  for (Short k=0; k<2; ++k) {
    sqdist += square(tgt_x[k] - src_x[k]);
  }

  if (sqdist > 10*ZERO_TOL) {
  const Real denom = 2*PI*sqdist;
  const Real denom2 = 2*PI*square(sqdist);
  const Real rot_strength = src_vort*src_area/denom;
  const Real pot_strength = src_div*src_area/denom;

  const Real pse_scaled_r = pse_kernel_input<PlaneGeometry,VecType>(tgt_x, src_x, pse_eps);
  const Real lap_ker = bivariateLaplacianOrder8(pse_scaled_r);

  // u = velocity, x component
  res[0] = -(tgt_x[1] - src_x[1])*rot_strength + (tgt_x[0] - src_x[0])*pot_strength;
  // v = velocity, y component
  res[1] =  (tgt_x[0] - src_x[0])*rot_strength + (tgt_x[1] - src_x[1])*pot_strength;
  // du/dx
  res[2] =  pot_strength - (tgt_x[0] - src_x[0])*((tgt_x[0] - src_x[0])*src_div -
    (tgt_x[1] - src_x[1])*src_vort)*src_area/denom2;
  // du/dy
  res[3] = -rot_strength - (tgt_x[1] - src_x[1])*((tgt_x[0] - src_x[0])*src_div -
    (tgt_x[1] - src_x[1])*src_vort)*src_area/denom2;
  // dv/dx
  res[4] =  rot_strength - (tgt_x[0] - src_x[0])*((tgt_x[1] - src_x[1])*src_div +
    (tgt_x[0] - src_x[0])*src_vort)*src_area/denom2;
  // dv/dy
  res[5] =  pot_strength - (tgt_x[1] - src_x[1])*((tgt_x[1] - src_x[1])*src_div +
    (tgt_x[0] - src_x[0])*src_vort)*src_area/denom2;
  // laplacian(s)
  res[6] = (src_s - tgt_s) * src_area * lap_ker / square(pse_eps);
  }
}

/**
  Reduction functor for Shallow Water direct summation in the plane

  Results contained in a 7-tuple as follows:
    index 0: u
    index 1: v
    index 2: du/dx
    index 3: du/dy
    index 4: dv/dx
    index 5: dv/dy
    index 6: laplacian(s) from PSE
*/
struct PlanarSWEDirectSum {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef ko::Tuple<Real,7> value_type;
  Index i; ///< index of target point
  crd_view tgtx;
  scalar_view_type tgt_sfc;
  crd_view srcx;
  scalar_view_type src_zeta;
  scalar_view_type src_sigma;
  scalar_view_type src_area;
  scalar_view_type src_sfc;
  Real pse_eps;
  bool collocated_src_tgt;

  KOKKOS_INLINE_FUNCTION
  PlanarSWEDirectSum(const Index& tind, const crd_view& tx, const scalar_view_type& tgtsfc,
    const crd_view& sx, const scalar_view_type& z, const scalar_view_type& sdiv,
    const scalar_view_type& a, const scalar_view_type& ssfc, const Real& eps) :
    i(tind), tgtx(tx), tgt_sfc(tgtsfc), srcx(sx), src_zeta(z), src_sigma(sdiv),
    src_area(a), src_sfc(ssfc), pse_eps(eps), collocated_src_tgt(tx==sx) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& r) const {
    if (!collocated_src_tgt || i != j) {
      const auto mtgt = ko::subview(tgtx, i, ko::ALL);
      const auto msrc = ko::subview(srcx, j, ko::ALL);
      ko::Tuple<Real,7> lsum;
      planeSweRhsPse(lsum, mtgt, tgt_sfc(i), msrc, src_zeta(j), src_sigma(j), src_area(j),
        src_sfc(j), pse_eps);
      r += lsum;
    }
  }
};

struct PlanarSWEVertexSums {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  vec_view vertvel;
  scalar_view_type vertddot;
  scalar_view_type vertlaps;
  crd_view vertx;
  scalar_view_type vertsfc;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facediv;
  scalar_view_type facearea;
  scalar_view_type facesfc;
  Real eps;
  Index nf;

  PlanarSWEVertexSums(vec_view& vvel, scalar_view_type& vdd, scalar_view_type& vlap,
    const crd_view& vx, const scalar_view_type& vsfc, const crd_view& fx,
    const scalar_view_type& fz, const scalar_view_type& fdiv, const scalar_view_type& fa,
    const scalar_view_type& fsfc, const Real& ep) : vertvel(vvel), vertddot(vdd),
    vertlaps(vlap), vertx(vx), vertsfc(vsfc), facex(fx), facevort(fz), facediv(fdiv),
    facearea(fa), facesfc(fsfc), eps(ep), nf(fx.extent(0)) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank(); // tgt vertex index
    ko::Tuple<Real,7> red;
    /* reduction over faces */
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
      PlanarSWEDirectSum(i, vertx, vertsfc, facex, facevort, facediv, facearea, facesfc, eps), red);
    vertvel(i,0) = red[0];
    vertvel(i,1) = red[1];
    vertddot(i) = red[2]*red[2] + 2*red[3]*red[4] + red[5]*red[5];
    vertlaps(i) = red[6]/square(eps);
  }
};

struct PlanarSWEVertexRHS {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  crd_view dx;
  scalar_view_type dzeta;
  scalar_view_type dsigma;
  scalar_view_type dh;
  crd_view vertx;
  vec_view vertvel;
  scalar_view_type vertvort;
  scalar_view_type vertdiv;
  scalar_view_type vertddot;
  scalar_view_type vertlaps;
  scalar_view_type vertdepth;
  Real f0, beta;
  Real g;
  Real dt;

  PlanarSWEVertexRHS(crd_view& dx_, scalar_view_type& dz, scalar_view_type& dsig,
   scalar_view_type& ddepth, const crd_view& vx, const vec_view& uv,
   const scalar_view_type& vvort, const scalar_view_type& vdiv, const scalar_view_type& vdd,
   const scalar_view_type& vls, const scalar_view_type& h, const Real& ff, const Real& bb,
   const Real& gg, const Real& dt_) :
   dx(dx_), dzeta(dz), dsigma(dsig), dh(ddepth), vertx(vx), vertvel(uv), vertvort(vvort),
   vertdiv(vdiv), vertddot(vdd), vertlaps(vls), vertdepth(h), f0(ff), beta(bb), g(gg), dt(dt_) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    const bool hasmass = vertdepth(i) > 0;
    dx(i,0) = (hasmass > 0 ? dt*vertvel(i,0) : 0);
    dx(i,1) = (hasmass > 0 ? dt*vertvel(i,1) : 0);
    const Real f = f0 + beta*vertx(i,1);
    const Real dfdt = beta*vertvel(i,1);
    dzeta(i) = (hasmass ? dt*(-dfdt - (vertvort(i) - f)*vertdiv(i)) : 0);
    dsigma(i) = (hasmass ? dt*(-f*vertvort(i) - vertddot(i) - g*vertlaps(i)) : 0);
    dh(i) = (hasmass ? dt*(-vertdiv(i)*vertdepth(i)) : 0);
  }
};

template <typename ProblemType>
struct PlanarSWESetVertexSfc {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  scalar_view_type sfc;
  scalar_view_type topo;
  scalar_view_type depth;
  crd_view vertx;

  PlanarSWESetVertexSfc(scalar_view_type& s, scalar_view_type& sb, const scalar_view_type& h,
    const crd_view& vx) : sfc(s), topo(sb), depth(h), vertx(vx) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    const auto mcrd = ko::subview(vertx, i, ko::ALL);
    topo(i) = ProblemType::bottom_height(mcrd);
    if (depth(i) < 0) {
      depth(i) = 0;
      sfc(i) = topo(i);
    }
    else {
      sfc(i) = depth(i) + topo(i);
    }
  }
};

template <typename ProblemType>
struct PlanarSWESetFaceSfc {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  scalar_view_type sfc;
  scalar_view_type depth;
  scalar_view_type topo;
  scalar_view_type mass;
  scalar_view_type area;
  mask_view_type mask;
  crd_view facex;

  PlanarSWESetFaceSfc(scalar_view_type& s, scalar_view_type& h, scalar_view_type& sb,
  const scalar_view_type& mm, const scalar_view_type& fa, const mask_view_type& fm, const crd_view& fx) :
    sfc(s), depth(h), topo(sb), mass(mm), area(fa), mask(fm), facex(fx) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    if (!mask(i)) {
      const auto mcrd = ko::subview(facex, i, ko::ALL);
      topo(i) = ProblemType::bottom_height(mcrd);
      if (mass(i) > 0) {
        depth(i) = mass(i) / area(i);
        sfc(i) = depth(i) + topo(i);
      }
      else {
        depth(i) = 0;
        sfc(i) = topo(i);
      }
    }
  }
};

struct PlanarSWEFaceSums {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  vec_view facevel;
  scalar_view_type faceddot;
  scalar_view_type facelaps;
  crd_view facex;
  scalar_view_type facevort;
  scalar_view_type facediv;
  scalar_view_type facearea;
  scalar_view_type facesfc;
  Real eps;
  Index nf;

  PlanarSWEFaceSums(vec_view& fv, scalar_view_type& fdd, scalar_view_type flap,
    const crd_view& fx, const scalar_view_type& fz, const scalar_view_type& fdiv,
    const scalar_view_type& fa, const scalar_view_type& fsfc, const Real& ep) :
    facevel(fv), faceddot(fdd), facelaps(flap), facex(fx), facevort(fz), facediv(fdiv),
    facearea(fa), facesfc(fsfc), eps(ep), nf(fx.extent(0)) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    ko::Tuple<Real,7> red;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
      PlanarSWEDirectSum(i, facex, facesfc, facex, facevort, facediv, facearea, facesfc, eps), red);
    facevel(i,0) = red[0];
    facevel(i,1) = red[1];
    faceddot(i) = square(red[2]) + 2*red[3]*red[5] + square(red[5]);
    facelaps(i) = red[6]/square(eps);
  }
};

struct PlanarSWEFaceRHS {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  typedef typename PlaneGeometry::vec_view_type vec_view;
  crd_view dx;
  scalar_view_type dzeta;
  scalar_view_type dsigma;
  scalar_view_type darea;
  crd_view facex;
  vec_view facevel;
  scalar_view_type facevort;
  scalar_view_type facediv;
  scalar_view_type faceddot;
  scalar_view_type facelaps;
  scalar_view_type facearea;
  mask_view_type mask;
  Real f0;
  Real beta;
  Real g;
  Real dt;

  PlanarSWEFaceRHS(crd_view& dx_, scalar_view_type& dz, scalar_view_type& dsig,
    scalar_view_type& da, const crd_view& fx, const vec_view& uv, const scalar_view_type& fzeta,
    const scalar_view_type& fdiv, const scalar_view_type& fdd, const scalar_view_type& flaps,
    const scalar_view_type& fa, const mask_view_type& fm, const Real& ff, const Real& bb,
    const Real& gg, const Real& dt_) :
    dx(dx_), dzeta(dz), dsigma(dsig), darea(da), facex(fx), facevel(uv), facevort(fzeta),
    facediv(fdiv), faceddot(fdd), facelaps(flaps), facearea(fa), mask(fm), f0(ff),
    beta(bb), g(gg), dt(dt_) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    if (!mask(i)) {
      dx(i,0) = dt*facevel(i,0);
      dx(i,1) = dt*facevel(i,1);
      const Real f = f0 + beta*facex(i,1);
      const Real df = beta*facevel(i,1);
      dzeta(i) = dt*(-df - (facevort(i)-f)*facediv(i));
      dsigma(i) = dt*(-f*facevort(i) - faceddot(i) - g*facelaps(i));
      darea(i) = dt*(facediv(i)*facearea(i));
    }
  }
};



}
#endif
