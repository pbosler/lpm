#ifndef LPM_SWE_GALLERY_HPP
#define LPM_SWE_GALLERY_HPP

#include "LpmConfig.h"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

struct PlanarGaussian {
  typedef PlaneGeometry geo;
  Real strength;
  Real shape_parameter;
  Kokkos::Tuple<Real,2> xy_ctr;

  KOKKOS_INLINE_FUNCTION
  PlanarGaussian(const Real str = 1, const Real b = 0.25,
                      const Real x0 = 0, const Real y0 = 0) :
      strength(str),
      shape_parameter(b),
      xy_ctr({x0,y0}) {}

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y) const {
    const Real xy[2] = {x,y};
    return impl(xy);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const { return 0; }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& xy) const {
    return impl(xy);
  }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Real impl(const PtType& xy) const {
    const Real rsq = square(PlaneGeometry::distance(xy, xy_ctr));
    return strength * exp(-shape_parameter * rsq );
  }
};

/**
  Functor that outputs the negative Laplacian of the Gaussian function
  represented by PlanarGaussian defined with the same parameters.
*/
struct PlanarNegativeLaplacianOfGaussian {
  typedef PlaneGeometry geo;
  Real strength;
  Real shape_parameter;
  Kokkos::Tuple<Real,2> xy_ctr;

  KOKKOS_INLINE_FUNCTION
  PlanarNegativeLaplacianOfGaussian(const PlanarGaussian& g) :
    strength(g.strength),
    shape_parameter(g.shape_parameter),
    xy_ctr(g.xy_ctr) {}

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y) const {
    Real xy[2] = {x,y};
    return impl(xy);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const { return 0; }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& xy) const {
    return impl(xy);
  }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Real impl(const PtType& xy) const {
    const Real rsq = square(PlaneGeometry::distance(xy, xy_ctr));
    const Real coeff = 4 * strength * shape_parameter * (1 - shape_parameter*rsq);
    return coeff * exp(-shape_parameter * rsq);
  }
};

struct PlanarGaussianTestVelocity {
  typedef PlaneGeometry geo;
  using crd_view = typename PlaneGeometry::crd_view_type;
  using vec_view = typename PlaneGeometry::vec_view_type;

  vec_view u;
  scalar_view_type double_dot;
  crd_view x;

  Real zeta_strength;
  Real zeta_b;
  Kokkos::Tuple<Real,2> zeta_ctr;
  Real sigma_strength;
  Real sigma_b;
  Kokkos::Tuple<Real,2> sigma_ctr;

  PlanarGaussianTestVelocity(vec_view u, scalar_view_type dd,
    const crd_view x,
    const PlanarNegativeLaplacianOfGaussian& vorticity,
    const PlanarNegativeLaplacianOfGaussian& divergence) :
    u(u),
    double_dot(dd),
    x(x),
    zeta_strength(vorticity.strength),
    zeta_b(vorticity.shape_parameter),
    zeta_ctr(vorticity.xy_ctr),
    sigma_strength(divergence.strength),
    sigma_b(divergence.shape_parameter),
    sigma_ctr(divergence.xy_ctr) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
    const Real zeta_dist_sq = square(PlaneGeometry::distance(xi, zeta_ctr));
    const Real sigma_dist_sq = square(PlaneGeometry::distance(xi, sigma_ctr));
    const Real zeta_exp = exp(-zeta_b * zeta_dist_sq);
    const Real sigma_exp = exp(-sigma_b * sigma_dist_sq);

    u(i,0) = -2*zeta_strength * zeta_b * (xi(1) - zeta_ctr[1]) * zeta_exp +
              2*sigma_strength * sigma_b * (xi(0) - sigma_ctr[0]) * sigma_exp;
    u(i,1) =  2*zeta_strength * zeta_b * (xi(0) - zeta_ctr[0]) * zeta_exp +
              2*sigma_strength * sigma_b * (xi(1) - sigma_ctr[1]) * sigma_exp;

    double_dot(i) =
      square( 4 * zeta_strength * square(zeta_b) *
        (xi(0) - zeta_ctr[0]) * (xi(1) - zeta_ctr[1]) * zeta_exp +
        2 * sigma_strength * sigma_b *
        (1 - 2*sigma_b * square(xi(0) - sigma_ctr[0])) * sigma_exp ) +
      square( 4 * zeta_strength * square(zeta_b) *
        (xi(0) - zeta_ctr[0]) * (xi(1) - zeta_ctr[1]) * zeta_exp -
        2 * sigma_strength * sigma_b *
        (1 - 2*sigma_b * square(xi(1) - sigma_ctr[1])) * sigma_exp ) +
      (4 * zeta_strength * zeta_b *
        (1 - 2*zeta_b * square(xi(0) - zeta_ctr[0])) * zeta_exp -
        8 * sigma_strength * square(sigma_b) *
        (xi(0) - sigma_ctr[0]) * (xi(1) - sigma_ctr[1]) * sigma_exp ) *
      (-2 * zeta_strength * zeta_b *
        (1 - 2*zeta_b * square(xi(1) - zeta_ctr[1])) * zeta_exp -
        4 * sigma_strength * square(sigma_b) *
        (xi(0) - sigma_ctr[0]) * (xi(1) - sigma_ctr[1]) * sigma_exp );

  }
};

struct PlanarGravityWaveFreeBoundaries {
  typedef PlaneGeometry geo;
  Real mtn_height;
  Real mtn_width_param;
  Real mtn_ctr_x;
  Real mtn_ctr_y;
  Real sfc_height;
  Real sfc_ptb_max;
  Real sfc_ptb_width_bx;
  Real sfc_ptb_width_by;
  Real sfc_ptb_ctr_x;
  Real sfc_ptb_ctr_y;
  Real f0;
  Real beta;

  KOKKOS_INLINE_FUNCTION
  PlanarGravityWaveFreeBoundaries() :
    mtn_height(0.8),
    mtn_width_param(5),
    mtn_ctr_x(0),
    mtn_ctr_y(0),
    sfc_height(1),
    sfc_ptb_max(0.1),
    sfc_ptb_width_bx(20),
    sfc_ptb_width_by(5),
    sfc_ptb_ctr_x(-1.125),
    sfc_ptb_ctr_y(0),
    f0(0),
    beta(0) {}

  inline std::string name() const {return "PlanarGravityWaveFreeBoundaries";}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real bottom_height(const PtType& xy) const {
    return mtn_height*exp(-mtn_width_param*(square(xy[0]-mtn_ctr_x) + square(xy[1] - mtn_ctr_y)));
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sfc0(const PtType& xy) const { return
    sfc_height + sfc_ptb_max*exp(-(sfc_ptb_width_bx*square(xy(0) - sfc_ptb_ctr_x) + sfc_ptb_width_by*square(xy(1) - sfc_ptb_ctr_y)));
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real zeta0(const PtType& xy) const { return 0;}

  template <typename PtType, typename VecType>
  KOKKOS_INLINE_FUNCTION
  void u0(VecType& uv, const PtType& xy) const {
    uv[0] = 0;
    uv[1] = 0;
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sigma0(const PtType& xy) const { return 0;}
};


struct NitscheStricklandVortex {
  typedef PlaneGeometry geo;
  Real vortex_width_b;
  Real sfc_height;
  Real f0;
  Real beta;

  KOKKOS_INLINE_FUNCTION
  NitscheStricklandVortex(const Real b = 0.5, const Real sfc = 1, const Real f=0, const Real bb=0) :
    vortex_width_b(0.5),
    sfc_height(sfc),
    f0(f),
    beta(bb) {}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real zeta0(const PtType& xy) const {
    const Real rsq = square(xy[0]) + square(xy[1]);
    const Real r = sqrt(rsq);
    return (3 * safe_divide(r) - 2 * vortex_width_b * r) * rsq * std::exp(-vortex_width_b * rsq);
  }

  inline std::string name() const { return "Nitsche&Strickland"; }

  template <typename VecType, typename PtType>
  KOKKOS_INLINE_FUNCTION
  void u0(VecType& uv, const PtType& xy) const {
    const Real rsq = square(xy[0]) + square(xy[1]);
    const Real utheta = rsq * std::exp(-vortex_width_b * rsq);
    const Real theta = std::atan2(xy[1], xy[0]);
    uv[0] = -utheta * std::sin(theta);
    uv[1] = utheta * std::cos(theta);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sigma0(const PtType& xy) const {return 0;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sfc0(const PtType& xy) const {return sfc_height; }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real h0(const PtType& xy) const {return
    sfc0(xy);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real bottom_height(const PtType& xy) const {return 0;}
};

struct SWETestCase2 {
  typedef SphereGeometry geo;
  Real h0max;
  Real Omega;
  Real u0max;

  KOKKOS_INLINE_FUNCTION
  SWETestCase2(const Real hh=1, const Real Omg=2*constants::PI, const Real uu=constants::PI/6) :
    h0max(hh),
    Omega(Omg),
    u0max(uu) {}

  template <typename VecType, typename PtType>
  KOKKOS_INLINE_FUNCTION
  void u0(VecType& uvw, const PtType& xyz) const {
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real zonal_u = u0max*cos(lat);
    const Real merid_v = 0;
    uvw[0] = -zonal_u * sin(lon);
    uvw[1] =  zonal_u * cos(lon);
    uvw[2] = 0;
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real zeta0(const PtType& xyz) const {
    return 2*u0max*xyz[2];
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sfc0(const PtType& xyz) const {
    return h0max - (Omega*u0max + 0.5*square(u0max))*square(xyz[2])/constants::GRAVITY;
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real h0(const PtType& xyz) const {return
    sfc0(xyz);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sigma0(const PtType& xy) const {return 0;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real bottom_height(const PtType& xyz) const {return 0;}
};


} // namespace Lpm

#endif
