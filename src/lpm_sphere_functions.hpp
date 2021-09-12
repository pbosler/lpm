#ifndef LPM_SPHERE_FUNCTIONS_HPP
#define LPM_SPHERE_FUNCTIONS_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"

#include "Kokkos_Core.hpp"

namespace Lpm {

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


/** @brief functor computes the dot product of a vector field with the local
  normal vector to the sphere.  Tangency is verified if the dot product = 0.
*/
struct SphereTangentFunctor {
  scalar_view_type udotx;
  SphereGeometry::crd_view_type x;
  SphereGeometry::vec_view_type u;

  SphereTangentFunctor(scalar_view_type& dp, const SphereGeometry::crd_view_type& pos,
    const SphereGeometry::vec_view_type& vel) : udotx(dp), x(pos), u(vel) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    const auto myx = ko::subview(x,i,ko::ALL());
    const auto myu = ko::subview(u,i,ko::ALL());
    udotx(i) = SphereGeometry::dot(myx,myu);
  }
};

} // namespace Lpm

#endif
