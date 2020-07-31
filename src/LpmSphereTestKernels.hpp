#ifndef LPM_SPHERE_TEST_KERNELS_HPP
#define LPM_SPHERE_TEST_KERNELS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {

struct SphereVelocityTangentTestFunctor {
  scalar_view_type udotx;
  SphereGeometry::crd_view_type x;
  SphereGeometry::vec_view_type u;

  SphereVelocityTangentTestFunctor(scalar_view_type& dp, const SphereGeometry::crd_view_type& pos,
    const SphereGeometry::vec_view_type& vel) : udotx(dp), x(pos), u(vel) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    const auto myx = ko::subview(x,i,ko::ALL());
    const auto myu = ko::subview(u,i,ko::ALL());
    udotx(i) = SphereGeometry::dot(myx,myu);
  }
};

}
#endif
