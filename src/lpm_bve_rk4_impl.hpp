#ifndef LPM_RK4_IMPL_HPP
#define LPM_RK4_IMPL_HPP

#include "LpmBVERK4.hpp"
#include "KokkosBlas.hpp"
#include <cassert>
#include "Kokkos_Core.hpp"

namespace Lpm {

struct BVERK4Update {
  static constexpr Real sixth = 1.0/6.0;
  static constexpr Real third = 1.0/3.0;

  crd_view x;
  crd_view x1;
  crd_view x2;
  crd_view x3;
  crd_view x4;

  scalar_view_type vort;
  scalar_view_type vort1;
  scalar_view_type vort2;
  scalar_view_type vort3;
  scalar_view_type vort4;

  BVERK4Update(crd_view& x_inout, const crd_view& xs1, const crd_view& xs2, const crd_view& xs3, const crd_view& xs4,
    scalar_view_type zeta_inout, const scalar_view_type& zs1, const scalar_view_type& zs2, const scalar_view_type& zs3,
    const scalar_view_type& zs4 ) : x(x_inout), x1(xs1), x2(xs2), x3(xs3), x4(xs4), vort(zeta_inout),
    vort1(zs1), vort2(zs2), vort3(zs3), vort4(zs4) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    for (Short j=0; j<3; ++j) {
      x(i,j) += sixth*(x1(i,j) + x4(i,j)) + third*(x2(i,j) + x3(i,j));
    }
    vort(i) += sixth*(vort1(i) + vort4(i)) + third*(vort2(i) + vort3(i));
  }
};


void BVERK4::advance_timestep(crd_view& vx, scalar_view_type& vzeta, vec_view& vvel,
      crd_view& fx, scalar_view_type& fzeta, vec_view& fvel, const scalar_view_type& fa, const mask_view_type& fm) {

  ko::Profiling::pushRegion("BVERK4::advance_timestep");

  ko::TeamPolicy<> vertex_policy(nverts, ko::AUTO());
  ko::TeamPolicy<> face_policy(nfaces, ko::AUTO());

  vertx = vx;
  vertvort = vzeta;
  vertvel = vvel;

  facex = fx;
  facevort = fzeta;
  facevel = fvel;

  facearea = fa;
  facemask = fm;

  /// RK Stage 1
//   KokkosBlas::axpby(dt, vertvel, 0.0, vertx1);
  KokkosBlas::scal(vertx1, dt, vertvel);
  ko::parallel_for("RK4-1 vertex vorticity", nverts, BVEVorticityTendency(vertvort1, vertvel, dt, Omega));

//   KokkosBlas::axpby(dt, facevel, 0.0, facex1);
  KokkosBlas::scal(facex1, dt, facevel);
  ko::parallel_for("RK4-1 face vorticity", nfaces, BVEVorticityTendency(facevort1, facevel, dt, Omega));

  /// RK Stage 2
  KokkosBlas::update(1.0, vertx, 0.5, vertx1, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 0.5, vertvort1, 0.0, vertvortwork);

  KokkosBlas::update(1.0, facex, 0.5, facex1, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 0.5, facevort1, 0.0, facevortwork);

  ko::parallel_for("RK4-2 vertex velocity", vertex_policy,
    BVEVertexVelocity(vertvel, vertxwork, facexwork, facevortwork, facearea, facemask, nfaces));
  ko::parallel_for("RK4-2 face velocity", face_policy,
    BVEFaceVelocity(  facevel, facexwork, facevortwork, facearea, facemask, nfaces));
  KokkosBlas::scal(vertx2, dt, vertvel);
  KokkosBlas::scal(facex2, dt, facevel);
  ko::parallel_for("RK4-2 vertex vorticity", nverts, BVEVorticityTendency(vertvort2, vertvel, dt, Omega));
  ko::parallel_for("RK4-2 face vorticity", nfaces, BVEVorticityTendency(facevort2, facevel, dt, Omega));

  /// RK Stage 3
  KokkosBlas::update(1.0, vertx, 0.5, vertx2, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 0.5, vertvort2, 0.0, vertvortwork);

  KokkosBlas::update(1.0, facex, 0.5, facex2, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 0.5, facevort2, 0.0, facevortwork);

  ko::parallel_for("RK4-3 vertex velocity", vertex_policy,
    BVEVertexVelocity(vertvel, vertxwork, facexwork, facevortwork, facearea, facemask, nfaces));
  ko::parallel_for("RK4-3 face velocity",face_policy,
    BVEFaceVelocity(facevel, facexwork, facevortwork, facearea, facemask, nfaces));
  KokkosBlas::scal(vertx3, dt, vertvel);
  KokkosBlas::scal(facex3, dt, facevel);
  ko::parallel_for("RK4-3 vertex vorticity", nverts, BVEVorticityTendency(vertvort3, vertvel, dt, Omega));
  ko::parallel_for(nfaces, BVEVorticityTendency(facevort3, facevel, dt, Omega));

  /// RK Stage 4
  KokkosBlas::update(1.0, vertx, 1.0, vertx3, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 1.0, vertvort3, 0.0, vertvortwork);

  KokkosBlas::update(1.0, facex, 1.0, facex3, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 1.0, facevort3, 0.0, facevortwork);

  ko::parallel_for("RK4-4 vertex velocity", vertex_policy,
    BVEVertexVelocity(vertvel, vertxwork, facexwork, facevortwork, facearea, facemask, nfaces));
  ko::parallel_for("RK4-4 face velocity",face_policy,
    BVEFaceVelocity(facevel, facexwork, facevortwork, facearea, facemask, nfaces));
  KokkosBlas::scal(vertx4, dt, vertvel);
  KokkosBlas::scal(facex4, dt, facevel);
  ko::parallel_for("RK4-4 vertex vorticity", nverts, BVEVorticityTendency(vertvort4, vertvel, dt, Omega));
  ko::parallel_for("RK4-4 face vorticity", nfaces, BVEVorticityTendency(facevort4, facevel, dt, Omega));

  ko::parallel_for("RK4 vertex update", nverts,
    BVERK4Update(vertx, vertx1, vertx2, vertx3, vertx4, vertvort, vertvort1, vertvort2, vertvort3, vertvort4));
  ko::parallel_for("RK4 face update", nfaces,
    BVERK4Update(facex, facex1, facex2, facex3, facex4, facevort, facevort1, facevort2, facevort4, facevort4));

  ko::parallel_for("RK4-0 vertex velocity", vertex_policy,
    BVEVertexVelocity(vertvel, vertx, facex, facevort, facearea, facemask, nfaces));
  ko::parallel_for("RK4-0 face velocity", face_policy,
    BVEFaceVelocity(facevel, facex, facevort, facearea, facemask, nfaces));


  ko::Profiling::popRegion();
}


}
#endif
