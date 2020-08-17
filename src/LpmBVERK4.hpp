#ifndef LPM_RK4_HPP
#define LPM_RK4_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmBVEKernels.hpp"
#include "LpmGeometry.hpp"

namespace Lpm {

class BVERK4 {
  public :
    crd_view vertx;
    scalar_view_type vertvort;
    vec_view vertvel;

    crd_view facex;
    scalar_view_type facevort;
    vec_view facevel;

    Real dt;
    Real Omega;

    Index nverts;
    Index nfaces;

    BVERK4(const Real& timestep, const Real& omg) : dt(timestep), Omega(omg), nverts(0), nfaces(0) {}

    void init(const Index& nv, const Index& nf);

    void advance_timestep(crd_view& vx, scalar_view_type& vzeta, vec_view& vvel,
      crd_view& fx, scalar_view_type& fzeta, vec_view& fvel, const scalar_view_type& fa, const mask_view_type& fm);


  protected:
    scalar_view_type facearea;
    mask_view_type facemask;

    crd_view vertx1;
    crd_view vertx2;
    crd_view vertx3;
    crd_view vertx4;
    crd_view vertxwork;

    scalar_view_type vertvort1;
    scalar_view_type vertvort2;
    scalar_view_type vertvort3;
    scalar_view_type vertvort4;
    scalar_view_type vertvortwork;

    crd_view facex1;
    crd_view facex2;
    crd_view facex3;
    crd_view facex4;
    crd_view facexwork;

    scalar_view_type facevort1;
    scalar_view_type facevort2;
    scalar_view_type facevort3;
    scalar_view_type facevort4;
    scalar_view_type facevortwork;

};




}
#endif
