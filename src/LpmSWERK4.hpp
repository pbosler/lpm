#ifndef LPM_SWE_RK4_HPP
#define LPM_SWE_RK4_HPP

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmShallowWater.hpp"

namespace Lpm {

template <typename SeedType, typename ProblemType>
class SWERK4 {
 public:
  static constexpr Real sixth = 1.0 / 6.0;
  static constexpr Real third = 1.0 / 3.0;

  typedef typename SeedType::geo::crd_view_type crd_view;
  typedef typename SeedType::geo::vec_view_type vec_view;

  crd_view vertx;
  vec_view vertvel;
  scalar_view_type vertvort;
  scalar_view_type vertdiv;
  scalar_view_type vertsfc;
  scalar_view_type vertdepth;
  scalar_view_type verttopo;

  crd_view facex;
  vec_view facevel;
  scalar_view_type facevort;
  scalar_view_type facediv;
  scalar_view_type facesfc;
  scalar_view_type facedepth;
  scalar_view_type facetopo;
  scalar_view_type facearea;

  Real dt;
  Real f0;
  Real beta;
  Real Omega;
  Real g;
  Real eps_pse;

  Index nverts;
  Index nfaces;

  SWERK4(const std::shared_ptr<ShallowWater<SeedType>> pm, const Real& tstep,
         const Real& eps)
      : vertx(pm->physVerts.crds),
        vertvel(pm->velocityVerts),
        vertvort(pm->relVortVerts),
        vertdiv(pm->divVerts),
        vertsfc(pm->surfaceHeightVerts),
        vertdepth(pm->depthVerts),
        verttopo(pm->topoVerts),
        facex(pm->physFaces.crds),
        facevel(pm->velocityFaces),
        facevort(pm->relVortFaces),
        facediv(pm->divFaces),
        facesfc(pm->surfaceHeightFaces),
        facedepth(pm->depthFaces),
        facetopo(pm->topoFaces),
        facearea(pm->faces.area),
        dt(tstep),
        f0(ProblemType::f0),
        beta(ProblemType::beta),
        Omega(ProblemType::OMEGA),
        g(ProblemType::g),
        eps_pse(eps),
        nverts(pm->nvertsHost()),
        nfaces(pm->nfacesHost()),
        facemass(pm->massFaces),
        facemask(pm->faces.mask) {
    init();
  }

  void advance_timestep();

  struct PositionUpdate {
    crd_view x;
    crd_view x1;
    crd_view x2;
    crd_view x3;
    crd_view x4;

    PositionUpdate(crd_view& x_, const crd_view& x1_, const crd_view& x2_,
                   const crd_view& x3_, const crd_view& x4_)
        : x(x_), x1(x1_), x2(x2_), x3(x3_), x4(x4_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Index& i, const Index& j) const {
      x(i, j) += sixth * (x1(i, j) + x4(i, j)) + third * (x2(i, j) + x3(i, j));
    }
  };

  struct ScalarUpdate {
    scalar_view_type f;
    scalar_view_type f1;
    scalar_view_type f2;
    scalar_view_type f3;
    scalar_view_type f4;

    ScalarUpdate(scalar_view_type& f_, const scalar_view_type& f1_,
                 const scalar_view_type& f2_, const scalar_view_type& f3_,
                 const scalar_view_type& f4_)
        : f(f_), f1(f1_), f2(f2_), f3(f3_), f4(f4_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Index& i) const {
      f(i) += sixth * (f1(i) + f4(i)) + third * (f2(i) + f3(i));
    }
  };

  std::string infoString(const std::string& label = "",
                         const int& tab_level = 0) const;

 protected:
  mask_view_type facemask;
  scalar_view_type facemass;

  void init();

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

  scalar_view_type vertdiv1;
  scalar_view_type vertdiv2;
  scalar_view_type vertdiv3;
  scalar_view_type vertdiv4;
  scalar_view_type vertdivwork;

  scalar_view_type verth1;
  scalar_view_type verth2;
  scalar_view_type verth3;
  scalar_view_type verth4;
  scalar_view_type verthwork;

  scalar_view_type vertddot;
  scalar_view_type vertlaps;

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

  scalar_view_type facediv1;
  scalar_view_type facediv2;
  scalar_view_type facediv3;
  scalar_view_type facediv4;
  scalar_view_type facedivwork;

  scalar_view_type faceddot;
  scalar_view_type facelaps;

  scalar_view_type facearea1;
  scalar_view_type facearea2;
  scalar_view_type facearea3;
  scalar_view_type facearea4;
  scalar_view_type faceareawork;

  void update_sfc();

  void compute_direct_sums();

  std::unique_ptr<ko::TeamPolicy<>> vertex_policy;
  std::unique_ptr<ko::TeamPolicy<>> face_policy;
};

}  // namespace Lpm
#endif
