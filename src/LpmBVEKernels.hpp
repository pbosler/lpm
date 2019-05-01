#ifndef LPM_BVE_KERNELS_HPP
#define LPM_BVE_KERNELS_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmTimeIntegrator.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

using RealVec = Kokkos::Tuple<Real,3>;

typedef ko::View<const Real*[3],Dev> src_vec_view;
typedef ko::View<const Real*,Dev> src_scalar_view;
typedef typename SphereGeometry::crd_view_type tgt_vec_view;
typedef scalar_view_type tgt_scalar_view;
typedef ko::TeamPolicy<>::member_type member_type;

template <typename TgtVec, typename SrcVec> KOKKOS_INLINE_FUNCTION
void biotSavartSphere(TgtVec& u, const SrcVec& x, const SrcVec& xx, const Real& zeta, const Real& area, const Real& smoother = 0) {
  const Real strength = -4*PI / (1 - SphereGeometry::dot(x,xx) + square(smoother)) * zeta * area;
    SphereGeometry::cross(u, x, xx);
    for (int j=0; j<3; ++j) {
      u[j] *= strength;
    }
}

struct BVEDirectSumCollocated {
    typedef RealVec value_type;
    static constexpr Real eps = 0;
    typedef Index size_type;
    
    src_vec_view srclocs;
    src_scalar_view srcvort;
    src_scalar_view srcarea;
    Index i;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumCollocated(const Index ii, const src_vec_view x, const src_scalar_view zeta,
        const src_scalar_view area) : 
            i(ii), srclocs(x), srcvort(zeta), srcarea(area) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        RealVec vel;
        if (i != j) {
            auto x = ko::subview(srclocs, i, ko::ALL());
            auto xx = ko::subview(srclocs, j, ko::ALL());
            biotSavartSphere(vel.data, x, xx, srcvort(j), srcarea(j));
        }
        for (int k=0; k<3; ++k) {
            uu.data[k] += vel.data[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst.data[j] += src.data[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu[j] = 0;
        }
    }
};

struct BVEDirectSumDistinct {
    typedef RealVec value_type;
    static constexpr Real eps = 0;
    typedef Index size_type;
    
    src_vec_view srclocs;
    src_scalar_view srcvort;
    src_scalar_view srcarea;
    src_vec_view tgtlocs;
    Index i;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumDistinct(const Index ii, const src_vec_view x, const src_scalar_view zeta,  
        const src_scalar_view area, const src_vec_view xx, tgt_vec_view vel) :
        i(ii), srclocs(x), srcvort(zeta), srcarea(area), tgtlocs(xx) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        RealVec vel;
        auto x = ko::subview(tgtlocs, i, ko::ALL());
        auto xx = ko::subview(srclocs, j, ko::ALL());
        biotSavartSphere(vel.data, x, xx, srcvort(j), srcarea(j));
        for (int k=0; k<3; ++k) {
            uu.data[k] += vel.data[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst.data[j] += src.data[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu.data[j] = 0.0;
        }
    }
};

struct BVEDirectSumSmooth {
    typedef RealVec value_type;
    static constexpr Int value_count = 3;
    typedef Index size_type;
    
    Real eps;
    src_vec_view srclocs;
    src_scalar_view srcvort;
    src_scalar_view srcarea;
    src_vec_view tgtlocs;
    Index i;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumSmooth(const Index ii, const src_vec_view x, const src_scalar_view zeta, const src_scalar_view area,
        const src_vec_view xx, tgt_vec_view vel, const Real sm) : i(ii), srclocs(x), srcvort(zeta), srcarea(area),
        tgtlocs(xx), eps(sm) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        RealVec vel;
        auto x = ko::subview(tgtlocs, i, ko::ALL());
        auto xx = ko::subview(srclocs, j, ko::ALL());
        biotSavartSphere(vel.data, x, xx, srcvort(j), srcarea(j), eps);
        for (int k=0; k<3; ++k) {
            uu.data[k] += vel.data[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type dst, const volatile value_type src) const {
        for (int j=0; j<3; ++j) {
            dst.data[j] += src.data[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu.data[j] = 0.0;
        }
    }
};


struct BVEVertexVelocity {
    src_vec_view physVertCrds;
    src_vec_view physFaceCrds;
    src_scalar_view faceRelVort;
    src_scalar_view faceArea;
    tgt_vec_view vertVelocity;
    Index n;
    
    KOKKOS_INLINE_FUNCTION
    BVEVertexVelocity(const Index nfaces, const src_vec_view pvc, const src_vec_view pfc, const src_scalar_view zeta,
        const src_scalar_view area, tgt_vec_view u) : physVertCrds(pvc), physFaceCrds(pfc),
        faceRelVort(zeta), faceArea(area), vertVelocity(u), n(nfaces) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        RealVec vel;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr, n), BVEDirectSumDistinct(i, physFaceCrds, faceRelVort, faceArea,
            physVertCrds, vertVelocity), vel);
        for (int j=0; j<3; ++j) {
            vertVelocity(i, j) = vel.data[j];
        }
    }
};

struct BVEFaceVelocity {
    src_vec_view physFaceCrds;
    src_scalar_view faceRelVort;
    src_scalar_view faceArea;
    tgt_vec_view faceVelocity;
    Index n;
    
    KOKKOS_INLINE_FUNCTION
    BVEFaceVelocity(const Index nfaces, const src_vec_view pfc, const src_scalar_view zeta, const src_scalar_view area,
        tgt_vec_view u) : n(nfaces), physFaceCrds(pfc), faceRelVort(zeta), faceArea(area), faceVelocity(u) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        RealVec vel;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr,n), BVEDirectSumCollocated(i, physFaceCrds, faceRelVort, faceArea), vel);
        for (int j=0; j<3; ++j) {
            faceVelocity(i,j) = vel.data[j];
        }
    }
};

struct BVERK4 : public RK4<3,BVEVertexVelocity, BVEFaceVelocity> {
  typedef ko::View<Real*[3],Dev> vec_view_type;
    
    /// from caller
  scalar_view_type vertvort, facevort, facearea;
    Real Omega;
    
    /// internal
    scalar_view_type vertvortin, vertvortstage1, vertvortstage2, vertvortstage3, vertvortstage4;
    scalar_view_type facevortin, facevortstage1, facevortstage2, facevortstage3, facevortstage4;
    
    BVERK4(vec_view_type vs, scalar_view_type vz, vec_view_type vv, vec_view_type fs, scalar_view_type fz, vec_view_type fv,
        scalar_view_type fa, const Real omega) : RK4<3,BVEVertexVelocity,BVEFaceVelocity>(vs, fs, vv, fv), vertvort(vz),
        facevort(fz), facearea(fa),
        vertvortin("vertvortin",nv), vertvortstage1("vertvortstage1", nv), vertvortstage2("vertvortstage2", nv),
        vertvortstage3("vertvortstage3", nv), vertvortstage4("vertvortstage4", nv), facevortin("facevortin", nf),
        facevortstage1("facevortstage1", nf), 
        facevortstage2("facevortstage2", nf), facevortstage3("facevortstage3", nf), facevortstage4("facevortstage4", nf),
        Omega(omega) {}
    
    struct DZetaDt {
        vec_view_type u;
        scalar_view_type dzeta;
        Real Omega;
        
        DZetaDt(vec_view_type uu, scalar_view_type dz, const Real omg) : u(uu), dzeta(dz), Omega(omg) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const Index i) const {
            dzeta(i) = -2*Omega*u(i,2);
        }
    };  
    
    struct StageSetup {
        vec_view_type crds0;
        vec_view_type crdsnext;
        scalar_view_type zeta0;
        scalar_view_type zetanext;
        vec_view_type u;
        scalar_view_type dzeta;
        Int stageNum;
        Real dt;
        
        StageSetup(vec_view_type x0, vec_view_type xn, scalar_view_type z0, scalar_view_type zn, vec_view_type vel, scalar_view_type dz, 
            const Int stage, const Real dt_) : crds0(x0), crdsnext(xn), zeta0(z0), zetanext(zn), u(vel), dzeta(dz),
            stageNum(stage), dt(dt_) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const Index i) const {
            
            for (int j=0; j<3; ++j) {
                crdsnext(i,j) = crds0(i,j) +  dt*(stageNum == 4 ? 1 : 0.5)*u(i,j);
            }
            zetanext(i) = zeta0(i) + dt*(stageNum == 4 ? 1 : 0.5)*dzeta(i);
        }
    };  
        
    void timestep(const Real dt) const override {
        const Real dto3 = dt/3.0;
        const Real dto6 = dt/6.0;
        
        ko::TeamPolicy<> vertex_policy(nv, ko::AUTO());
        ko::TeamPolicy<> face_policy(nf, ko::AUTO());
        
        /// compute stage 1 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(nf, vertcrds, facecrds, facevort, facearea, vertstage1));
        ko::parallel_for(face_policy, BVEFaceVelocity(nf, facecrds, facevort, facearea, facestage1));
        /// stage 1 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage1, vertvortstage1, Omega));
        ko::parallel_for(nf, DZetaDt(facestage1, facevortstage1, Omega));
        
        /// stage2 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage1, vertvortstage1, 2, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage1, facevortstage1, 2, dt));
        /// stage 2 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(nf, vertinput, faceinput, facevortin, facearea, vertstage2));
        ko::parallel_for(face_policy, BVEFaceVelocity(nf, faceinput, facevortin, facearea, facestage2));
        /// stage 2 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage2, vertvortstage2, Omega));
        ko::parallel_for(nf, DZetaDt(facestage2, facevortstage2, Omega));

        /// stage 3 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage2, vertvortstage2, 3, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage2, facevortstage2, 3, dt));
        /// stage 3 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(nf, vertinput, faceinput, facevortin, facearea, vertstage3));
        ko::parallel_for(face_policy, BVEFaceVelocity(nf, faceinput, facevortin, facearea, facestage3));
        /// stage 3 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage3, vertvortstage3, Omega));
        ko::parallel_for(nf, DZetaDt(facestage3, facevortstage3, Omega));
        
        /// stage 4 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage3, vertvortstage3, 4, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage3, facevortstage3, 4, dt));
        /// stage 4 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(nf, vertinput, faceinput, facevortin, facearea, vertstage4));
        ko::parallel_for(face_policy, BVEFaceVelocity(nf, faceinput, facevortin, facearea, facestage4));
        /// stage 4 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage4, vertvortstage4, Omega));
        ko::parallel_for(nf, DZetaDt(facestage4, facevortstage4, Omega));
        
        /// Update
        ko::parallel_for(nv, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<3; ++j) {
                vertcrds(i,j) += dto6*vertstage1(i,j) + dto3*vertstage2(i,j) + dto3*vertstage3(i,j) + dto6*vertstage4(i,j);
            }
            vertvort(i) += dto6*vertvortstage1(i) + dto3*vertvortstage2(i) + dto3*vertvortstage3(i) + dto6*vertvortstage4(i);
        });
        ko::parallel_for(nf, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<3; ++j) {
                facecrds(i,j) += dto6*facestage1(i,j) + dto3*facestage2(i,j) + dto3*facestage3(i,j) + dto6*facestage4(i,j);
            }
            facevort(i) += dto6*facevortstage1(i) + dto3*facevortstage2(i) + dto3*facevortstage3(i) + dto6*facevortstage4(i);
        });
    } 
};

}
#endif
