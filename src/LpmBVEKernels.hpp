#ifndef LPM_BVE_KERNELS_HPP
#define LPM_BVE_KERNELS_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmTimeIntegrator.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

typedef ko::View<const Real*[3],Dev> const_vec_view;
typedef ko::View<const Real*,Dev> const_scalar_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef scalar_view_type scalar_view;
typedef ko::TeamPolicy<>::member_type member_type;

/**
    Biot-Savart Kernel for the sphere.
        Computes contribution of source vortex located at xx with circulation zeta(xx) * area(xx) to tgt location x.    
        
        u(x) = cross(x, xx)*zeta(xx)*area(xx);

    Ref: Kimura & Okamoto 1987.
*/
template <typename TgtVec, typename SrcVec> KOKKOS_INLINE_FUNCTION
void biotSavartSphere(TgtVec& u, const SrcVec& x, const SrcVec& xx, const Real& zeta, const Real& area, const Real& smoother = 0) {
  const Real strength = - zeta * area / (4*PI * (1 - SphereGeometry::dot(x,xx) + square(smoother)));
    ko::Tuple<Real,3> tup = SphereGeometry::cross(x, xx);
    for (int j=0; j<3; ++j) {
      u[j] = strength * tup[j];
    }
}

/**
    Computational kernel for || reduction over collocated point vortices (singular vortices, no smoothing).
        The singular point is skipped by not allowing self interaction (require tgt i != src j).
    
    i = index of target
    xx srclocs = coordinates of point vortices
    zeta(xx) srcvort = vorticity of point vortices
    area(xx) srcarea = area of point vortices
    x tgtlocs = srclocs (collocated)
    
*/
struct BVEDirectSumCollocated {
    typedef ko::Tuple<Real,3> value_type;
    static constexpr Real eps = 0;
    
    Index i;
    const_vec_view srclocs;
    const_scalar_view srcvort;
    const_scalar_view srcarea;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumCollocated(const Index ii, const const_vec_view xx, const const_scalar_view zeta,
        const const_scalar_view area) : 
            i(ii), srclocs(xx), srcvort(zeta), srcarea(area) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        value_type vel;
        if (i != j) {
            auto x = ko::subview(srclocs, i, ko::ALL());
            auto xx = ko::subview(srclocs, j, ko::ALL());
            biotSavartSphere(vel, x, xx, srcvort(j), srcarea(j));
        }
        for (int k=0; k<3; ++k) {
            uu[k] += vel[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst[j] += src[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu[j] = 0;
        }
    }
};

/**
    Computational kernel for || reduction over point vortices (singular vortices, no smoothing) whose tgt points
        are known to be distinct from src locations a priori.
        The singular point is skipped by definition.
    
    i = index of target
    xx srclocs = coordinates of point vortices
    zeta(xx) srcvort = vorticity of point vortices
    area(xx) srcarea = area of point vortices
    x tgtlocs = coordinates of evaluation points, tgtlocs(i,:) != srclocs(j,:) for all i,j
*/
struct BVEDirectSumDistinct {
    typedef ko::Tuple<Real,3> value_type;
    static constexpr Real eps = 0;
    
    Index i;
    const_vec_view tgtlocs;
    const_vec_view srclocs;
    const_scalar_view srcvort;
    const_scalar_view srcarea;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumDistinct(const Index ii, const const_vec_view x, const const_vec_view xx, const const_scalar_view zeta,  
        const const_scalar_view area) : i(ii), tgtlocs(x), srclocs(xx), srcvort(zeta), srcarea(area) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        value_type vel;
        auto x = ko::subview(tgtlocs, i, ko::ALL());
        auto xx = ko::subview(srclocs, j, ko::ALL());
        biotSavartSphere(vel, x, xx, srcvort(j), srcarea(j));
        for (int k=0; k<3; ++k) {
            uu[k] += vel[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst[j] += src[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu[j] = 0.0;
        }
    }
};

/**
    Computational kernel for || reduction over blob vortices (smoothing) whose tgt points
        are arbitrary and possible near source points.
        The singular point is regularized.
    
    i = index of target
    xx srclocs = coordinates of point vortices
    zeta(xx) srcvort = vorticity of point vortices
    area(xx) srcarea = area of point vortices
    x tgtlocs = coordinates of evaluation points, tgtlocs(i,:) != srclocs(j,:) for all i,j
*/
struct BVEDirectSumSmooth {
    typedef ko::Tuple<Real,3> value_type;
        
    Index i;
    const_vec_view tgtlocs;
    const_vec_view srclocs;
    const_scalar_view srcvort;
    const_scalar_view srcarea;
    Real eps;
    
    KOKKOS_INLINE_FUNCTION
    BVEDirectSumSmooth(const Index ii, const const_vec_view x, const_vec_view xx, 
        const const_scalar_view zeta, const const_scalar_view area, const Real sm) : 
        i(ii), tgtlocs(x), srclocs(xx), srcvort(zeta), srcarea(area), eps(sm) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& uu) const {
        value_type vel;
        auto x = ko::subview(tgtlocs, i, ko::ALL());
        auto xx = ko::subview(srclocs, j, ko::ALL());
        biotSavartSphere(vel, x, xx, srcvort(j), srcarea(j), eps);
        for (int k=0; k<3; ++k) {
            uu[k] += vel[k];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst[j] += src[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& uu) {
        for (int j=0; j<3; ++j) {
            uu[j] = 0.0;
        }
    }
};

/**
    Computational kernel for || evaluation of velocities at mesh vertices where mesh faces are the source points.
        For the BVE (which is incompressible) vertices are always distinct from faces.
    
    physVertCrds = tgt locations in physical space
    physFaceCrds = src locations in physical space
    faceRelVort = src vorticity
    faceArea = src area
    vertVelocity = computed velocities at tgt locations
    n = number of sources
*/
struct BVEVertexVelocity {
    const_vec_view physVertCrds;
    const_vec_view physFaceCrds;
    const_scalar_view faceRelVort;
    const_scalar_view faceArea;
    vec_view vertVelocity;
    Index n;
    
    KOKKOS_INLINE_FUNCTION
    BVEVertexVelocity(const const_vec_view pvc, const const_vec_view pfc, const const_scalar_view zeta,
        const const_scalar_view area, vec_view u, const Index nf) : physVertCrds(pvc), physFaceCrds(pfc),
        faceRelVort(zeta), faceArea(area), vertVelocity(u), n(nf) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        ko::Tuple<Real,3> vel;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr, n), BVEDirectSumDistinct(i, physVertCrds, physFaceCrds, faceRelVort, faceArea), vel);
        for (int j=0; j<3; ++j) {
            vertVelocity(i,j) = vel[j];
        }
    }
};

/**
    Computational kernel for || evaluation of velocities at mesh faces where mesh faces are the source points.
        Evaluation and source points are collocated.
    
    physFaceCrds = src locations in physical space
    faceRelVort = src vorticity
    faceArea = src area
    faceVelocity = computed velocities at tgt locations
    n = number of sources
*/
struct BVEFaceVelocity {
    const_vec_view physFaceCrds;
    const_scalar_view faceRelVort;
    const_scalar_view faceArea;
    vec_view faceVelocity;
    Index n;
    
    KOKKOS_INLINE_FUNCTION
    BVEFaceVelocity(const const_vec_view pfc, const const_scalar_view zeta, const const_scalar_view area, 
        vec_view u, const Index nf) : physFaceCrds(pfc), faceRelVort(zeta), faceArea(area), faceVelocity(u), n(nf) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        ko::Tuple<Real,3> vel;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr,n), BVEDirectSumCollocated(i, physFaceCrds, faceRelVort, faceArea), vel);
        for (int j=0; j<3; ++j) {
            faceVelocity(i,j) = vel[j];
        }
    }
};

/**
    RK4 integrator for a Lagrangian particle mesh BVE solver on the sphere.
    
    Position update data are inherited from LpmTimeIntegrator.hpp, RK4<3,BVEVertexVelocity, BVEFaceVelocity>.
    
    This class adds data for vorticity dynamics on a rotating sphere.
    
    vertvort = vorticity at mesh vertex particles (in/out)
    facevort = vorticity at mesh face particles (in/out)
    facearea = area of mesh faces (constant --- flow is incompressible)
*/
struct BVERK4 : public RK4<3,BVEVertexVelocity, BVEFaceVelocity> {
    scalar_view vertvort, facevort;
    const_scalar_view facearea;
    Real Omega;
    
    
    BVERK4(vec_view vs, scalar_view vz, vec_view vv, vec_view fs, scalar_view fz, vec_view fv,
        const const_scalar_view fa, const Real omega) : 
        RK4<3,BVEVertexVelocity,BVEFaceVelocity>(vs, fs, vv, fv), // parent handles vert,face crds and velocities
        vertvort(vz), facevort(fz), facearea(fa),
        vertvortin("vertvortin",nv), vertvortstage1("vertvortstage1", nv), vertvortstage2("vertvortstage2", nv),
        vertvortstage3("vertvortstage3", nv), vertvortstage4("vertvortstage4", nv), 
        facevortin("facevortin", nf), facevortstage1("facevortstage1", nf), 
        facevortstage2("facevortstage2", nf), facevortstage3("facevortstage3", nf), facevortstage4("facevortstage4", nf),
        Omega(omega) {}
        
    void timestep(const Real dt) const override {        
        /// compute stage 1 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(vertcrds, facecrds, facevort, facearea, vertstage1, nf));
        ko::parallel_for(face_policy, BVEFaceVelocity(facecrds, facevort, facearea, facestage1, nf));
        /// stage 1 vorticity
        ko::parallel_for(nv, DZetaDt(vertvel, vertvortstage1, Omega));
        ko::parallel_for(nf, DZetaDt(facevel, facevortstage1, Omega));
        
        /// stage2 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage1, vertvortstage1, 2, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage1, facevortstage1, 2, dt));
        /// stage 2 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(vertinput, faceinput, facevortin, facearea, vertstage2, nf));
        ko::parallel_for(face_policy, BVEFaceVelocity(faceinput, facevortin, facearea, facestage2, nf));
        /// stage 2 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage2, vertvortstage2, Omega));
        ko::parallel_for(nf, DZetaDt(facestage2, facevortstage2, Omega));

        /// stage 3 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage2, vertvortstage2, 3, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage2, facevortstage2, 3, dt));
        /// stage 3 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(vertinput, faceinput, facevortin, facearea, vertstage3, nf));
        ko::parallel_for(face_policy, BVEFaceVelocity(faceinput, facevortin, facearea, facestage3, nf));
        /// stage 3 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage3, vertvortstage3, Omega));
        ko::parallel_for(nf, DZetaDt(facestage3, facevortstage3, Omega));
        
        /// stage 4 setup
        ko::parallel_for(nv, StageSetup(vertcrds, vertinput, vertvort, vertvortin, vertstage3, vertvortstage3, 4, dt));
        ko::parallel_for(nf, StageSetup(facecrds, faceinput, facevort, facevortin, facestage3, facevortstage3, 4, dt));
        /// stage 4 velocity
        ko::parallel_for(vertex_policy, BVEVertexVelocity(vertinput, faceinput, facevortin, facearea, vertstage4, nf));
        ko::parallel_for(face_policy, BVEFaceVelocity(faceinput, facevortin, facearea, facestage4, nf));
        /// stage 4 vorticity
        ko::parallel_for(nv, DZetaDt(vertstage4, vertvortstage4, Omega));
        ko::parallel_for(nf, DZetaDt(facestage4, facevortstage4, Omega));
        
        /// Update
        ko::parallel_for(nv, RK4Update(vertcrds, vertstage1, vertstage2, vertstage3, vertstage4, 
            vertvort, vertvortstage1, vertvortstage2, vertvortstage3, vertvortstage4, dt));
        ko::parallel_for(nf, RK4Update(facecrds, facestage1, facestage2, facestage3, facestage4,
            facevort, facevortstage1, facevortstage2, facevortstage3, facevortstage4, dt));
    } 
    
    /// internal
        scalar_view_type vertvortin, vertvortstage1, vertvortstage2, vertvortstage3, vertvortstage4;
        scalar_view_type facevortin, facevortstage1, facevortstage2, facevortstage3, facevortstage4;
        
        /**
            Functor to set up the inputs to the next RK4 stage.
            
            crds0 : initial (t=t0) coordinates (input)
            crdsnext : new crds for next rk stage (output)
            zeta0 : initial (t=t0) vorticity (input)
            zetanext : new vorticity for next rk stage (output)
            u : velocity (input) from previous rk stage
            dzeta : vorticity time derivative from previous rk stage
            stageNum : integer in [1,4]
            dt : time step
        */
        struct StageSetup {
            vec_view_type crds0;
            vec_view_type crdsnext;
            scalar_view_type zeta0;
            scalar_view_type zetanext;
            vec_view_type u;
            scalar_view_type dzeta;
            Int stageNum;
            Real dt;
        
            StageSetup(vec_view_type x0, vec_view_type xn, scalar_view_type z0, scalar_view_type zn, 
                vec_view_type vel, scalar_view_type dz, const Int stage, const Real dt_) : 
                crds0(x0), crdsnext(xn), zeta0(z0), zetanext(zn), u(vel), dzeta(dz),
                stageNum(stage), dt(dt_) {}
        
            KOKKOS_INLINE_FUNCTION
            void operator() (const Index i) const {
                for (int j=0; j<3; ++j) {
                    crdsnext(i,j) = crds0(i,j) +  dt*(stageNum == 4 ? 1 : 0.5)*u(i,j);
                }
                zetanext(i) = zeta0(i) + dt*(stageNum == 4 ? 1 : 0.5)*dzeta(i);
            }
        }; 

        /**
            Functor to compute the vorticity derivative due to Coriolis force
            
                D\zeta/Dt = -2*Omega*w
            
            u : velocity (input) vector (u,v,w) in R3
            dzeta : vorticity derivative (output)
            Omega : rotation rate of the sphere about the z-axis
        */    
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
        
        /** 
            Functor to compute the t=t0+dt solutions using previously computed stages.
            
            crds : in t=t0, out t=t0+dt
            zeta : in t=t0, out t=t0+dt
        */
        struct RK4Update {
            vec_view crds;
            const_vec_view crds1;
            const_vec_view crds2;
            const_vec_view crds3;
            const_vec_view crds4;
            scalar_view zeta;
            const_scalar_view zeta1;
            const_scalar_view zeta2;
            const_scalar_view zeta3;
            const_scalar_view zeta4;
            Real dto3;
            Real dto6;
            
            RK4Update(vec_view x, const_vec_view x1, const_vec_view x2, const_vec_view x3, const_vec_view x4,
                      scalar_view z, const_scalar_view z1, const_scalar_view z2, const_scalar_view z3, const_scalar_view z4, 
                      const Real delt) : crds(x), crds1(x1), crds2(x2), crds3(x3), crds4(x4), zeta(z), zeta1(z1),
                      zeta2(z2), zeta3(z3), zeta4(z4), dto3(delt/3.0), dto6(delt/6.0) {}
            
            KOKKOS_INLINE_FUNCTION
            void operator() (const Index& i) const {
                for (int j=0; j<3; ++j) {
                    crds(i,j) += dto6*crds1(i,j) + dto3*crds2(i,j) + dto3*crds3(i,j) + dto6*crds4(i,j);
                }
                zeta(i) += dto6*zeta1(i) + dto3*zeta2(i) + dto3*zeta3(i) + dto6*zeta4(i);
            }
                        
        };
};

}
#endif
