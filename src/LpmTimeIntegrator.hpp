#ifndef LPM_TIMETEGRATOR_HPP
#define LPM_TIMETEGRATOR_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {

template <int ndim, typename VertVelFunctor, typename FaceVelFunctor> struct RK4
{
    typedef ko::View<Real*[ndim],Dev> vec_view_type;
    
    virtual ~RK4() {}
    
    /// from caller
    Index nv, nf;
    vec_view_type vertcrds, facecrds, vertvel, facevel;
    scalar_view_type facearea, faceconvolution;
    
    /// internal
    vec_view_type vertinput, vertstage1, vertstage2, vertstage3, vertstage4;
    vec_view_type faceinput, facestage1, facestage2, facestage3, facestage4;
    
    RK4(vec_view_type vs, vec_view_type fs, vec_view_type vv, vec_view_type fv,
        scalar_view_type f, scalar_view_type a) : nv(vs.extent(0)), nf(fs.extent(0)), 
        vertcrds(vs), facecrds(fs), vertvel(vv), facevel(fv),
        faceconvolution(f), facearea(a),
        vertinput("vertinput", nv), vertstage1("vertstage1", nv),
        vertstage2("vertstage2", nv), vertstage3("vertstage3", nv),
        vertstage4("vertstage4", nv), faceinput("faceinput", nf),
        facestage1("facestage1", nv), facestage2("facestage2", nf),
        facestage3("facestage3", nf), facestage4("facestage4", nf) {}
    
    virtual void timestep(const Real dt) const {
        
        const Real dto6 = dt/6.0;
        const Real dto3 = dt/3.0;
        
        // commpute stage 1
        ko::parallel_for(nv, VertVelFunctor(nf, vertcrds, facecrds,
            faceconvolution, facearea, vertstage1));
        ko::parallel_for(nf, FaceVelFunctor(nf, facecrds, faceconvolution, 
            facearea, facestage1);
        
        /// setup for stage 2
        ko::parallel_for(nv, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<ndim; ++j) {
                vertinput(i,j) = vertcrds(i,j) + 0.5*dt*vertstage1(i,j);
            }
        });
        ko::parallel_for(nf, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<ndim; ++j) {
                faceinput(i,j) = facecrds(i,j) + 0.5*dt*facestage1(i,j);
            }
        });
        
        /// compute stage 2
        ko::parallel_for(nv, VertVelFunctor(nf, vertinput, faceinput, 
            faceconvolution, facearea, vertstage2));
        ko::parallel_for(nf, FaceVelFunctor(nf, faceinput, faceconvolution, 
            facearea, facestage2));
        
        /// setup for stage 3
        ko::parallel_for(nv, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<ndim; ++j) {
                vertinput(i,j) = vertcrds(i,j) + 0.5*dt*vertstage2(i,j);
            }
        });
        ko::parallel_for(nf, KOKKOS_LAMBDA (const Index i) {
            for (int j=0; j<ndim; ++j) {
                faceinput(i,j) = facecrds(i,j) + 0.5*dt*facestage2(i,j);
            }
        });
    
        /// compute stage 3
        ko::parallel_for(nv, VertVelFunctor(nf, vertinput, faceinput, 
            faceconvolution, facearea, vertstage3));
        ko::parallel_for(nf, FaceVelFunctor(nf, faceinput, faceconvolution, 
            facearea, facestage3));
        
        /// setup for stage 4
        ko::parallel_for(nv, KOKKOS_LAMBDA (const Int i) {
            for (int j=0; j<ndim; ++j) {
                vertinput(i,j) = vertcrds(i,j) + dt*vertstage3(i,j);
            }
        });
        ko::parallel_for(nf, KOKKOS_LAMBDA (const Int i) {
            for (int j=0; j<ndim; ++j) {
                faceinput(i,j) = facecrds(i,j) + dt*facestage3(i,j);
            }
        });
        
        /// compute stage 4
        ko::parallel_for(nv, VertVelFunctor(nf, vertinput, faceinput,
            faceconvolution, facearea, vertstage4));
        ko::parallel_for(nf, FaceVelFunctor(nf, faceinput, faceconvolution,
            facearea, facestage4));
        
        /// update
        ko::parallel_for(nv, KOKKOS_LAMBDA (const Int i) {
            for (int j=0; j<ndim; ++j) {
                vertcrds(i,j) += dto6*vertstage1(i,j) + dto3*vertstage2(i,j) + 
                    dto3*vertstage3(i,j) + dto6*vertstage4(i,j);
            }
        });
        ko::parallel_for(nf, KOKKOS_LAMBDA (const Int i) {
            for (int j=0; j<ndim; ++j) {
                facecrds(i,j) += dto6*facestage1(i,j) + dto3*facestage2(i,j) + 
                    dto3*facestage3(i,j) + dto6*facestage4(i,j);
            }
        }
        ko::parallel_for(nv, VertVelFunctor(nf, vertcrds, facecrds,
            faceconvolution, facearea, vertvel));
        ko::parallel_for(nf, FaceVelFunctor(nf, facecrds, faceconvolution, 
            facearea, facevel));
    }
};




}
#endif
