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
    
    /// internal
    vec_view_type vertinput, vertstage1, vertstage2, vertstage3, vertstage4;
    vec_view_type faceinput, facestage1, facestage2, facestage3, facestage4;
    
    RK4(vec_view_type vs, vec_view_type fs, vec_view_type vv, vec_view_type fv) : nv(vs.extent(0)), nf(fs.extent(0)), 
        vertcrds(vs), facecrds(fs), vertvel(vv), facevel(fv),
        vertinput("vertinput", nv), vertstage1("vertstage1", nv),
        vertstage2("vertstage2", nv), vertstage3("vertstage3", nv),
        vertstage4("vertstage4", nv), faceinput("faceinput", nf),
        facestage1("facestage1", nv), facestage2("facestage2", nf),
        facestage3("facestage3", nf), facestage4("facestage4", nf) {}
    
  virtual void timestep(const Real dt) const = 0;

};




}
#endif
