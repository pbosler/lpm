#ifndef LPM_TIMETEGRATOR_HPP
#define LPM_TIMETEGRATOR_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

/**
    RK4 Abstract Factory for Lagrangian particles meshes.
    
    On input: Velocities are presumed set, so that the stage 1 position update is determined by them.
    
    On output: coordinates are updated to new locations, velocity is evaluated at those locations.
    
    vertcrds = mesh vertex coordinates (in/out)
    facecrds = mesh face coordinates (in/out)
    vertvel = mesh vertex velocities (in/out)
    facevel = mesh face velocities (in/out)
*/
template <int ndim, typename VertVelFunctor, typename FaceVelFunctor> class RK4
{
    public:
    typedef ko::View<Real*[ndim],Dev> vec_view_type;
    ko::TeamPolicy<DevExe> vertex_policy;
    ko::TeamPolicy<DevExe> face_policy;
    static constexpr Int nthreads_per_team = 1; /// currently unused.  See LpmKokkosUtil.hpp
    
    virtual ~RK4() {}
    
    /// from caller
    vec_view_type vertcrds, facecrds, vertvel, facevel;
        
    RK4(vec_view_type vs, vec_view_type fs, vec_view_type vv, vec_view_type fv) : 
        nv(vs.extent(0)), nf(fs.extent(0)), 
        vertcrds(vs), facecrds(fs), vertvel(vv), facevel(fv),
        vertinput("vertinput", nv), vertstage1("vertstage1", nv),
        vertstage2("vertstage2", nv), vertstage3("vertstage3", nv),
        vertstage4("vertstage4", nv), faceinput("faceinput", nf),
        facestage1("facestage1", nf), facestage2("facestage2", nf),
        facestage3("facestage3", nf), facestage4("facestage4", nf),
        vertex_policy(ExeSpaceUtils<DevExe>::get_default_team_policy(nv, nthreads_per_team)),
        face_policy(ExeSpaceUtils<DevExe>::get_default_team_policy(nf, nthreads_per_team)) {}
    
    virtual void timestep(const Real dt) const = 0;

  protected:
    Index nv, nf; /// number of vertices, number of faces (e.g., leaves in a tree-based mesh)
    /// RK4 stages for position updates
    vec_view_type vertinput, vertstage1, vertstage2, vertstage3, vertstage4;
    vec_view_type faceinput, facestage1, facestage2, facestage3, facestage4;
};




}
#endif
