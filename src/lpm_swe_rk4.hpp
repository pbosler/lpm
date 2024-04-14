#ifndef LPM_SWE_RK4
#define LPM_SWE_RK4

#include "LpmConfig.h"

namespace Lpm {

template <typename SeedType, typename TopoType>
class SWERK4 {
  public:
    using geo = typename SeedType::geo;
    using crd_view = typename SeedType::geo::crd_view_type;
    using vec_view = typename SeedType::geo::vec_view_type;

    Real dt;
    Int t_idx;

    crd_view passive_x;
    scalar_view_type passive_rel_vort;
    scalar_view_type passive_divergence;
    scalar_view_type passive_depth;
    scalar_view_type passive_surface;
    vec_view passive_vel;
    scalar_view_type passive_ddot;
    scalar_view_type passive_laps;
    scalar_view_type passive_bottom;

    crd_view active_x;
    scalar_view_type active_rel_vort;
    scalar_view_type active_divergence;
    scalar_view_type active_depth;
    scalar_view_type active_surface;
    vec_view active_vel;
    scalar_view_type active_area;
    scalar_view_type active_mass;
    mask_view_type active_mask;
    scalar_view_type active_ddot;
    scalar_view_type active_laps;
    scalar_view_type active_bottom;


    SWE<SeedType>& swe;
    TopoType topo;
    Real eps; /// velocity kernel smoothing parameter
    Real pse_eps; /// pse kernel width parameter

    void advance_timestep_impl();

    // constructor
    SWERK4(const Real timestep, SWE<SeedType>& swe_mesh, TopoType& topo);

    std::string info_string(const int tab_level=0) const;

  private:

    crd_view passive_x1;
    crd_view passive_x2;
    crd_view passive_x3;
    crd_view passive_x4;
    crd_view passive_xwork;

    scalar_view_type passive_rel_vort1;
    scalar_view_type passive_rel_vort2;
    scalar_view_type passive_rel_vort3;
    scalar_view_type passive_rel_vort4;
    scalar_view_type passive_rel_vortwork;

    scalar_view_type passive_div1;
    scalar_view_type passive_div2;
    scalar_view_type passive_div3;
    scalar_view_type passive_div4;
    scalar_view_type passive_divwork;

    scalar_view_type passive_depth1;
    scalar_view_type passive_depth2;
    scalar_view_type passive_depth3;
    scalar_view_type passive_depth4;
    scalar_view_type passive_depthwork;

    crd_view active_x1;
    crd_view active_x2;
    crd_view active_x3;
    crd_view active_x4;
    crd_view active_xwork;

    scalar_view_type active_rel_vort1;
    scalar_view_type active_rel_vort2;
    scalar_view_type active_rel_vort3;
    scalar_view_type active_rel_vort4;
    scalar_view_type active_rel_vortwork;

    scalar_view_type active_div1;
    scalar_view_type active_div2;
    scalar_view_type active_div3;
    scalar_view_type active_div4;
    scalar_view_type active_divwork;

    scalar_view_type active_area1;
    scalar_view_type active_area2;
    scalar_view_type active_area3;
    scalar_view_type active_area4;
    scalar_view_type active_areawork;

    std::unique_ptr<Kokkos::TeamPolicy<>> passive_policy;
    std::unique_ptr<Kokkos::TeamPolicy<>> active_policy;

    void set_fixed_views();

    void advance_timestep(crd_view& vx,
                          scalar_view_type& vzeta,
                          scalar_view_type& vsigma,
                          scalar_view_type& vh,
                          vec_view& vvel,
                          crd_view& fx,
                          scalar_view_type& fzeta,
                          scalar_view_type& fsigma,
                          scalar_view_type& fh,
                          vec_view& fvel,
                          scalar_view_type& farea,
                          const scalar_view_type& fmass,
                          const mask_view_type& fmask);
};


} // namespace Lpm

#endif
