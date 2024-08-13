#ifndef LPM_SWE_RK2_STAGGERED_HPP
#define LPM_SWE_RK2_STAGGERED_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "lpm_staggered_swe.hpp"

namespace Lpm {

template <typename SeedType, typename TopoType>
class SWERK2Staggered {
  public:
    using geo = typename SeedType::geo;
    using crd_view = typename SeedType::geo::crd_view_type;
    using vec_view = typename SeedType::geo::vec_view_type;

    Real dt;
    Int t_idx;

    StaggeredSWE<SeedType, TopoType>& swe;
    Real eps; /// velocity kernel smoothing parameter

    gmls::Params gmls_params;
    std::unique_ptr<GatherMeshData<SeedType>> gather;
    std::unique_ptr<ScatterMeshData<SeedType>> scatter;

    // constructor
    SWERK2Staggered(const Real timestep,
      StaggeredSWE<SeedType,TopoType>& swe,
      const gmls::Params& gmls_params);

    std::string info_string(const int tab_level=0) const;

    void advance_timestep_impl();

  private:
    crd_view x1;
    crd_view x2;
    crd_view xwork;

    scalar_view_type rel_vort1;
    scalar_view_type rel_vort2;
    scalar_view_type rel_vortwork;

    scalar_view_type divergence1;
    scalar_view_type divergence2;
    scalar_view_type divergencework;

    scalar_view_type area1;
    scalar_view_type area2;
    scalar_view_type areawork;

    std::unique_ptr<Kokkos::TeamPolicy<>> vertex_policy;

    std::map<std::string, ScalarField<FaceField>> face_field_map_gather;
    std::map<std::string, ScalarField<FaceField>> face_field_map_scatter;

    std::vector<Compadre::TargetOperation> gmls_ops;
};


} // namespace Lpm

#endif
