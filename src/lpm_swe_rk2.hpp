#ifndef LPM_SWE_RK2_HPP
#define LPM_SWE_RK2_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"


namespace Lpm {

/** 2nd order Runge Kutta time stepper for shallow water equations.

  uses GMLS for computing the surface laplacian.
*/
template <typename SeedType, typename TopoType>
class SWERK2 {
  public:
    using geo = typename SeedType::geo;
    using crd_view = typename SeedType::geo::crd_view_type;
    using vec_view = typename SeedType::geo::vec_view_type;

    Real dt;
    Int t_idx;

    SWE<SeedType>& swe;
    TopoType topo;
    Real eps; /// velocity kernel smoothing parameter

    gmls::Params gmls_params;
    std::unique_ptr<GatherMeshData<SeedType>> gather;
    std::unique_ptr<ScatterMeshData<SeedType>> scatter;

    // constructor
    SWERK2(const Real timestep, SWE<SeedType>& swe_mesh, TopoType& topo, const gmls::Params& gmls_params);

    std::string info_string(const int tab_level=0) const;

    void advance_timestep_impl();

    private:

    crd_view passive_x1;
    crd_view passive_x2;
    crd_view passive_xwork;

    scalar_view_type passive_rel_vort1;
    scalar_view_type passive_rel_vort2;
    scalar_view_type passive_rel_vortwork;

    scalar_view_type passive_div1;
    scalar_view_type passive_div2;
    scalar_view_type passive_divwork;

    scalar_view_type passive_depth1;
    scalar_view_type passive_depth2;
    scalar_view_type passive_depthwork;

    crd_view active_x1;
    crd_view active_x2;
    crd_view active_xwork;

    scalar_view_type active_rel_vort1;
    scalar_view_type active_rel_vort2;
    scalar_view_type active_rel_vortwork;

    scalar_view_type active_div1;
    scalar_view_type active_div2;
    scalar_view_type active_divwork;

    scalar_view_type active_area1;
    scalar_view_type active_area2;
    scalar_view_type active_areawork;

    std::unique_ptr<Kokkos::TeamPolicy<>> passive_policy;
    std::unique_ptr<Kokkos::TeamPolicy<>> active_policy;

    std::map<std::string, ScalarField<VertexField>> passive_field_map_gather;
    std::map<std::string, ScalarField<FaceField>> active_field_map_gather;
    std::map<std::string, ScalarField<VertexField>> passive_field_map_scatter;
    std::map<std::string, ScalarField<FaceField>> active_field_map_scatter;

    std::vector<Compadre::TargetOperation> gmls_ops;
};

} // namespace Lpm

#endif
