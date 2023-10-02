#ifndef LPM_SWE_SURFACE_LAPLACIAN_HPP
#define LPM_SWE_SURFACE_LAPLACIAN_HPP

#include "LpmConfig.h"
#include "lpm_pse.hpp"
#include "lpm_compadre.hpp"
#include "lpm_swe.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"

namespace Lpm {

template <typename SeedType>
class SWEPSELaplacian {
  public:
    using crd_view = typename SeedType::geo::crd_view_type;
    using geo = typename SeedType::geo;
    using pse_type = typename pse::BivariateOrder8<typename SeedType::geo>;

    scalar_view_type surf_lap_passive;
    scalar_view_type surf_lap_active;
    crd_view x_passive;
    crd_view x_active;
    scalar_view_type surface_passive;
    scalar_view_type surface_active;
    scalar_view_type area_active;
    Index n_passive;
    Index n_active;
    Real eps;
//
//     /** update views with new data (e.g., for a new stage of runge-kutta)
//     */
//     void update(const crd_view tx, const crd_view sx, const scalar_view_type s, const scalar_view_type a);
//
//     /** on output, both surf_lap_active and surf_lap_passive are updated
//     */
    void compute();
//   private:

};

template <typename SeedType>
class SWEGMLSLaplacian {
  public:
    using crd_view = typename SeedType::geo::crd_view_type;
    using geo = typename SeedType::geo;

    scalar_view_type surf_lap_passive;
    scalar_view_type surf_lap_active;
    crd_view x_passive;
    crd_view x_active;
    scalar_view_type surface_passive;
    scalar_view_type surface_active;
    scalar_view_type area_active;
    Index n_passive;
    Index n_active;

    void update_src_data(const crd_view xp, const crd_view xa, const scalar_view_type sp, const scalar_view_type sa, const scalar_view_type ar);


    SWEGMLSLaplacian(SWE<SeedType>& swe,
      const gmls::Params& params);

    void compute();

  private:
    scalar_view_type gathered_surface;
    scalar_view_type gathered_laplacian;
    std::unique_ptr<GatherMeshData<SeedType>> gathered_mesh;
    std::unique_ptr<ScatterMeshData<SeedType>> scatter_mesh;
    const gmls::Params& params;
    gmls::Neighborhoods neighbors;
    std::unique_ptr<Compadre::GMLS> scalar_gmls;
};

} // namespace Lpm

#endif
