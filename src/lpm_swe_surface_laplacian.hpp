#ifndef LPM_SWE_SURFACE_LAPLACIAN_HPP
#define LPM_SWE_SURFACE_LAPLACIAN_HPP

#include "LpmConfig.h"
#include "lpm_pse.hpp"
#include "lpm_compadre.hpp"

namespace Lpm {

template <typename SeedType>
class SWEPSELaplacian {
  public:
    using crd_view = typename SeedType::geo::crd_view_type;
    using pse_type = typename pse::BivariateOrder8<typename SeedType::geo>;

    scalar_view_type surf_lap_passive;
    scalar_view_type surf_lap_active;
    crd_view x_passive;
    crd_view x_active;
    scalar_view_type surface_passive;
    scalar_view_type surface_active;
    scalar_view_type area_active;
    Real eps;
    Index n_passive;
    Index n_active;

    /** update views with new data (e.g., for a new stage of runge-kutta)
    */
    void update(const crd_view tx, const crd_view sx, const scalar_view_type s, const scalar_view_type a);

    /** on output, both surf_lap_active and surf_lap_passive are updated
    */
    void compute();
  private:

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

    void update(const crd_view tx, const crd_view sx, const scalar_view_type s,
      const scalar_view_type a);

    SWEGMLSLaplacian(scalar_view_type laplacian,
      const crd_view tx,
      const crd_view sx,
      const scalar_view_type s,
      const gmls::Params& params);

    void compute()

  private:
    scalar_view_type gathered_surface;
    scalar_view_type gathered_laplacian;
    std::unique_ptr<GatheredMeshData<SeedType>> gathered_mesh;
    std::unique_ptr<ScatterMeshData<SeedType>> scattered_mesh;
    const gmls::Params& params;
    gmls::Neighborhoods neighbors;
    Compadre::GMLS scalar_gmls;
};

} // namespace Lpm

#endif
