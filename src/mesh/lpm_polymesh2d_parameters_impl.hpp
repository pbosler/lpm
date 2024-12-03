#ifndef LPM_POLYMESH_PARAMETERS_IMPL_HPP
#define LPM_POLYMESH_PARAMETERS_IMPL_HPP

#include "lpm_polymesh2d_parameters.hpp"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

template <typename SeedType>
PolyMeshParameters<SeedType>::PolyMeshParameters(const Int depth, const Real r, const Int amr_buff, const Int amr_lim,
    const bool bndry_zone, const Real bndry_radius, const bool periodic) :
    init_depth(depth),
    radius(r),
    amr_buffer(amr_buff),
    amr_limit(amr_lim),
    enable_boundary_zone(bndry_zone),
    boundary_radius(bndry_radius),
    periodic_boundary(periodic),
    seed(r) {

    LPM_ASSERT(depth >= 0);
    LPM_ASSERT(r > 0);
    LPM_ASSERT(amr_buff >= 0);
    LPM_ASSERT(amr_lim >= 0);

    if (bndry_zone) {
      LPM_REQUIRE_MSG( (std::is_same<typename SeedType::geo, PlaneGeometry>::value),
        "Boundary zones require planar geometry.");
      LPM_REQUIRE_MSG( r >= bndry_radius, "Boundary zone must not exceed mesh radius.");
    }
    if (periodic) {
      LPM_REQUIRE_MSG( (std::is_same<typename SeedType::geo, PlaneGeometry>::value),
        "Periodic boundaries require planar geometry.");
    }

    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, depth + amr_buff);


  }

}

#endif
