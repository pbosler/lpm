#ifndef LPM_POLYMESH2D_PARAMETERS_HPP
#define LPM_POLYMESH2D_PARAMETERS_HPP

#include "LpmConfig.h"
#include "mesh/lpm_mesh_seed.hpp"

namespace Lpm {

/** @brief Parameters that define a mesh and its memory requirements
    for initialization.

  @param depth initial depth of mesh quadtree, uniform resolution
  @param r radius of initial mesh in physical space
  @param amr values > 0 allocate additional memory for adaptive refinement
*/
template <typename SeedType>
struct PolyMeshParameters {
  Index nmaxverts;  /// max number of vertices to allocate in memory
  Index nmaxedges;  /// max number of edges to allocate in memory
  Index nmaxfaces;  /// max number of faces to allocated in memory
  Int init_depth;   /// initial depth of mesh quadtree
  Real radius;      /// radius of initial mesh in physical space
  Int amr_limit;    /// maximum number of times a panel may be refined beyond its initial depth
  Int amr_buffer;    /// if > 0, the allocated memory includes space for adaptive
                    /// refinement
  bool enable_boundary_zone; /// if true, particles outside boundary_radius will be flagged to define a boundary zone (planar only)
  Real boundary_radius; /// particles outside this radius define the boundary zone (planar only)
  bool periodic_boundary; /// if true, defines periodic boundary conditions in both planar directions (planar only)
  MeshSeed<SeedType> seed;  /// instance of the MeshSeed that initializes the
                            /// particles and panels

  PolyMeshParameters() = default;

  PolyMeshParameters(const PolyMeshParameters& other) = default;

  /** Use this constructor when memory allocations and mesh seed have been determined elsewhere.
  */
  PolyMeshParameters(const Index nmv, const Index nme, const Index nmf)
      : nmaxverts(nmv),
        nmaxedges(nme),
        nmaxfaces(nmf),
        init_depth(0),
        radius(1),
        amr_limit(0),
        amr_buffer(0),
        enable_boundary_zone(false),
        boundary_radius(1),
        periodic_boundary(false) {}

  /** @brief Primary constructor.

    @param [in] depth Initial uniform depth of mesh quadtree
    @param [in] r radius Radius in R3 or R2 of mesh's maximum extent
    @param [in] amr Memory allocations will yield enough space for each face to be divided (depth + amr) times;
      hence, any amr computation requires amr > 0.
  */
  PolyMeshParameters(const Int depth, const Real r = 1, const Int amr_buff = 0, const Int amr_lim = 0,
    const bool bndry_zone = false, const Real bndry_radius = 1, const bool periodic = false);
};

} // namespace Lpm
#endif
