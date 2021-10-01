#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_gpu_octree.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::octree;

TEST_CASE("gpu_octree", "[tree]") {

  Comm comm;

  Logger<> logger("gpu_octree", Log::level::info, comm);

//   typedef IcosTriSphereSeed seed_type;
  typedef CubedSphereSeed seed_type;

  MeshSeed<seed_type> seed;

  const Int mesh_tree_depth = 4;

  /**
    Set memory allocations
  */
  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;
  seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, mesh_tree_depth);

  /**
    Build the particle/panel mesh
  */
  auto sphere = std::shared_ptr<PolyMesh2d<seed_type>>(new
    PolyMesh2d<seed_type>(nmaxverts, nmaxedges, nmaxfaces));
  sphere->tree_init(mesh_tree_depth, seed);
  sphere->update_device();

  /** Output mesh to a vtk file */
  VtkPolymeshInterface<seed_type> vtk(sphere);
  vtk.write("gpu_octree_test_sphere.vtp");

  logger.info("building octree.");

  const int octree_depth = 4;

  SECTION("no connectivity") {
    GpuOctree octree(sphere->vertices.phys_crds->crds, octree_depth);

    logger.info(octree.info_string());

    octree.write_vtk("gpu_octree_test.vtu");
  }
//   SECTION("connectivity") {
//   }
}
