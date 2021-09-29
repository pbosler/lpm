#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_node_array_d.hpp"
#include "tree/lpm_node_array_internal.hpp"
#include "tree/lpm_node_array_internal_impl.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::tree;

TEST_CASE("node_array_d_unit_tests", "[tree]") {

  Comm comm;

  Logger<> logger("node_array_d", Log::level::info, comm);

  typedef IcosTriSphereSeed seed_type;
  //typedef CubedSphereSeed seed_type;

  MeshSeed<seed_type> seed;

  const Int mesh_tree_depth = 3;

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
  vtk.write("node_array_d_test_sphere.vtp");

  logger.info("building octree leaves.");

  const int octree_depth = 4;
  NodeArrayD leaves(sphere->vertices.phys_crds->crds, octree_depth);
  REQUIRE(leaves.max_depth == octree_depth);
  REQUIRE(leaves.level == octree_depth);
  logger.info(leaves.info_string());

  leaves.write_vtk("node_array_4_test.vtu");

  logger.info("building octree next level.");

  NodeArrayInternal level3(leaves);
  REQUIRE(level3.level == octree_depth-1);
  REQUIRE(level3.max_depth == octree_depth);
  logger.info(level3.info_string());
  level3.write_vtk("node_array_3_test.vtu");

  NodeArrayInternal level2(level3);
  level2.write_vtk("node_array_2_test.vtu");
  logger.info(level2.info_string());

  NodeArrayInternal level1(level2);
  logger.info(level1.info_string());
  level1.write_vtk("node_array_1_test.vtu");

  NodeArrayInternal level0(level1);
  level0.write_vtk("node_array_0_test.vtu");
  logger.info(level0.info_string());

  auto root_count = Kokkos::create_mirror_view(level0.node_idx_count);
  Kokkos::deep_copy(root_count, level0.node_idx_count);
  REQUIRE(root_count() == sphere->vertices.nh());
  logger.info("root node contains all points.");

  auto l1parents = Kokkos::create_mirror_view(level1.node_parents);
  auto root_kids = Kokkos::create_mirror_view(level0.node_kids);
  REQUIRE(l1parents.extent(0) == 8);
  REQUIRE(root_kids.extent(0) == 1);
  REQUIRE(root_kids.extent(1) == 8);
  Kokkos::deep_copy(l1parents, level1.node_parents);
  Kokkos::deep_copy(root_kids, level0.node_kids);
  for (auto k=0; k<8; ++k) {
    REQUIRE(l1parents(k) == 0);
    REQUIRE(root_kids(0,k) == k);
  }
  logger.info("root kids/parents make sense.");

  if (octree_depth == 4 and mesh_tree_depth == 3) {
    REQUIRE(leaves.nnodes == 1952);
    REQUIRE(level3.nnodes == 448);

    logger.info("regression testing complete.");
  }

}
