#include "LpmConfig.h"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"
#include "lpm_coords_impl.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_box2d.hpp"
#include "tree/lpm_quadtree_lookup_tables.hpp"
#include "tree/lpm_node_array_2d.hpp"
#include "tree/lpm_node_array_2d_impl.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"
#include "util/lpm_python3_util.hpp"
#include "catch.hpp"
#include <memory>
#include <fstream>

using namespace Lpm;
using namespace Lpm::quadtree;

TEST_CASE("node_array_2d", "[tree]") {
  Comm comm;

  auto logger = std::shared_ptr<Logger<>>(new Logger<>("node_array_2d_test", Log::level::debug, comm));

  const int npts = 400;
  const Real max_radius = 4;
  Coords<PlaneGeometry> coords(npts);
  coords.init_random(max_radius);

  logger->info(coords.info_string("coords_random"));

  const int max_quadtree_depth = 3;
  NodeArray2D leaves(coords.crds, max_quadtree_depth, logger);
  logger->info(leaves.info_string());

  auto h_keys = Kokkos::create_mirror_view(leaves.node_keys);
  auto h_idx0 = Kokkos::create_mirror_view(leaves.node_idx_start);
  auto h_idxn = Kokkos::create_mirror_view(leaves.node_idx_count);
  auto h_prnt = Kokkos::create_mirror_view(leaves.node_parents);
  Kokkos::deep_copy(h_keys, leaves.node_keys);
  Kokkos::deep_copy(h_idx0, leaves.node_idx_start);
  Kokkos::deep_copy(h_idxn, leaves.node_idx_count);
  Kokkos::deep_copy(h_prnt, leaves.node_parents);

  Kokkos::View<Box2d*,Host> boxes("boxes",leaves.nnodes);
  Kokkos::View<Real*[2],Host> box_vert_crds("box_vert_crds", 4*leaves.nnodes);
  for (auto i=0; i<leaves.nnodes; ++i) {
    logger->debug("node idx {} key {} idx0 {} idxn {} parent {}",
      i, h_keys(i), h_idx0(i), h_idxn(i), h_prnt(i));

    REQUIRE(h_prnt(i) == NULL_IDX);

    boxes(i) = box_from_key(h_keys(i), max_quadtree_depth, max_quadtree_depth,
      leaves.bbox);

    for (int v=0; v<4; ++v) {
      const auto vert = boxes(i).vertex_crds(v);
      box_vert_crds(4*i+v,0) = vert[0];
      box_vert_crds(4*i+v,1) = vert[1];
    }
  }

  auto h_pts = coords.get_host_crd_view();
  std::ofstream ofile("node_array_2d_test.py");
  REQUIRE(ofile.good());
  write_numpy_import(ofile);
  write_2d_numpy_array(ofile, "points", h_pts);
  write_1d_numpy_array(ofile, "node_keys", h_keys);
  write_1d_numpy_array(ofile, "node_idx0", h_idx0);
  write_1d_numpy_array(ofile, "node_idxn", h_idxn);
  write_2d_numpy_array(ofile, "box_vert_crds", box_vert_crds);
  ofile.close();

}
