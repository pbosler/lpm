#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_functions.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_string_util.hpp"
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "lpm_constants.hpp"
#include "lpm_velocity_gallery.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace Lpm;

struct PlanePolyMeshFnUnitTest {
  int depth;
  Real radius;
  int amr_limit;

  PlanePolyMeshFnUnitTest(const int d=3, const Real r=1, const int a=2) :
    depth(d),
    radius(r),
    amr_limit(a) {}

  void run() {
    Comm comm;
    std::ostringstream ss;

    Logger<> logger("polymesh2d function test", Log::level::debug, comm);
    logger.debug("test run called.");

    {
      PolyMeshParameters<QuadRectSeed> qr0_params(0, 1, 1);
      const auto qr0 = std::shared_ptr<PolyMesh2d<QuadRectSeed>>(new
        PolyMesh2d<QuadRectSeed>(qr0_params));
      logger.info(qr0->info_string());
      REQUIRE(qr0->n_faces_host() == 4);

      typedef FaceDivider<PlaneGeometry, QuadFace> qr_divider;
      qr_divider::divide(0, qr0->vertices, qr0->edges, qr0->faces);
      REQUIRE(qr0->faces.kids(0,0) == 4);
      qr_divider::divide(qr0->faces.kids(0,0), qr0->vertices, qr0->edges, qr0->faces);


      logger.info(qr0->info_string());
      auto face_xy_host = Kokkos::subview(qr0->faces.phys_crds->get_host_crd_view(),
         std::make_pair(0, qr0->faces.nh()), Kokkos::ALL);
      auto vert_xy_host = Kokkos::subview(qr0->vertices.phys_crds->get_host_crd_view(),
        std::make_pair(0, qr0->vertices.nh()), Kokkos::ALL);

#ifdef LPM_USE_VTK
      VtkPolymeshInterface<QuadRectSeed> vtk(qr0);
      vtk.write("qr0_divide2.vtp");
#endif

      Index leaf_edges0[2*LPM_MAX_AMR_LIMIT];
      Int n_leaf_edges0;
      get_leaf_edges_from_parent<Index*,QuadRectSeed>(leaf_edges0, n_leaf_edges0, 0, qr0->edges.kids);

      logger.debug(qr0->edges.info_string("all_edges", 0, true));
      logger.info("parent edge 0 has {} leaves: [{}, {}, {}]", n_leaf_edges0,
        leaf_edges0[0], leaf_edges0[1], leaf_edges0[2]);

      REQUIRE(qr0->n_faces_host() == 12);
      REQUIRE(qr0->edges.kids(0,0) == 12);
      REQUIRE(qr0->edges.kids(0,1) == 13);
      REQUIRE(qr0->edges.kids(12,0) == 24);
      REQUIRE(qr0->edges.kids(12,1) == 25);
      REQUIRE(qr0->edges.lefts(24) == 8);
      REQUIRE(qr0->edges.rights(24) == LPM_NULL_IDX);
      REQUIRE(qr0->edges.lefts(25) == 9);
      REQUIRE(face_xy_host(8,0) == Approx(-7.0/8));
      REQUIRE(face_xy_host(8,1) == Approx( 7.0/8));
      REQUIRE(n_leaf_edges0 == 3);
      REQUIRE(leaf_edges0[0] == 24);
      REQUIRE(leaf_edges0[1] == 25);
      REQUIRE(leaf_edges0[2] == 13);

      Index leaf_edges7[8*LPM_MAX_AMR_LIMIT];
      Int n_leaf_edges7;
      ccw_edges_around_face<Index*,QuadRectSeed>(leaf_edges7, n_leaf_edges7,
        7, qr0->faces.edges, qr0->edges.lefts, qr0->edges.kids);

      logger.info("face 7 has {} {}", n_leaf_edges7,
        sprarr("leaf_edges7", leaf_edges7, n_leaf_edges7));

      REQUIRE(leaf_edges7[0] == 29);
      REQUIRE(leaf_edges7[1] == 28);
      REQUIRE(leaf_edges7[2] == 21);
      REQUIRE(leaf_edges7[3] == 17);
      REQUIRE(leaf_edges7[4] == 18);

      Index adj_faces5[8*LPM_MAX_AMR_LIMIT];
      Int n_adj5;
      ccw_adjacent_faces<Index*, QuadRectSeed>(adj_faces5, n_adj5,
        5, qr0->faces.edges, qr0->edges.lefts, qr0->edges.rights,
        qr0->edges.kids);

      logger.info("face 5 has {} {}", n_adj5,
        sprarr("adjacent faces", adj_faces5, n_adj5));

      REQUIRE(n_adj5 == 5);
      REQUIRE(adj_faces5[0] == LPM_NULL_IDX);
      REQUIRE(adj_faces5[1] == 1);
      REQUIRE(adj_faces5[2] == 6);
      REQUIRE(adj_faces5[3] == 10);
      REQUIRE(adj_faces5[4] == 9);

      const Real qp[2] = {-7.0/8, 7.0/8};
      const auto fidx = locate_point_walk_search<const Real*, QuadRectSeed>(qp, 2,
        qr0->edges.lefts, qr0->edges.rights, qr0->edges.kids,
        qr0->faces.edges, face_xy_host);

      logger.info("{} located in face {}",
        sprarr("query pt", qp, 2), fidx);

      REQUIRE(fidx == 8);


      const auto ridx = nearest_root_face<const Real*, QuadRectSeed>(qp, face_xy_host);
      REQUIRE(ridx == 0);
      const auto fidx2 =
        locate_point_tree_search<const Real*, QuadRectSeed>(qp, ridx,
          face_xy_host, qr0->faces.kids);

      REQUIRE(fidx2 == 8);

      const auto fidx3 =
        locate_face_containing_pt<const Real*, QuadRectSeed>(qp, qr0->edges.lefts,
          qr0->edges.rights, qr0->edges.kids, face_xy_host, qr0->faces.kids,
          qr0->faces.edges);

      REQUIRE(fidx3 == 8);


    }
  }


};

TEST_CASE("polymesh2d functions: planar meshes", "") {

  PlanePolyMeshFnUnitTest utest;
  utest.run();


}

// TEST_CASE("polymesh2d functions: spherical meshes", "") {
//   SECTION("tri panels") {
//   }
//   SECTION("quad panels") {
//   }
// }
