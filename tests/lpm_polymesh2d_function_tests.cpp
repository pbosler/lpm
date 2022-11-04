#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_string_util.hpp"
// #include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <memory>
#include <sstream>

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

      REQUIRE(qr0->faces.kid_host(0,0) == 4);

      qr_divider::divide(qr0->faces.kid_host(0,0), qr0->vertices, qr0->edges, qr0->faces);
      logger.info(qr0->info_string());

      logger.debug("setting up local view copies to use with lambda");
      auto face_xy = Kokkos::subview(qr0->faces.phys_crds->crds,
         std::make_pair(0, qr0->faces.nh()), Kokkos::ALL);
      auto face_xy_host = qr0->faces_phys_crds();
      const auto faces_edges = qr0->faces.edges;
      const auto edges_lefts = qr0->edges.lefts;
      const auto edges_rights = qr0->edges.rights;
      auto faces_kids = qr0->faces.kids;
//       auto vert_xy = Kokkos::subview(qr0->vertices.phys_crds->crds,
//         std::make_pair(0, qr0->vertices.nh()), Kokkos::ALL);

#ifdef LPM_USE_VTK
      logger.debug("starting vtk output.");
      VtkPolymeshInterface<QuadRectSeed> vtk(qr0);
      vtk.write("qr0_divide2.vtp");
      logger.debug("vtk output complete.");
#endif

      REQUIRE(qr0->n_faces_host() == 12);
      REQUIRE(qr0->edges.kid_host(0,0) == 12);
      REQUIRE(qr0->edges.kid_host(0,1) == 13);
      REQUIRE(qr0->edges.kid_host(12,0) == 24);
      REQUIRE(qr0->edges.kid_host(12,1) == 25);
      REQUIRE(qr0->edges.left_host(24) == 8);
      REQUIRE(qr0->edges.right_host(24) == LPM_NULL_IDX);
      REQUIRE(qr0->edges.left_host(25) == 9);
      REQUIRE(face_xy_host(8,0) == Approx(-7.0/8));
      REQUIRE(face_xy_host(8,1) == Approx( 7.0/8));
      qr0->update_device();

      Kokkos::View<Index[2*LPM_MAX_AMR_LIMIT]> leaf_edges0("leaf_edges0");
      auto leaf_edges0_host = Kokkos::create_mirror_view(leaf_edges0);
      n_view_type n_leaf_edges0("n_leaf_edges0");
      auto n_leaf_edges0_host = Kokkos::create_mirror_view(n_leaf_edges0);
      const auto edges_kids = qr0->edges.kids;

      Kokkos::View<Index[8*LPM_MAX_AMR_LIMIT]> leaf_edges7("leaf_edges7");
      auto leaf_edges7_host = Kokkos::create_mirror_view(leaf_edges7);
      n_view_type n_leaf_edges7("n_leaf_edges7");
      auto n_leaf_edges7_host = Kokkos::create_mirror_view(n_leaf_edges7);


      Kokkos::View<Index[8*LPM_MAX_AMR_LIMIT]> adj_faces5("adj_faces5");
      auto adj_faces5_host = Kokkos::create_mirror_view(adj_faces5);
      n_view_type n_adj5("n_adj5");
      auto n_adj5_host = Kokkos::create_mirror_view(n_adj5);

      Kokkos::View<Real[2]> qp("qp");
      auto qp_host = Kokkos::create_mirror_view(qp);
      qp_host[0] = -7.0/8;
      qp_host[1] =  7.0/8;
      Kokkos::deep_copy(qp, qp_host);
      n_view_type fidx0("fidx0");
      n_view_type fidx1("fidx1");
      n_view_type fidx2("fidx2");
      n_view_type ridx("ridx");

      auto fidx0_host = Kokkos::create_mirror_view(fidx0);
      auto fidx1_host = Kokkos::create_mirror_view(fidx1);
      auto fidx2_host = Kokkos::create_mirror_view(fidx2);
      auto ridx_host = Kokkos::create_mirror_view(ridx);

      logger.info("launching test kernels on device...");

      Kokkos::parallel_for(7, KOKKOS_LAMBDA (const Index i) {

        if (i==0) {
          qr0->get_leaf_edges_from_parent(leaf_edges0, n_leaf_edges0(), 0);
        }
        else if (i==1) {
          qr0->ccw_edges_around_face(leaf_edges7, n_leaf_edges7(), 7);
        }
        else if (i==2) {
          qr0->ccw_adjacent_faces(adj_faces5, n_adj5(), 5);
        }
        else if (i==3) {
          fidx0() =
            qr0->locate_pt_walk_search(qp, 2);
        }
        else if (i==4) {
          ridx() =
            qr0->nearest_root_face(qp);
        }
        else if (i==5) {
          fidx1() =
            qr0->locate_pt_tree_search(qp, 0);
        }
        else if (i==6) {
          fidx2() =
            qr0->locate_face_containing_pt(qp);
        }
      });

      Kokkos::deep_copy(leaf_edges0_host, leaf_edges0);
      Kokkos::deep_copy(n_leaf_edges0_host, n_leaf_edges0);
      Kokkos::deep_copy(leaf_edges7_host, leaf_edges7);
      Kokkos::deep_copy(n_leaf_edges7_host, n_leaf_edges7);
      Kokkos::deep_copy(adj_faces5_host, adj_faces5);
      Kokkos::deep_copy(n_adj5_host, n_adj5);
      Kokkos::deep_copy(fidx0_host, fidx0);
      Kokkos::deep_copy(fidx1_host, fidx1);
      Kokkos::deep_copy(fidx2_host, fidx2);

      logger.debug(qr0->edges.info_string("all_edges", 0, true));
      logger.info("parent edge 0 has {} leaves: [{}, {}, {}]", n_leaf_edges0_host(),
        leaf_edges0_host[0], leaf_edges0_host[1], leaf_edges0_host[2]);

      REQUIRE(n_leaf_edges0_host() == 3);
      REQUIRE(leaf_edges0_host[0] == 24);
      REQUIRE(leaf_edges0_host[1] == 25);
      REQUIRE(leaf_edges0_host[2] == 13);

      logger.info("face 7 has {} {}", n_leaf_edges7_host(),
        sprarr("leaf_edges7", leaf_edges7_host.data(), n_leaf_edges7_host()));

      REQUIRE(leaf_edges7_host[0] == 29);
      REQUIRE(leaf_edges7_host[1] == 28);
      REQUIRE(leaf_edges7_host[2] == 21);
      REQUIRE(leaf_edges7_host[3] == 17);
      REQUIRE(leaf_edges7_host[4] == 18);

      logger.info("face 5 has {} {}", n_adj5_host(),
        sprarr("adjacent faces", adj_faces5_host.data(), n_adj5_host()));

      REQUIRE(n_adj5_host() == 5);
      REQUIRE(adj_faces5_host[0] == LPM_NULL_IDX);
      REQUIRE(adj_faces5_host[1] == 1);
      REQUIRE(adj_faces5_host[2] == 6);
      REQUIRE(adj_faces5_host[3] == 10);
      REQUIRE(adj_faces5_host[4] == 9);

      logger.info("{} located in face {}",
        sprarr("query pt", qp_host.data(), 2), fidx0_host());

      REQUIRE(fidx0_host() == 8);
      REQUIRE(ridx_host() == 0);
      REQUIRE(fidx2_host() == 8);

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
