#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_tracer_gallery.hpp"
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
      REQUIRE(qr0->n_faces_host() == 4);

      typedef FaceDivider<PlaneGeometry, QuadFace> qr_divider;
      qr_divider::divide(0, qr0->vertices, qr0->edges, qr0->faces);

      logger.debug(qr0->info_string("divide 0", 0, true));
      REQUIRE(qr0->faces.kid_host(0,0) == 4);

      qr_divider::divide(qr0->faces.kid_host(0,0), qr0->vertices, qr0->edges, qr0->faces);
      logger.debug(qr0->info_string("divide kid(0,0)", 0, true));

      logger.debug("setting up local view copies to use with lambda");
      auto face_xy = Kokkos::subview(qr0->faces.phys_crds->crds,
         std::make_pair(0, qr0->faces.nh()), Kokkos::ALL);
      auto face_xy_host = qr0->faces_phys_crds();
      const auto faces_edges = qr0->faces.edges;
      const auto edges_lefts = qr0->edges.lefts;
      const auto edges_rights = qr0->edges.rights;
      auto faces_kids = qr0->faces.kids;
      auto vcrds = qr0->vertices.phys_crds->crds;
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
      Kokkos::View<Real[2]> qp_ref("qp_ref");
      auto qp_host = Kokkos::create_mirror_view(qp);
      qp_host[0] = -0.875;
      qp_host[1] =  0.875;
      Kokkos::deep_copy(qp, qp_host);
      n_view_type fidx0("fidx0");
      n_view_type fidx1("fidx1");
      n_view_type fidx2("fidx2");
      n_view_type ridx("ridx");
      auto qp_ref_host = Kokkos::create_mirror_view(qp_ref);

      auto fidx0_host = Kokkos::create_mirror_view(fidx0);
      auto fidx1_host = Kokkos::create_mirror_view(fidx1);
      auto fidx2_host = Kokkos::create_mirror_view(fidx2);
      auto ridx_host = Kokkos::create_mirror_view(ridx);

      logger.debug("launching test kernels on device...");

      Kokkos::parallel_for(8, KOKKOS_LAMBDA (const Index i) {

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
        else if (i==7) {
          qr0->ref_elem_coords(qp_ref, qp);
        }
      });
      logger.debug("returned from kernel 1");

      Kokkos::View<Index*> face_idxs("face_idxs", qr0->n_faces_host());
      Kokkos::parallel_for(qr0->n_faces_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto fcrd = Kokkos::subview(face_xy, i, Kokkos::ALL);
          face_idxs(i) = qr0->locate_face_containing_pt(fcrd);
        });
      logger.debug("returned from locate face kernel");
      auto face_idxs_host = Kokkos::create_mirror_view(face_idxs);
      Kokkos::deep_copy(face_idxs_host, face_idxs);
      logger.debug("face_idxs = {}", sprarr("face_idxs", face_idxs_host.data(),
        qr0->n_faces_host()));
      const std::vector<Index> face_idx_correct(
        {10, 1, 2, 3, 8, 5, 6, 7, 8, 9, 10, 11});
      for (int i=0; i<qr0->n_faces_host(); ++i) {
        REQUIRE(face_idxs_host(i) == face_idx_correct[i]);
      }

      Kokkos::View<Index*> vert_idxs("vert_idxs", qr0->n_vertices_host());
      Kokkos::parallel_for(qr0->n_vertices_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto vcrd = Kokkos::subview(vcrds, i, Kokkos::ALL);
          vert_idxs(i) = qr0->locate_face_containing_pt(vcrd);
        });
      auto vert_idxs_host = Kokkos::create_mirror_view(vert_idxs);
      Kokkos::deep_copy(vert_idxs_host, vert_idxs);
      logger.debug("vert_idxs = {}", sprarr("vert_idxs", vert_idxs_host.data(),
        qr0->n_vertices_host()));
      const std::vector<Index> vert_idx_correct(
        {8, 5, 1, 1, 2, 2, 3, 7, 6, 9, 5, 6, 11, 10, 8, 9, 10, 8, 8});
      for (int i=0; i<qr0->n_vertices_host(); ++i) {
        REQUIRE(vert_idxs_host(i) == vert_idx_correct[i]);
      }

      logger.debug("returned from kernels.");

      Kokkos::deep_copy(leaf_edges0_host, leaf_edges0);
      Kokkos::deep_copy(n_leaf_edges0_host, n_leaf_edges0);
      Kokkos::deep_copy(leaf_edges7_host, leaf_edges7);
      Kokkos::deep_copy(n_leaf_edges7_host, n_leaf_edges7);
      Kokkos::deep_copy(adj_faces5_host, adj_faces5);
      Kokkos::deep_copy(n_adj5_host, n_adj5);
      Kokkos::deep_copy(fidx0_host, fidx0);
      Kokkos::deep_copy(fidx1_host, fidx1);
      Kokkos::deep_copy(fidx2_host, fidx2);
      Kokkos::deep_copy(qp_ref_host, qp_ref);

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

      logger.info("pt with global coords ({}, {}) has local coords ({}, {})",
        qp_host[0], qp_host[1], qp_ref_host[0], qp_ref_host[1]);
      REQUIRE( (qp_ref_host[0] == 0 and qp_ref_host[1] == 0) );
    }
  }


};

struct PlanarGrid {
  Kokkos::View<Real*[2]> pts;
  Kokkos::View<Real*> wts;
  typename Kokkos::View<Real*[2]>::HostMirror h_pts;
  typename Kokkos::View<Real*>::HostMirror h_wts;
  Real xmin;
  Real xmax;
  Real ymin;
  Real ymax;

  PlanarGrid(const int n, const Real xmi, const Real xma) :
    pts("pts", n*n),
    wts("wts", n*n),
    xmin(xmi),
    xmax(xma),
    ymin(xmi),
    ymax(xma) {

    h_pts = Kokkos::create_mirror_view(pts);
    h_wts = Kokkos::create_mirror_view(wts);

    const Real dx = (xmax - xmin)/(n-1);
    const Real dy = (ymin - ymax)/(n-1);
    for (int i=0; i<n*n; ++i) {
      const Int ii = i/n;
      const Int jj = i%n;
      h_pts(i, 0) = xmin + ii*dx;
      h_pts(i, 1) = ymin + jj*dy;
      h_wts(i) = dx*dy;
    }
    Kokkos::deep_copy(pts, h_pts);
    Kokkos::deep_copy(wts, h_wts);
  }

  inline int size() const {return pts.extent(0);}

  inline int n() const {return int(sqrt(pts.extent(0)));}
};

template <typename SeedType>
struct PlaneInterpolationTest {
  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "planar test");
  int start_depth;
  int end_depth;
  Real radius;
  PlanarGrid grid;
  std::vector<Real> dxs;
  std::vector<Real> interp_l1;
  std::vector<Real> interp_l2;
  std::vector<Real> interp_linf;

  PlaneInterpolationTest(const int sd, const int ed, const int n_unif=60) :
    start_depth(sd),
    end_depth(ed),
    radius(6),
    grid(n_unif, -radius, radius) {}

  void run() {
    Comm comm;
    std::ostringstream ss;

    Logger<> logger("planar interpolation test", Log::level::info, comm);
    logger.debug("test run called.");

    PlanarGaussian gaussian;

    for (int i=0; i<(end_depth - start_depth) + 1; ++i) {
      const int amr_limit = 0;
      const int depth = start_depth + i;

      std::ostringstream ss;
      ss << "planar_interp_conv_" << SeedType::id_string() << depth;
      const auto test_name = ss.str();
      ss.str("");

      logger.info("starting test {}", test_name);
      Timer timer(test_name);
      timer.start();

      PolyMeshParameters<SeedType> params(depth, radius, amr_limit);
      const auto pm = std::make_shared<PolyMesh2d<SeedType>>(params);
      dxs.push_back(pm->appx_mesh_size());

      ScalarField<VertexField> gaussian_verts("gaussian", pm->vertices.nh());
      ScalarField<FaceField> gaussian_faces("gaussian", pm->faces.nh());
      ScalarField<VertexField> gaussian_verts_interp("gaussian_interp", pm->vertices.nh());
      ScalarField<FaceField> gaussian_faces_interp("gaussian_interp", pm->faces.nh());
      const auto vcrds = pm->vertices.phys_crds->crds;
      const auto face_xy = pm->faces.phys_crds->crds;
      const auto vg = gaussian_verts.view;
      const auto fg = gaussian_faces.view;
      scalar_view_type grid_gaussian("grid_gaussian", grid.size());
      scalar_view_type grid_gaussian_interp("grid_gaussian_interp", grid.size());
      scalar_view_type grid_error("grid_error", grid.size());

      Kokkos::parallel_for(pm->vertices.nh(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(vcrds, i, Kokkos::ALL);
          vg(i) = gaussian(xy);
        });
      Kokkos::parallel_for(pm->faces.nh(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(face_xy, i, Kokkos::ALL);
          fg(i) = gaussian(xy);
        });
      Kokkos::parallel_for(grid.size(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(grid.pts, i, Kokkos::ALL);
          grid_gaussian(i) = gaussian(xy);
        });
      logger.debug("finished setting initial data");
//       pm->scalar_interpolate(grid_gaussian_interp, grid.pts,
//         gaussian_verts);
      pm->scalar_interpolate(gaussian_verts_interp.view, vcrds, gaussian_verts);
      pm->scalar_interpolate(gaussian_faces_interp.view, face_xy, gaussian_verts);

#ifdef LPM_USE_VTK
      VtkPolymeshInterface<SeedType> vtk(pm);
      vtk.add_scalar_point_data(gaussian_verts.view);
      vtk.add_scalar_point_data(gaussian_verts_interp.view);
      vtk.add_scalar_cell_data(gaussian_faces.view);
      vtk.add_scalar_cell_data(gaussian_faces_interp.view);
      vtk.write(test_name + vtp_suffix());
#endif
    }
  }
};

TEST_CASE("polymesh2d functions: planar meshes", "") {

  PlanePolyMeshFnUnitTest utest;
  utest.run();


}

TEST_CASE("interpolation_test", "") {
  const int start_depth = 3;
  const int end_depth = 3;
  SECTION("planar tri") {
    typedef TriHexSeed seed_type;

    PlaneInterpolationTest<seed_type> interp_test(start_depth, end_depth);
    interp_test.run();
  }
  SECTION("spherical tri") {
  }
  SECTION("planar quad") {
    typedef QuadRectSeed seed_type;
    PlaneInterpolationTest<seed_type> interp_test(start_depth, end_depth);
    interp_test.run();
  }
  SECTION("spherical quad") {
  }
}

// TEST_CASE("polymesh2d functions: spherical meshes", "") {
//   SECTION("tri panels") {
//   }
//   SECTION("quad panels") {
//   }
// }
