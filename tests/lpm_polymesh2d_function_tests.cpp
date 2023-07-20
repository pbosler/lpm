#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_field.hpp"
#include "lpm_lat_lon_pts.hpp"
#include "lpm_planar_grid.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_matlab_io.hpp"
#include "lpm_tracer_gallery.hpp"
// #include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include <sstream>

using namespace Lpm;

using Catch::Approx;

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
      const int zero_depth = 0;
      const Real mesh_radius = 1;
      const int amr_depth = 1;
      PolyMeshParameters<QuadRectSeed> qr0_params(zero_depth, mesh_radius, amr_depth);
      auto qr0 = PolyMesh2d<QuadRectSeed>(qr0_params);
      REQUIRE(qr0.n_faces_host() == 4);
      logger.debug("pre-divide: {}", qr0.info_string());

      typedef FaceDivider<PlaneGeometry, QuadFace> qr_divider;
      qr_divider::divide(0, qr0.vertices, qr0.edges, qr0.faces);

      logger.debug(qr0.info_string("divide 0", 0, true));
      REQUIRE(qr0.faces.kid_host(0,0) == 4);

      qr_divider::divide(qr0.faces.kid_host(0,0), qr0.vertices, qr0.edges, qr0.faces);
      logger.debug(qr0.info_string("divide kid(0,0)", 0, true));

      logger.debug("setting up local view copies to use with lambda");
      auto face_xy = Kokkos::subview(qr0.faces.phys_crds.view,
         std::make_pair(0, qr0.faces.nh()), Kokkos::ALL);
      auto face_xy_host = qr0.faces_phys_crds();
      const auto faces_edges = qr0.faces.edges;
      const auto edges_lefts = qr0.edges.lefts;
      const auto edges_rights = qr0.edges.rights;
      auto faces_kids = qr0.faces.kids;
      auto vcrds = qr0.vertices.phys_crds.view;
//       auto vert_xy = Kokkos::subview(qr0.vertices.phys_crds.view,
//         std::make_pair(0, qr0.vertices.nh()), Kokkos::ALL);

#ifdef LPM_USE_VTK
      logger.debug("starting vtk output.");
      VtkPolymeshInterface<QuadRectSeed> vtk(qr0);
      vtk.write("qr0_divide2.vtp");
      logger.debug("vtk output complete.");
#endif

      REQUIRE(qr0.n_faces_host() == 12);
      REQUIRE(qr0.edges.kid_host(0,0) == 12);
      REQUIRE(qr0.edges.kid_host(0,1) == 13);
      REQUIRE(qr0.edges.kid_host(12,0) == 24);
      REQUIRE(qr0.edges.kid_host(12,1) == 25);
      REQUIRE(qr0.edges.left_host(24) == 8);
      REQUIRE(qr0.edges.right_host(24) == LPM_NULL_IDX);
      REQUIRE(qr0.edges.left_host(25) == 9);
      REQUIRE(face_xy_host(8,0) == Approx(-7.0/8));
      REQUIRE(face_xy_host(8,1) == Approx( 7.0/8));
      qr0.update_device();

      Kokkos::View<Index[2*LPM_MAX_AMR_LIMIT]> leaf_edges0("leaf_edges0");
      auto leaf_edges0_host = Kokkos::create_mirror_view(leaf_edges0);
      n_view_type n_leaf_edges0("n_leaf_edges0");
      auto n_leaf_edges0_host = Kokkos::create_mirror_view(n_leaf_edges0);
      const auto edges_kids = qr0.edges.kids;

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
          qr0.get_leaf_edges_from_parent(leaf_edges0, n_leaf_edges0(), 0);
        }
        else if (i==1) {
          qr0.ccw_edges_around_face(leaf_edges7, n_leaf_edges7(), 7);
        }
        else if (i==2) {
          qr0.ccw_adjacent_faces(adj_faces5, n_adj5(), 5);
        }
        else if (i==3) {
          fidx0() =
            qr0.locate_pt_walk_search(qp, 2);
        }
        else if (i==4) {
          ridx() =
            qr0.nearest_root_face(qp);
        }
        else if (i==5) {
          fidx1() =
            qr0.locate_pt_tree_search(qp, 0);
        }
        else if (i==6) {
          fidx2() =
            qr0.locate_face_containing_pt(qp);
        }
        else if (i==7) {
          qr0.ref_elem_coords(qp_ref, qp);
        }
      });
      logger.debug("returned from kernel 1");

      Kokkos::View<Index*> face_idxs("face_idxs", qr0.n_faces_host());
      Kokkos::parallel_for(qr0.n_faces_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto fcrd = Kokkos::subview(face_xy, i, Kokkos::ALL);
          face_idxs(i) = qr0.locate_face_containing_pt(fcrd);
        });
      logger.debug("returned from locate face kernel");
      auto face_idxs_host = Kokkos::create_mirror_view(face_idxs);
      Kokkos::deep_copy(face_idxs_host, face_idxs);
      logger.debug("face_idxs = {}", sprarr("face_idxs", face_idxs_host.data(),
        qr0.n_faces_host()));
      const std::vector<Index> face_idx_correct(
        {10, 1, 2, 3, 8, 5, 6, 7, 8, 9, 10, 11});
      for (int i=0; i<qr0.n_faces_host(); ++i) {
        REQUIRE(face_idxs_host(i) == face_idx_correct[i]);
      }

      Kokkos::View<Index*> vert_idxs("vert_idxs", qr0.n_vertices_host());
      Kokkos::parallel_for(qr0.n_vertices_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto vcrd = Kokkos::subview(vcrds, i, Kokkos::ALL);
          vert_idxs(i) = qr0.locate_face_containing_pt(vcrd);
        });
      auto vert_idxs_host = Kokkos::create_mirror_view(vert_idxs);
      Kokkos::deep_copy(vert_idxs_host, vert_idxs);
      logger.debug("vert_idxs = {}", sprarr("vert_idxs", vert_idxs_host.data(),
        qr0.n_vertices_host()));
      const std::vector<Index> vert_idx_correct(
        {8, 5, 1, 1, 2, 2, 3, 7, 6, 9, 5, 6, 11, 10, 8, 9, 10, 8, 8});
      for (int i=0; i<qr0.n_vertices_host(); ++i) {
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

      logger.debug(qr0.edges.info_string("all_edges", 0, true));
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

template <typename SeedType, typename TracerType, typename OutputGrid>
struct InterpolationTest {
  int start_depth;
  int end_depth;
  Real radius;
  OutputGrid grid;
  std::vector<Real> dxs;
  std::vector<Real> grid_interp_l1;
  std::vector<Real> grid_interp_l2;
  std::vector<Real> grid_interp_linf;
  std::vector<Real> grid_interp_l1_rate;
  std::vector<Real> grid_interp_l2_rate;
  std::vector<Real> grid_interp_linf_rate;
  std::vector<Real> face_interp_l1;
  std::vector<Real> face_interp_l2;
  std::vector<Real> face_interp_linf;
  std::vector<Real> face_interp_l1_rate;
  std::vector<Real> face_interp_l2_rate;
  std::vector<Real> face_interp_linf_rate;

  InterpolationTest(const int sd, const int ed, const int n_unif=60) :
    start_depth(sd),
    end_depth(ed),
    radius((std::is_same<typename SeedType::geo, PlaneGeometry>::value ? 6 : 1)),
    grid(n_unif, radius) {}

  void run() {
    Comm comm;
    std::ostringstream ss;

    Logger<> logger("interpolation test", Log::level::info, comm);
    logger.debug("test run called.");

    TracerType tracer;

    for (int i=0; i<(end_depth - start_depth) + 1; ++i) {
      const int amr_limit = 0;
      const int depth = start_depth + i;

      std::ostringstream ss;
      ss << "interp_conv_" << SeedType::id_string() << depth;
      const auto test_name = ss.str();
      ss.str("");

      logger.info("starting test {}", test_name);
      Timer timer(test_name);
      timer.start();

      PolyMeshParameters<SeedType> params(depth, radius, amr_limit);
      const auto pm = PolyMesh2d<SeedType>(params);
      dxs.push_back(pm.appx_mesh_size());

      ScalarField<VertexField> tracer_verts("tracer", pm.vertices.nh());
      ScalarField<FaceField> tracer_faces("tracer", pm.faces.nh());
      ScalarField<VertexField> tracer_verts_interp("tracer_interp", pm.vertices.nh());
      ScalarField<FaceField> tracer_faces_interp("tracer_interp", pm.faces.nh());
      const auto vcrds = pm.vertices.phys_crds.view;
      const auto face_xy = pm.faces.phys_crds.view;
      const auto vg = tracer_verts.view;
      const auto fg = tracer_faces.view;
      scalar_view_type grid_tracer("grid_tracer", grid.size());
      scalar_view_type grid_tracer_interp("grid_tracer_interp", grid.size());
      scalar_view_type grid_error("grid_error", grid.size());
      scalar_view_type vert_error("error", pm.n_vertices_host());
      scalar_view_type face_error("error", pm.n_faces_host());
      Kokkos::View<Index*> grid_face_idx("grid_face_idx", grid.size());
      auto h_face_idx = Kokkos::create_mirror_view(grid_face_idx);

      Kokkos::parallel_for(pm.vertices.nh(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(vcrds, i, Kokkos::ALL);
          vg(i) = tracer(xy);
        });
      Kokkos::parallel_for(pm.faces.nh(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(face_xy, i, Kokkos::ALL);
          fg(i) = tracer(xy);
        });
      Kokkos::parallel_for(grid.size(),
        KOKKOS_LAMBDA (const Index i) {
          const auto xy = Kokkos::subview(grid.pts, i, Kokkos::ALL);
          grid_tracer(i) = tracer(xy);
          grid_face_idx(i) = pm.locate_face_containing_pt(xy);
        });
      logger.debug("finished setting initial data");

      pm.scalar_interpolate(grid_tracer_interp, grid.pts,
        tracer_verts);
      pm.scalar_interpolate(tracer_verts_interp.view, vcrds, tracer_verts);
      pm.scalar_interpolate(tracer_faces_interp.view, face_xy, tracer_verts);

      ErrNorms grid_err_norms(grid_error, grid_tracer_interp, grid_tracer,
        grid.wts);
      grid_interp_l1.push_back(grid_err_norms.l1);
      grid_interp_l2.push_back(grid_err_norms.l2);
      grid_interp_linf.push_back(grid_err_norms.linf);
      ErrNorms face_err_norms(face_error, tracer_faces_interp.view, tracer_faces.view,
        pm.faces.area_host());
      face_interp_l1.push_back(face_err_norms.l1);
      face_interp_l2.push_back(face_err_norms.l2);
      face_interp_linf.push_back(face_err_norms.linf);
      Kokkos::parallel_for(pm.n_vertices_host(),
        ComputeErrorFtor<scalar_view_type, scalar_view_type, scalar_view_type, 1>(vert_error, tracer_verts_interp.view, tracer_verts.view));;

      auto h_grid_tracer = Kokkos::create_mirror_view(grid_tracer);
      auto h_grid_tracer_interp = Kokkos::create_mirror_view(grid_tracer_interp);
      auto h_grid_error = Kokkos::create_mirror_view(grid_error);
      Kokkos::deep_copy(h_face_idx, grid_face_idx);
      Kokkos::deep_copy(h_grid_tracer, grid_tracer);
      Kokkos::deep_copy(h_grid_tracer_interp, grid_tracer_interp);
      Kokkos::deep_copy(h_grid_error, grid_error);
      std::ofstream mfile(test_name + ".m");
      write_array_matlab(mfile, "gridxy", grid.h_pts);
      write_vector_matlab(mfile, "gridwts", grid.h_wts);
      write_vector_matlab(mfile, "tracer", h_grid_tracer);
      write_vector_matlab(mfile, "tracer_interp", h_grid_tracer_interp);
      write_vector_matlab(mfile, "tracer_error", h_grid_error);
      write_vector_matlab(mfile, "face_idx", h_face_idx);
      mfile << "nx = " << grid.nx() << ";\n";
      mfile << "ny = " << grid.ny() << ";\n";
      mfile.close();

      logger.info("grid interpolation error: {}", grid_err_norms.info_string());
      logger.info("polymesh faces interpolation error: {}", face_err_norms.info_string());

#ifdef LPM_USE_VTK
      VtkPolymeshInterface<SeedType> vtk(pm);
      vtk.add_scalar_point_data(tracer_verts.view);
      vtk.add_scalar_point_data(tracer_verts_interp.view);
      vtk.add_scalar_point_data(vert_error);
      vtk.add_scalar_cell_data(tracer_faces.view);
      vtk.add_scalar_cell_data(tracer_faces_interp.view);
      vtk.add_scalar_cell_data(face_error);
      vtk.write(test_name + vtp_suffix());
#endif
    }

    grid_interp_l1_rate = convergence_rates(dxs, grid_interp_l1);
    grid_interp_l2_rate = convergence_rates(dxs, grid_interp_l2);
    grid_interp_linf_rate = convergence_rates(dxs, grid_interp_linf);
    face_interp_l1_rate = convergence_rates(dxs, face_interp_l1);
    face_interp_l2_rate = convergence_rates(dxs, face_interp_l2);
    face_interp_linf_rate = convergence_rates(dxs, face_interp_linf);

    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "grid_interp_l1", grid_interp_l1, grid_interp_l1_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "grid_interp_l2", grid_interp_l2, grid_interp_l2_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "grid_interp_linf", grid_interp_linf, grid_interp_linf_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "face_interp_l1", face_interp_l1, face_interp_l1_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "face_interp_l2", face_interp_l2, face_interp_l2_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs, "face_interp_linf", face_interp_linf, face_interp_linf_rate));

    REQUIRE( (face_interp_l2_rate.back() > 2.0 or face_interp_l2_rate.back() == Approx(2.0).epsilon(0.01) ) );
  }
};

TEST_CASE("polymesh2d functions: planar meshes", "") {

  PlanePolyMeshFnUnitTest utest;
  utest.run();


}

TEST_CASE("interpolation_test", "") {
  const int start_depth = 3;
  const int end_depth = 6;
  SECTION("planar tri") {
    typedef TriHexSeed seed_type;
    typedef PlanarGaussian tracer_type;
    typedef PlanarGrid grid_type;

    InterpolationTest<seed_type, tracer_type, grid_type> interp_test(start_depth, end_depth);
    interp_test.run();
  }
  SECTION("spherical tri") {
    typedef IcosTriSphereSeed seed_type;
    typedef SphereXYZTrigTracer tracer_type;
    typedef LatLonPts grid_type;

    InterpolationTest<seed_type, tracer_type, grid_type> interp_test(start_depth, end_depth, 45);
    interp_test.run();
  }
  SECTION("planar quad") {
    typedef QuadRectSeed seed_type;
    typedef PlanarGaussian tracer_type;
    typedef PlanarGrid grid_type;

    InterpolationTest<seed_type, tracer_type, grid_type> interp_test(start_depth, end_depth);
    interp_test.run();
  }
  SECTION("spherical quad") {
    typedef CubedSphereSeed seed_type;
    typedef SphereXYZTrigTracer tracer_type;
    typedef LatLonPts grid_type;

    InterpolationTest<seed_type, tracer_type, grid_type> interp_test(start_depth, end_depth, 45);
    interp_test.run();
  }
}

TEST_CASE("mesh to mesh", "") {
  Comm comm;
  std::ostringstream ss;

  Logger<> logger("mesh-mesh interpolation test", Log::level::info, comm);
  SECTION("sphere") {
    const int depth = 5;
    PolyMeshParameters<IcosTriSphereSeed> ic_params(depth);
    PolyMeshParameters<CubedSphereSeed> cs_params(depth);
    auto ic = PolyMesh2d<IcosTriSphereSeed>(ic_params);
    auto cs = PolyMesh2d<CubedSphereSeed>(cs_params);

    ScalarField<VertexField> ic_tracer_verts("tracer", ic.vertices.nh());
    ScalarField<FaceField> ic_tracer_faces("tracer", ic.faces.nh());
    ScalarField<VertexField> ic_tracer_verts_interp("tracer_interp", ic.vertices.nh());
    ScalarField<FaceField> ic_tracer_faces_interp("tracer_interp", ic.faces.nh());
    scalar_view_type ic_vert_error("error", ic.n_vertices_host());
    scalar_view_type ic_face_error("error", ic.n_faces_host());
    const auto ic_vcrds = ic.vertices.phys_crds.view;
    const auto ic_fcrds = ic.faces.phys_crds.view;
    const auto ic_vt = ic_tracer_verts.view;
    const auto ic_ft = ic_tracer_faces.view;

    ScalarField<VertexField> cs_tracer_verts("tracer", cs.vertices.nh());
    ScalarField<FaceField> cs_tracer_faces("tracer", cs.faces.nh());
    ScalarField<VertexField> cs_tracer_verts_interp("tracer_interp", cs.vertices.nh());
    ScalarField<FaceField> cs_tracer_faces_interp("tracer_interp", cs.faces.nh());
    scalar_view_type cs_vert_error("error", cs.n_vertices_host());
    scalar_view_type cs_face_error("error", cs.n_faces_host());
    const auto cs_vcrds = cs.vertices.phys_crds.view;
    const auto cs_fcrds = cs.faces.phys_crds.view;
    const auto cs_vt = cs_tracer_verts.view;
    const auto cs_ft = cs_tracer_faces.view;

    SphereXYZTrigTracer tracer;

    Kokkos::parallel_for(ic.vertices.nh(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(ic_vcrds, i, Kokkos::ALL);
        ic_vt(i) = tracer(xy);
      });
    Kokkos::parallel_for(ic.faces.nh(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(ic_fcrds, i, Kokkos::ALL);
        ic_ft(i) = tracer(xy);
      });

    Kokkos::parallel_for(cs.vertices.nh(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(cs_vcrds, i, Kokkos::ALL);
        cs_vt(i) = tracer(xy);
      });
    Kokkos::parallel_for(cs.faces.nh(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(cs_fcrds, i, Kokkos::ALL);
        cs_ft(i) = tracer(xy);
      });

    ic.scalar_interpolate(cs_tracer_verts_interp.view, cs_vcrds, ic_tracer_verts);
    ic.scalar_interpolate(cs_tracer_faces_interp.view, cs_fcrds, ic_tracer_verts);
    cs.scalar_interpolate(ic_tracer_verts_interp.view, ic_vcrds, cs_tracer_verts);
    cs.scalar_interpolate(ic_tracer_faces_interp.view, ic_fcrds, cs_tracer_verts);

    Kokkos::parallel_for(ic.n_vertices_host(),
        ComputeErrorFtor<scalar_view_type, scalar_view_type, scalar_view_type, 1>(ic_vert_error, ic_tracer_verts_interp.view, ic_tracer_verts.view));
    Kokkos::parallel_for(cs.n_vertices_host(),
      ComputeErrorFtor<scalar_view_type, scalar_view_type, scalar_view_type, 1>(cs_vert_error, cs_tracer_verts_interp.view, cs_tracer_verts.view));

    ErrNorms ic_err_norms(ic_face_error, ic_tracer_faces_interp.view, ic_ft,
        ic.faces.area_host());
    ErrNorms cs_err_norms(cs_face_error, cs_tracer_faces_interp.view, cs_ft,
        cs.faces.area_host());

    logger.info("icos tri err norms: {}", ic_err_norms.info_string());
    logger.info("cubed sphere err norms: {}", cs_err_norms.info_string());

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<IcosTriSphereSeed> ic_vtk(ic);
    ic_vtk.add_scalar_point_data(ic_tracer_verts.view);
    ic_vtk.add_scalar_point_data(ic_tracer_verts_interp.view);
    ic_vtk.add_scalar_point_data(ic_vert_error);
    ic_vtk.add_scalar_cell_data(ic_tracer_faces.view);
    ic_vtk.add_scalar_cell_data(ic_tracer_faces_interp.view);
    ic_vtk.add_scalar_cell_data(ic_face_error);
    ic_vtk.write("icos_tri_interp_test" + vtp_suffix());

    VtkPolymeshInterface<CubedSphereSeed> cs_vtk(cs);
    cs_vtk.add_scalar_point_data(cs_tracer_verts.view);
    cs_vtk.add_scalar_point_data(cs_tracer_verts_interp.view);
    cs_vtk.add_scalar_point_data(cs_vert_error);
    cs_vtk.add_scalar_cell_data(cs_tracer_faces.view);
    cs_vtk.add_scalar_cell_data(cs_tracer_faces_interp.view);
    cs_vtk.add_scalar_cell_data(cs_face_error);
    cs_vtk.write("cubed_sph_interp_test" + vtp_suffix());
#endif
  }
}

// TEST_CASE("polymesh2d functions: spherical meshes", "") {
//   SECTION("tri panels") {
//   }
//   SECTION("quad panels") {
//   }
// }
