#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_lat_lon_pts.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "util/lpm_matlab_io.hpp"
#include "lpm_constants.hpp"
#include "lpm_velocity_gallery.hpp"
#include "fortran/lpm_ssrfpack_interface.hpp"
#include "fortran/lpm_ssrfpack_interface_impl.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace Lpm;

template <typename VelocityType, typename SeedType>
struct SSRFPackConvergenceTest {
  typedef typename std::conditional<
    std::is_same<SeedType, CubedSphereSeed>::value,
    IcosTriSphereSeed,
    CubedSphereSeed>::type OtherSeedType;
  int start_depth;
  int end_depth;
  std::vector<Real> dxs;
  std::vector<Real> ll_l1;
  std::vector<Real> ll_l2;
  std::vector<Real> ll_linf;
  std::vector<Real> self_l1;
  std::vector<Real> self_l2;
  std::vector<Real> self_linf;
  std::vector<Real> face_l1;
  std::vector<Real> face_l2;
  std::vector<Real> face_linf;

  SSRFPackConvergenceTest(const int sd, const int ed) :
    start_depth(sd),
    end_depth(ed) {}

  void run() {
    Comm comm;

    SphereXYZTrigTracer tracer;

    Logger<> logger("ssrfpack_conv", Log::level::debug, comm);
    logger.debug("test run called.");

    Timer run_timer("SSRFPackConvergenceTest");
    run_timer.start();

    /*

    set up output grid

    */
    const int nlat = 45;
    const int nlon = 90;
    LatLonPts ll(nlat, nlon);
    scalar_view_type ll_tracer("tracer", ll.size());
    scalar_view_type ll_tracer_interp("tracer_interp", ll.size());
    scalar_view_type ll_error("error", ll.size());

    Kokkos::parallel_for("init_tracer_exact", ll.size(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xyz = Kokkos::subview(ll.pts, i, Kokkos::ALL);
        ll_tracer(i) = tracer(xyz);
      });

    /*

    set up output polymesh

    */
    const int output_depth = 4;
    PolyMeshParameters<OtherSeedType> oparams(output_depth);
    auto opm = TransportMesh2d<OtherSeedType>(oparams);
    opm.template initialize_velocity<VelocityType>();
    opm.initialize_tracer(tracer);
    opm.allocate_scalar_tracer("tracer_interp");
    opm.allocate_scalar_tracer("tracer_error");

    for (int i=0; i<(end_depth - start_depth) + 1; ++i ) {
      const int depth = start_depth + i;
      std::ostringstream ss;
      ss << "ssrfpack_conv_" << SeedType::id_string() << depth;
      const auto test_name = ss.str();
      ss.str("");

      Timer itimer(test_name);
      itimer.start();

      logger.info("starting test: {}", test_name);
      // initialize mesh
      PolyMeshParameters<SeedType> params(depth);
      auto pm = TransportMesh2d<SeedType>(params);
      pm.template initialize_velocity<VelocityType>();
      pm.initialize_tracer(tracer);
      pm.allocate_scalar_tracer("tracer_interp");
      pm.allocate_scalar_tracer("tracer_error");

      dxs.push_back(pm.mesh.appx_mesh_size());

      /*
      collect only active faces and vertices (no divided faces)
      */
      GatherMeshData<SeedType> gathered(pm.mesh);
      gathered.unpack_coordinates();
      gathered.init_scalar_fields(pm.tracer_verts, pm.tracer_faces);
      gathered.gather_scalar_fields(pm.tracer_verts, pm.tracer_faces);
      gathered.update_host();

      std::map<std::string, std::string> in_out_map;
      in_out_map.emplace(tracer.name(), "tracer_interp");

      SSRFPackInterface<SeedType> ssrfpack(gathered, in_out_map);
      const auto ll_interp_view = Kokkos::create_mirror_view(ll_tracer_interp);
      std::map<std::string, typename scalar_view_type::HostMirror> ll_map;
      ll_map.emplace("tracer_interp", ll_interp_view);
      logger.info(ssrfpack.info_string());
      ssrfpack.interpolate(ll.h_pts, ll_map);

      ssrfpack.interpolate(pm.mesh, pm.tracer_verts, pm.tracer_faces);

      ssrfpack.interpolate(opm.mesh, opm.tracer_verts, opm.tracer_faces);

      compute_error(pm.tracer_verts.at("tracer_error").view,
                    pm.tracer_verts.at("tracer_interp").view,
                    pm.tracer_verts.at(tracer.name()).view);

      compute_error(opm.tracer_verts.at("tracer_error").view,
                    opm.tracer_verts.at("tracer_interp").view,
                    opm.tracer_verts.at(tracer.name()).view);

      ErrNorms ll_err_norms(ll_error, ll_tracer_interp, ll_tracer, ll.wts);

      ErrNorms self_face_interp_err(pm.tracer_faces.at("tracer_error").view,
        pm.tracer_faces.at("tracer_interp").view,
        pm.tracer_faces.at(tracer.name()).view,
        pm.mesh.faces.area);

      ErrNorms other_face_interp_err(opm.tracer_faces.at("tracer_error").view,
        opm.tracer_faces.at("tracer_interp").view,
        opm.tracer_faces.at(tracer.name()).view,
        opm.mesh.faces.area);

      logger.info("ll interp error: {}", ll_err_norms.info_string());
      logger.info("self face interp error: {}", self_face_interp_err.info_string());
      logger.info("other face interp error: {}", other_face_interp_err.info_string());

      ll_l1.push_back(ll_err_norms.l1);
      ll_l2.push_back(ll_err_norms.l2);
      ll_linf.push_back(ll_err_norms.linf);

      self_l1.push_back(self_face_interp_err.l1);
      self_l2.push_back(self_face_interp_err.l2);
      self_linf.push_back(self_face_interp_err.linf);

      face_l1.push_back(other_face_interp_err.l1);
      face_l2.push_back(other_face_interp_err.l2);
      face_linf.push_back(other_face_interp_err.linf);

#ifdef LPM_USE_VTK
      VtkPolymeshInterface<SeedType> vtk(pm.mesh);
      vtk.add_scalar_point_data(pm.tracer_verts.at(tracer.name()).view);
      vtk.add_scalar_point_data(pm.tracer_verts.at("tracer_interp").view);
      vtk.add_scalar_point_data(pm.tracer_verts.at("tracer_error").view);
      vtk.add_scalar_cell_data(pm.tracer_faces.at(tracer.name()).view);
      vtk.add_scalar_cell_data(pm.tracer_faces.at("tracer_interp").view);
      vtk.add_scalar_cell_data(pm.tracer_faces.at("tracer_error").view);
      vtk.write(test_name + vtp_suffix());

      VtkPolymeshInterface<OtherSeedType> ovtk(opm.mesh);
      ovtk.add_scalar_point_data(opm.tracer_verts.at(tracer.name()).view);
      ovtk.add_scalar_point_data(opm.tracer_verts.at("tracer_interp").view);
      ovtk.add_scalar_point_data(opm.tracer_verts.at("tracer_error").view);
      ovtk.add_scalar_cell_data(opm.tracer_faces.at(tracer.name()).view);
      ovtk.add_scalar_cell_data(opm.tracer_faces.at("tracer_interp").view);
      ovtk.add_scalar_cell_data(opm.tracer_faces.at("tracer_error").view);
      ovtk.write("other_tgt_" + test_name + vtp_suffix());
#endif

      itimer.stop();
      logger.info("{} complete : {}", test_name, itimer.info_string());
    }

    const auto ll_l1_rate = convergence_rates(dxs, ll_l1);
    const auto ll_l2_rate = convergence_rates(dxs, ll_l2);
    const auto ll_linf_rate = convergence_rates(dxs, ll_linf);

    const auto self_l1_rate = convergence_rates(dxs, self_l1);
    const auto self_l2_rate = convergence_rates(dxs, self_l2);
    const auto self_linf_rate = convergence_rates(dxs, self_linf);

    const auto face_l1_rate = convergence_rates(dxs, face_l1);
    const auto face_l2_rate = convergence_rates(dxs, face_l2);
    const auto face_linf_rate = convergence_rates(dxs, face_linf);

    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "latlon_l1", ll_l1, ll_l1_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "latlon_l2", ll_l2, ll_l2_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "latlon_linf", ll_linf, ll_linf_rate));

    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "self_l1", self_l1, self_l1_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "self_l2", self_l2, self_l2_rate));
    logger.info(convergence_table(SeedType::id_string() + "_dx", dxs,
      "self_linf", self_linf, self_linf_rate));

    logger.info(convergence_table(SeedType::id_string() +"_dx", dxs,
      "face_l1", face_l1, face_l1_rate));
    logger.info(convergence_table(SeedType::id_string() +"_dx", dxs,
      "face_l2", face_l2, face_l2_rate));
    logger.info(convergence_table(SeedType::id_string() +"_dx", dxs,
      "face_linf", face_linf, face_linf_rate));

    run_timer.stop();
    logger.info("tests complete : {}", run_timer.info_string());
  }
};


TEST_CASE("planar_bivar", "") {
  const int start_depth = 2;
  int end_depth = 5;

  typedef SphericalRigidRotation velocity_type;
  SECTION("icostri_sphere") {
    typedef IcosTriSphereSeed seed_type;

    SSRFPackConvergenceTest<velocity_type,seed_type>
      ssrfpack_test(start_depth, end_depth);
    ssrfpack_test.run();

  }
  SECTION("cubed_sphere") {
    typedef CubedSphereSeed seed_type;
    SSRFPackConvergenceTest<velocity_type,seed_type>
      ssrfpack_test(start_depth, end_depth);
    ssrfpack_test.run();
  }
}
