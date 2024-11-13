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
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_pse.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "lpm_constants.hpp"
#include "lpm_velocity_gallery.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace Lpm;
using namespace Lpm::pse;

template <typename VelocityType, typename SeedType> struct PSEConvergenceTest {
  int start_depth;
  int end_depth;
  Real radius;
  std::vector<Real> dxs;
  std::vector<Real> interp_l1;
  std::vector<Real> interp_l2;
  std::vector<Real> interp_linf;

  std::vector<Real> lap_l1;
  std::vector<Real> lap_l2;
  std::vector<Real> lap_linf;

  using pse_type = pse::BivariateOrder8;

  PSEConvergenceTest(const int sd, const int ed, const Real r) :
    start_depth(sd),
    end_depth(ed),
    radius(r) {}

  void run() {
    Comm comm;

    typename std::conditional<
      std::is_same<typename SeedType::geo, PlaneGeometry>::value,
      PlanarGaussian, SphereXYZTrigTracer>::type tracer;

    Logger<> logger("pse_conv", Log::level::debug, comm);
    logger.debug("test run called.");

    for (int i=0; i< (end_depth-start_depth) + 1; ++i) {
      const int amr_limit = 0;
      const int depth = start_depth + i;

      std::ostringstream ss;
      ss << "pse_conv_" << SeedType::id_string() << depth;
      const auto test_name = ss.str();
      ss.str("");

      Timer timer(test_name);
      timer.start();

      logger.info("starting test: {} {}", test_name, depth);

      PolyMeshParameters<SeedType> params(depth, radius, amr_limit);
      const auto pm = std::unique_ptr<TransportMesh2d<SeedType>>(new
         TransportMesh2d<SeedType>(params));
      pm->template initialize_velocity<VelocityType>();
      pm->initialize_tracer(tracer);
      pm->allocate_scalar_tracer("tracer_laplacian");
      pm->allocate_scalar_tracer("tracer_pse");
      pm->allocate_scalar_tracer("tracer_pse_laplacian");
      pm->allocate_scalar_tracer("tracer_error");
      pm->allocate_scalar_tracer("tracer_laplacian_error");

      const auto rlap_verts = pm->tracer_verts.at("tracer_laplacian").view;
      const auto vlxy = pm->mesh.vertices.lag_crds.view;
      Kokkos::parallel_for(pm->mesh.vertices.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(vlxy, i, Kokkos::ALL);
        rlap_verts(i) = tracer.laplacian(xy);
      });
      const auto rlap_faces = pm->tracer_faces.at("tracer_laplacian").view;
      const auto flxy = pm->mesh.faces.lag_crds.view;
      Kokkos::parallel_for(pm->mesh.faces.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(flxy, i, Kokkos::ALL);
        rlap_faces(i) = tracer.laplacian(xy);
      });

      Kokkos::TeamPolicy<> vertex_policy(pm->mesh.vertices.nh(), Kokkos::AUTO());
      Kokkos::TeamPolicy<> face_policy(pm->mesh.faces.nh(), Kokkos::AUTO());

      dxs.push_back(pm->mesh.appx_mesh_size());
      const auto pse_epsilon = PSEKernel<PlaneGeometry>::get_epsilon(dxs[i]);

      const auto verts_tracer = pm->tracer_verts.at(tracer.name()).view;
      const auto faces_tracer = pm->tracer_faces.at(tracer.name()).view;
      const auto face_area = pm->mesh.faces.area;

      auto verts_pse_interp = pm->tracer_verts.at("tracer_pse").view;
      auto verts_pse_lap = pm->tracer_verts.at("tracer_pse_laplacian").view;
      auto faces_pse_interp = pm->tracer_faces.at("tracer_pse").view;
      auto faces_pse_lap = pm->tracer_faces.at("tracer_pse_laplacian").view;

      Kokkos::parallel_for(vertex_policy,
        pse::ScalarInterpolation<pse_type>(verts_pse_interp, vlxy, flxy,
          faces_tracer, face_area, pse_epsilon, pm->mesh.faces.nh()));
      Kokkos::parallel_for(vertex_policy,
        pse::ScalarLaplacian<pse_type>(verts_pse_lap, vlxy, flxy,
          verts_tracer, faces_tracer, face_area, pse_epsilon, pm->mesh.faces.nh()));
      Kokkos::parallel_for(face_policy,
        pse::ScalarInterpolation<pse_type>(faces_pse_interp, flxy, flxy, faces_tracer,
          face_area, pse_epsilon, pm->mesh.faces.nh()));
      Kokkos::parallel_for(face_policy,
        pse::ScalarLaplacian<pse_type>(faces_pse_lap, flxy, flxy, faces_tracer, faces_tracer,
          face_area, pse_epsilon, pm->mesh.faces.nh()));

      auto vert_interp_error = pm->tracer_verts.at("tracer_error").view;
      auto vert_lap_error = pm->tracer_verts.at("tracer_laplacian_error").view;
      auto face_interp_error = pm->tracer_faces.at("tracer_error").view;
      auto face_lap_error = pm->tracer_faces.at("tracer_laplacian_error").view;

      logger.info(pm->info_string());

      compute_error(vert_interp_error, verts_pse_interp, verts_tracer);
      compute_error(vert_lap_error, verts_pse_lap, rlap_verts);

      ErrNorms interp_err(face_interp_error, faces_pse_interp, faces_tracer, face_area);
      ErrNorms lap_err(face_lap_error, faces_pse_lap, rlap_faces, face_area);

#ifdef LPM_USE_VTK
      VtkPolymeshInterface<SeedType> vtk = vtk_interface(*pm);
      vtk.write(test_name + vtp_suffix());
#endif

      const auto erinterp = reduce_error(face_interp_error,
        faces_pse_interp,
        faces_tracer);
      const auto erlap = reduce_error(face_lap_error, faces_pse_lap,
        rlap_faces);

      interp_l1.push_back(interp_err.l1);
      interp_l2.push_back(interp_err.l2);
      interp_linf.push_back(interp_err.linf);

      lap_l1.push_back(lap_err.l1);
      lap_l2.push_back(lap_err.l2);
      lap_linf.push_back(lap_err.linf);

      logger.info("dx = {}, pse_epsilon = {}", dxs[i], pse_epsilon);
      logger.info(interp_err.info_string());
      logger.info(lap_err.info_string());

      timer.stop();
      logger.info(timer.info_string());
    }
    const auto interp_l1_rate = convergence_rates(dxs, interp_l1);
    const auto interp_l2_rate = convergence_rates(dxs, interp_l2);
    const auto interp_linf_rate = convergence_rates(dxs, interp_linf);

    const auto lap_l1_rate = convergence_rates(dxs, lap_l1);
    const auto lap_l2_rate = convergence_rates(dxs, lap_l2);
    const auto lap_linf_rate = convergence_rates(dxs, lap_linf);

    logger.info(convergence_table("dx", dxs, "interp_l1", interp_l1, interp_l1_rate));
    logger.info(convergence_table("dx", dxs, "interp_l2", interp_l2, interp_l2_rate));
    logger.info(convergence_table("dx", dxs, "interp_linf", interp_linf, interp_linf_rate));

    logger.info(convergence_table("dx", dxs, "lap_l1", lap_l1, lap_l1_rate));
    logger.info(convergence_table("dx", dxs, "lap_l2", lap_l2, lap_l2_rate));
    logger.info(convergence_table("dx", dxs, "lap_linf", lap_linf, lap_linf_rate));
  }

};

TEST_CASE("planar mesh", "") {
  const int start_depth = 2;
  int end_depth = 4;

  auto& ts = TestSession::get();
  if (ts.params.find("end-depth") != ts.params.end()) {
    end_depth = std::stoi(ts.params["end-depth"]);
  }
  for (const auto& p : ts.params) {
    std::cout << p.first << " = " << p.second << "\n";
  }

  SECTION("triangular panels") {
    typedef TriHexSeed seed_type;
    typedef PlanarConstantEastward velocity_type;
    const Real radius = 6;

    PSEConvergenceTest<velocity_type, seed_type> pse_test(start_depth, end_depth, radius);
    pse_test.run();
  }

  SECTION("quadrilateral panels") {
    typedef QuadRectSeed seed_type;
    typedef PlanarConstantEastward velocity_type;
    const Real radius = 6;

    PSEConvergenceTest<velocity_type, seed_type> pse_test(start_depth, end_depth, radius);
    pse_test.run();
  }
}
// TEST_CASE("sphere mesh", "") {
//   const int start_depth = 2;
//   int end_depth = 4;
//
//   auto& ts = TestSession::get();
//   if (ts.params.find("end-depth") != ts.params.end()) {
//     end_depth = std::stoi(ts.params["end-depth"]);
//   }
//
//   SECTION("quadrilateral panels") {
//     typedef CubedSphereSeed seed_type;
//     typedef SphericalRigidRotation velocity_type;
//     const Real radius = 1;
//
//     PSEConvergenceTest<velocity_type, seed_type> pse_test(start_depth, end_depth, radius);
//     pse_test.run();
//   }
//
//   SECTION("triangular panels") {
//     typedef IcosTriSphereSeed seed_type;
//     typedef SphericalRigidRotation velocity_type;
//     const Real radius = 1;
//
//     PSEConvergenceTest<velocity_type, seed_type> pse_test(start_depth, end_depth, radius);
//     pse_test.run();
//   }
// }
