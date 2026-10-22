#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

using namespace Lpm;

TEMPLATE_TEST_CASE ("sphere_test_case2_init", "[swe]", CubedSphereSeed, IcosTriSphereSeed) {
  using Geometry = SphereGeometry;
  using Coriolis = CoriolisSphere;
  using SeedType = TestType;

  Comm comm;
  const std::string test_name = "sphere_tc2";
  Logger<> logger(test_name, Log::level::debug, comm);

  int tree_depth = 4;
  constexpr Real sphere_radius = 1.0;
  constexpr Real h0 = 5.0;
  constexpr Real g = 1.0;
  constexpr Real u0 = constants::PI/6;
  constexpr Real Omega = 2*constants::PI;
  Coriolis coriolis(Omega);

  SWETestCase2 test_case2(h0, Omega, u0, g);
  const Real Ro = rossby_number(u0, 2*Omega, sphere_radius);
  const Real Fr = froude_number(u0, g, h0);
  logger.info(test_case2.info_string());
  logger.info("SWETestCase2 initialized with Rossby number = {}, Froude number = {}", Ro, Fr);

  PolyMeshParameters<SeedType> mesh_params(tree_depth);

  auto sphere = std::make_unique<SWE<SeedType>>(mesh_params, coriolis);

  logger.debug("initializing test case on sphere particle/panel mesh.");
  sphere->init_swe_problem(test_case2);

  logger.debug("computing velocity.");
  sphere->compute_velocity_direct_sum();

  logger.debug("initializing gmls for surface laplacian");
  constexpr Int gmls_order = 8;
  constexpr Real gmls_radius_multiplier = 2.0;
  sphere->initialize_gmls(gmls_order, gmls_radius_multiplier);
  logger.debug("gmls initialized.");
  sphere->compute_surface_laplacian_gmls();

  logger.info(sphere->info_string());

  typename Geometry::vec_view_type velocity_error_passive("velocity_error", sphere->mesh.n_vertices_host());
  typename Geometry::vec_view_type velocity_error_active("velocity_error", sphere->mesh.n_faces_host());
  scalar_view_type laplacian_error_passive("laplacian_error", sphere->mesh.n_vertices_host());
  scalar_view_type laplacian_error_active("laplacian_error", sphere->mesh.n_faces_host());

  auto crds_view = sphere->mesh.vertices.phys_crds.view;
  auto vel_view = sphere->velocity_passive.view;
  auto lap_view = sphere->surf_lap_passive.view;
  Kokkos::parallel_for(sphere->mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto m_x = Kokkos::subview(crds_view, i, Kokkos::ALL);
      const auto m_u = Kokkos::subview(vel_view, i, Kokkos::ALL);
      Real u_exact[3];
      test_case2.velocity(u_exact, m_x);
      for (int j=0; j<SphereGeometry::ndim; ++j) {
        velocity_error_passive(i,j) = m_u(j) - u_exact[j];
      }
      laplacian_error_passive(i) = lap_view(i) - test_case2.surface_laplacian(m_x);
    });
  crds_view = sphere->mesh.faces.phys_crds.view;
  vel_view = sphere->velocity_active.view;
  lap_view = sphere->surf_lap_active.view;
  Kokkos::parallel_for(sphere->mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto m_x = Kokkos::subview(crds_view, i, Kokkos::ALL);
      const auto m_u = Kokkos::subview(vel_view, i, Kokkos::ALL);
      Real u_exact[3];
      test_case2.velocity(u_exact, m_x);
      for (int j=0; j<SphereGeometry::ndim; ++j) {
        velocity_error_active(i,j) = m_u(j) - u_exact[j];
      }
      laplacian_error_active(i) = lap_view(i) - test_case2.surface_laplacian(m_x);
    });




  auto vtk = vtk_mesh_interface(*sphere);
  vtk.add_scalar_point_data(laplacian_error_passive);
  vtk.add_scalar_cell_data(laplacian_error_active);
  vtk.add_vector_point_data(velocity_error_passive);
  vtk.add_vector_cell_data(velocity_error_active);
  vtk.write(test_name + SeedType::id_string() + vtp_suffix());
}
