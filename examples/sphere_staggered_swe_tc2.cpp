#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_staggered_swe.hpp"
#include "lpm_staggered_swe_impl.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe_rk2_staggered.hpp"
#include "lpm_swe_rk2_staggered_impl.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

using namespace Lpm;

template <typename SeedType>
void tc2_setup_tracers(StaggeredSWE<SeedType, ZeroFunctor>& swe);

template <typename SeedType>
void tc2_exact_sol(StaggeredSWE<SeedType, ZeroFunctor>& swe, const CoriolisSphere& coriolis);

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("sphere_swe_tc2", Log::level::debug, comm);

  // compile-time settings
  // mesh seed (choose one, comment-out the other)
  using seed_type = CubedSphereSeed;
//   using seed_type = IcosTriSphereSeed;

  // sphere test case 2 problem setup
  using topography_type  = ZeroFunctor;
  using divergence_type = ZeroFunctor;
  using init_sfc_type = SphereTestCase2InitialSurface;
  using coriolis_type = CoriolisSphere;
  using vorticity_type = SphereTestCase2Vorticity;
  using geo = SphereGeometry;
  using solver_type = SWERK2Staggered<seed_type, topography_type>;

  Kokkos::initialize(argc, argv);
  { // Kokkos scope
    user::Input input("sphere_tc2");
    {
      // define user parameters
      user::Option tfinal_option("tfinal", "-tf", "--time_final", "time final", 0.5);
      input.add_option(tfinal_option);

      user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of steps", 5);
      input.add_option(nsteps_option);

      user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree depth", 4);
      input.add_option(tree_depth_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("staggered_swe_tc2"));
      input.add_option(output_file_root_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

      user::Option gmls_poynomial_order_option("gmls_polynomial_order", "-g", "--gmls-order", "gmls polynomial order", 4);
      input.add_option(gmls_poynomial_order_option);

      user::Option verbose_output_option("verbose", "-v", "--verbose", "verbose output to logger", false);
      input.add_option(verbose_output_option);
    }
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    const int nsteps = input.get_option("nsteps").get_int();
    const Real dt = input.get_option("tfinal").get_real() / nsteps;

    int frame_counter = 0;
    const int write_frequency = input.get_option("output_write_frequency").get_int();
    logger.info(input.info_string());

    Timer total_time("total_time");

    // initialize planar particle/panel mesh
    constexpr Real sphere_radius = 1.0;
    constexpr Real gravity = init_sfc_type::g;
    constexpr Real omega = 2*constants::PI;
    constexpr int amr_limit = 0;
    PolyMeshParameters<seed_type> mesh_params(
      input.get_option("tree_depth").get_int(),
      sphere_radius,
      amr_limit);
    coriolis_type coriolis(omega);
    topography_type topo;
    auto sphere = std::make_unique<StaggeredSWE<seed_type, topography_type>>(mesh_params, coriolis, topo);
    sphere->eps = input.get_option("kernel_smoothing_parameter").get_real();
    const Real cr = constants::PI/6 * dt / sphere->mesh.appx_mesh_size();
    logger.info("Courant number for this problem is appx. {}", cr);
    // set problem initial conditions
    init_sfc_type sfc;
    vorticity_type vorticity;
    divergence_type div;
    gmls::Params gmls_sfc_lap_params(input.get_option("gmls_polynomial_order").get_int());
    sphere->init_fields(sfc, vorticity, div, gmls_sfc_lap_params);

    tc2_setup_tracers(*sphere);
    tc2_exact_sol(*sphere, coriolis);

    logger.debug("found {} nans in velocity field", sphere->velocity.nan_count(sphere->mesh.n_vertices_host()));
    logger.debug("found {} nans in double_dot field", sphere->double_dot.nan_count(sphere->mesh.n_vertices_host()));
    logger.debug("found {} nans in double_dot_avg field", sphere->double_dot_avg.nan_count(sphere->mesh.n_faces_host()));
    logger.info(sphere->info_string(0,input.get_option("verbose").get_bool()));

#ifdef LPM_USE_VTK
    const std::string eps_str = "eps"+float_str(sphere->eps);
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int());
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + eps_str + "_";
    {
      sphere->update_host();
      auto vtk = vtk_mesh_interface(*sphere);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file: {}", sphere->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif
    /**
      Time stepping
    */
    auto solver = std::make_unique<solver_type>(dt, *sphere, gmls_sfc_lap_params);
    for (int t_idx = 0; t_idx < nsteps; ++t_idx) {
      sphere->advance_timestep(*solver);
      tc2_exact_sol(*sphere, coriolis);

#ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {
        sphere->update_host();
        auto vtk = vtk_mesh_interface(*sphere);
        auto ctr_str = zero_fill_str(++frame_counter);
        const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
        logger.info("writing output at t = {} to file: {}", sphere->t, vtk_fname);
        vtk.write(vtk_fname);
      }
#endif
    }

    total_time.stop();
    logger.info("total time: {}", total_time.info_string());
  } // kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}

template <typename SeedType>
void tc2_setup_tracers(StaggeredSWE<SeedType,ZeroFunctor>& swe) {

  swe.allocate_scalar_tracer("f_zeta");
  swe.allocate_scalar_tracer("div_rhs");
  swe.allocate_scalar_tracer("slap_exact");
  swe.allocate_scalar_tracer("f_zeta_exact");
  swe.allocate_scalar_tracer("s_exact");

  swe.allocate_scalar_diag("grad_f_cross_u_exact");
}

template <typename SeedType>
void tc2_exact_sol(StaggeredSWE<SeedType, ZeroFunctor>& swe, const CoriolisSphere& coriolis) {
  const Real Omega = 2*constants::PI;
  const Real u0 = constants::PI/6;
  constexpr Real g = 1;
  constexpr Real h0 = 10;

  auto fzeta_view = swe.tracers.at("f_zeta").view;
  auto fzeta_exact_view = swe.tracers.at("f_zeta_exact").view;
  auto s_exact_view = swe.tracers.at("s_exact").view;
  auto slap_exact_view = swe.tracers.at("slap_exact").view;
  auto rhs_view = swe.tracers.at("div_rhs").view;

  const auto face_x = swe.mesh.faces.phys_crds.view;
  const auto zeta_view = swe.relative_vorticity.view;
  const auto gfcu_avg_view = swe.grad_f_cross_u_avg.view;
  const auto dd_avg_view = swe.double_dot_avg.view;
  const auto slap_view = swe.surface_laplacian.view;
  Kokkos::parallel_for(swe.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(face_x, i, Kokkos::ALL);
      const Real cos_lat_sq = 1-square(xi[2]);
      const Real sin_lat_sq = square(xi[2]);

      fzeta_view(i) = coriolis.f(xi) * zeta_view(i);
      fzeta_exact_view(i) = 4*Omega*u0*sin_lat_sq;

      s_exact_view(i) = h0 + Omega * u0 * cos_lat_sq / g;
      slap_exact_view(i) = 2*Omega*u0/g * (2*sin_lat_sq - cos_lat_sq);

      rhs_view(i) = fzeta_view(i) + gfcu_avg_view(i) - dd_avg_view(i) - slap_view(i);
    });

  auto gf_cross_u_exact_view = swe.diags.at("grad_f_cross_u_exact").view;
  const auto vert_x = swe.mesh.vertices.phys_crds.view;
  Kokkos::parallel_for(swe.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vert_x, i, Kokkos::ALL);
      const Real cos_lat_sq = 1-square(xi[2]);
      gf_cross_u_exact_view(i) = -2*Omega*u0*cos_lat_sq;
    });
}

