#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe.hpp"
// #include "lpm_swe_problem_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "lpm_swe_rk2.hpp"
#include "lpm_swe_rk2_impl.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

using namespace Lpm;

template <typename SeedType>
void tc2_setup_tracers(SWE<SeedType>& swe);

template <typename SeedType>
void tc2_exact_sol(SWE<SeedType>& swe, const CoriolisSphere& coriolis);

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
  using init_sfc_type = SphereTestCase2InitialSurface;
  using coriolis_type = CoriolisSphere;
  using vorticity_type = SphereTestCase2Vorticity;
  using geo = SphereGeometry;

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

      user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer", "amr memory buffer", 0);
      input.add_option(amr_refinement_buffer_option);

      user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit", "amr refinement limit", 0);
      input.add_option(amr_refinement_limit_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("sphere_tc2"));
      input.add_option(output_file_root_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

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
    PolyMeshParameters<seed_type> mesh_params(
      input.get_option("tree_depth").get_int(),
      sphere_radius,
      input.get_option("amr_limit").get_int());

    coriolis_type coriolis(omega);

    auto sphere = std::make_unique<SWE<seed_type>>(mesh_params, coriolis);
    sphere->g = gravity;
    const Real cr = constants::PI/6 * dt / sphere->mesh.appx_mesh_size();
    logger.info("Courant number for this problem is appx. {}", cr);

    // set problem initial conditions
    topography_type topo;
    init_sfc_type sfc;
    sphere->init_surface(topo, sfc);
    constexpr bool depth_set = true;
    vorticity_type vorticity;
    sphere->init_vorticity(vorticity, depth_set);
    constexpr bool do_velocity = true;
    sphere->set_kernel_parameters(input.get_option("kernel_smoothing_parameter").get_real(),
      0);
    sphere->init_direct_sums(do_velocity);

    tc2_setup_tracers(*sphere);
    tc2_exact_sol(*sphere, coriolis);

    // setup time stepper
    constexpr Int gmls_order = 4;
    const gmls::Params gmls_params(gmls_order, SphereGeometry::ndim);
    auto solver = std::make_unique<SWERK2<seed_type, topography_type>>(dt, *sphere, topo, gmls_params);
    logger.info(solver->info_string());

    logger.info("mesh initialized");
    constexpr int tabs = 0;
    constexpr bool verbose = false;
    logger.info(sphere->info_string(tabs, verbose));

#ifdef LPM_USE_VTK
    const std::string eps_str = "eps"+float_str(input.get_option("kernel_smoothing_parameter").get_real());
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


    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      sphere->advance_timestep(*solver);
      logger.debug("t = {}", sphere->t);

#ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {

        tc2_exact_sol(*sphere, coriolis);
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
void tc2_setup_tracers(SWE<SeedType>& swe) {
  swe.allocate_scalar_tracer("coriolis_grad_cross_u");
  swe.allocate_scalar_tracer("f_zeta");
  swe.allocate_scalar_tracer("div_rhs");
  swe.allocate_scalar_tracer("slap_exact");
  swe.allocate_scalar_tracer("f_zeta_exact");
  swe.allocate_scalar_tracer("grad_f_cross_u_exact");
  swe.allocate_scalar_tracer("s_exact");
}

template <typename SeedType>
void tc2_exact_sol(SWE<SeedType>& swe, const CoriolisSphere& coriolis) {
  constexpr Real Omega = 2*constants::PI;
  constexpr Real u0 = constants::PI/6;
  constexpr Real g = 1;
  constexpr Real h0 = 10;
  auto fz_view = swe.tracer_passive.at("f_zeta").view;
  auto f_zeta_exact_view = swe.tracer_passive.at("f_zeta_exact").view;
  auto slap_exact_view = swe.tracer_passive.at("slap_exact").view;
  auto cgu_view = swe.tracer_passive.at("coriolis_grad_cross_u").view;
  auto cgu_exact_view = swe.tracer_passive.at("grad_f_cross_u_exact").view;
  auto rhs_view = swe.tracer_passive.at("div_rhs").view;
  auto zeta_view = swe.rel_vort_passive.view;
  auto vc_view = swe.mesh.vertices.phys_crds.view;
  auto vv_view = swe.velocity_passive.view;
  auto slap_view = swe.surf_lap_passive.view;
  auto s_exact_view = swe.tracer_passive.at("s_exact").view;
  auto ddot_view = swe.double_dot_passive.view;
  Kokkos::parallel_for(swe.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vc_view, i, Kokkos::ALL);
      const auto ui = Kokkos::subview(vv_view, i, Kokkos::ALL);
      const Real cos_lat_sq = 1-square(xi[2]);
      const Real sin_lat_sq = square(xi[2]);

      cgu_view(i) = coriolis.grad_f_cross_u(xi, ui);
      cgu_exact_view(i) = -2*Omega*u0*cos_lat_sq;

      fz_view(i) = zeta_view(i) * coriolis.f(xi);
      f_zeta_exact_view(i) = 4*Omega*u0*sin_lat_sq;

      s_exact_view(i) = h0 + Omega * u0 * cos_lat_sq / g;

      slap_exact_view(i) = 2*Omega*u0/g * (2*sin_lat_sq - cos_lat_sq);

      rhs_view(i) = fz_view(i) + cgu_view(i) - ddot_view(i) - slap_view(i);
    });

  fz_view = swe.tracer_active.at("f_zeta").view;
  f_zeta_exact_view = swe.tracer_active.at("f_zeta_exact").view;
  slap_exact_view = swe.tracer_active.at("slap_exact").view;
  cgu_view = swe.tracer_active.at("coriolis_grad_cross_u").view;
  cgu_exact_view = swe.tracer_active.at("grad_f_cross_u_exact").view;
  rhs_view = swe.tracer_active.at("div_rhs").view;
  zeta_view = swe.rel_vort_active.view;
  vc_view = swe.mesh.faces.phys_crds.view;
  vv_view = swe.velocity_active.view;
  slap_view = swe.surf_lap_active.view;
  s_exact_view = swe.tracer_active.at("s_exact").view;
  ddot_view = swe.double_dot_active.view;

  Kokkos::parallel_for(swe.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vc_view, i, Kokkos::ALL);
      const auto ui = Kokkos::subview(vv_view, i, Kokkos::ALL);
      const Real cos_lat_sq = 1-square(xi[2]);
      const Real sin_lat_sq = square(xi[2]);

      cgu_view(i) = coriolis.grad_f_cross_u(xi, ui);
      cgu_exact_view(i) = -2*Omega*u0*cos_lat_sq;

      fz_view(i) = zeta_view(i) * coriolis.f(xi);
      f_zeta_exact_view(i) = 4*Omega*u0*sin_lat_sq;

      slap_exact_view(i) = 2*Omega*u0/g * (2*sin_lat_sq - cos_lat_sq);

      s_exact_view(i) = h0 + Omega * u0 * cos_lat_sq / g;

      rhs_view(i) = fz_view(i) + cgu_view(i) - ddot_view(i) - slap_view(i);
    });

}
