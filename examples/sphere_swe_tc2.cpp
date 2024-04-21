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

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("sphere_swe_tc2", Log::level::debug, comm);

  // compile-time settings
  // mesh seed (choose one, comment-out the other
  using seed_type = CubedSphereSeed;
//   using seed_type = IcosTriSphereSeed;

  // sphere test case 2 problem setup
  using topography_type  = ZeroBottom;
  using init_sfc_type = SphereTestCase2InitialSurface;
  using coriolis_type = CoriolisSphere;
  using vorticity_type = SphereTestCase2Vorticity;
  using geo = SphereGeometry;
  using pse_type = pse::BivariateOrder8<PlaneGeometry>;

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

      user::Option pse_power_option("pse_kernel_width_power", "-pse", "--pse-kernel-width-power", "pse kernel width power",
        11.0/20);
      input.add_option(pse_power_option);
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

    // set problem initial conditions
    topography_type topo;
    init_sfc_type sfc;
    sphere->init_surface(topo, sfc);
    constexpr bool depth_set = true;
    vorticity_type vorticity;
    sphere->init_vorticity(vorticity, depth_set);
    constexpr bool do_velocity = true;
    sphere->set_kernel_parameters(input.get_option("kernel_smoothing_parameter").get_real(),
      pse_type::epsilon(sphere->mesh.appx_mesh_size(), input.get_option("pse_kernel_width_power").get_real()));

    sphere->init_direct_sums(do_velocity);

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
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int());
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + "_";
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
