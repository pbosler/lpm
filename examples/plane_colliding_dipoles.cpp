#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "lpm_swe_rk4.hpp"
#include "lpm_swe_rk4_impl.hpp"
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
  Logger<> logger("colliding_dipoles", Log::level::debug, comm);

  // compile-time settings
  // mesh seed
  using seed_type = QuadRectSeed; // TriHexSeed;

  // plane gravity wave problem setup
  using topography_type  = ZeroBottom;
  using init_sfc_type = UniformDepthSurface;
  using vorticity_type = CollidingDipolePairPlane;
  using coriolis_type = CoriolisBetaPlane;
  using geo = PlaneGeometry;
  using pse_type = pse::BivariateOrder8<geo>;

  Kokkos::initialize(argc, argv);
  { // Kokkos scope
    user::Input input("plane_gravity_wave");
    {
      // define user parameters
      user::Option tfinal_option("tfinal", "-tf", "--time_final", "time final", 0.5);
      input.add_option(tfinal_option);

      user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of steps", 5);
      input.add_option(nsteps_option);

      user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree depth", 4);

      input.add_option(tree_depth_option);
      user::Option f_coriolis_option("f-coriolis", "-f", "--f-coriolis", "f coriolis", 0.0);
      input.add_option(f_coriolis_option);

      user::Option beta_coriolis_option("beta-coriolis", "-b", "--beta-coriolis", "beta coriolis", 0.0);
      input.add_option(beta_coriolis_option);

      user::Option mesh_radius_option("mesh_radius", "-r", "--radius", "mesh radius", 6.0);
      input.add_option(mesh_radius_option);

      user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer", "amr memory buffer", 0);
      input.add_option(amr_refinement_buffer_option);

      user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit", "amr refinement limit", 0);
      input.add_option(amr_refinement_limit_option);

      user::Option output_file_directory_option("output_file_directory", "-odir", "--output-dir", "output file directory", std::string("."));
      input.add_option(output_file_directory_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("planar_dipoles"));
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
    PolyMeshParameters<seed_type> mesh_params(
      input.get_option("tree_depth").get_int(),
      input.get_option("mesh_radius").get_real(),
      input.get_option("amr_limit").get_int());

    coriolis_type coriolis(input.get_option("f-coriolis").get_real(),
      input.get_option("beta-coriolis").get_real());

    auto plane = std::make_unique<SWE<seed_type>>(mesh_params, coriolis);

    // set problem initial conditions
    topography_type topo;
    init_sfc_type sfc;
    plane->init_surface(topo, sfc);
    constexpr bool depth_set = true;
    constexpr bool do_velocity = true;
    vorticity_type vorticity;
    plane->init_vorticity(vorticity, depth_set);
    plane->set_kernel_parameters(input.get_option("kernel_smoothing_parameter").get_real(),
      pse_type::epsilon(plane->mesh.appx_mesh_size(), input.get_option("pse_kernel_width_power").get_real()));
    plane->init_direct_sums(do_velocity);

    logger.info("mesh initialized");
    logger.info(plane->info_string());

#ifdef LPM_USE_VTK
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int());
    const std::string vtk_file_root = input.get_option("output_file_directory").get_str() +
       "/" + input.get_option("output_file_root").get_str() +
       "_" + seed_type::id_string() + resolution_str + "_";
    {
      plane->update_host();
      auto vtk = vtk_mesh_interface(*plane);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file: {}", plane->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif

    // setup time stepper
    auto solver = std::make_unique<SWERK4<seed_type, topography_type>>(dt, *plane, topo);
    logger.info(solver->info_string());

    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      plane->advance_timestep(*solver);
      logger.debug("t = {}", plane->t);

#ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {
        plane->update_host();
        auto vtk = vtk_mesh_interface(*plane);
        auto ctr_str = zero_fill_str(++frame_counter);
        const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
        logger.info("writing output at t = {} to file: {}", plane->t, vtk_fname);
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
