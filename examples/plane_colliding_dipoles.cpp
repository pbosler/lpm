#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "lpm_incompressible2d_rk2_impl.hpp"
#include "mesh/lpm_bivar_remesh.hpp"
#include "mesh/lpm_bivar_remesh_impl.hpp"
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
  using vorticity_type = CollidingDipolePairPlane;
  using coriolis_type = CoriolisBetaPlane;
  using geo = PlaneGeometry;

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

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("colliding_dipoles"));
      input.add_option(output_file_root_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

      user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", Int(1e9));
      input.add_option(remesh_interval_option);

      user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
      input.add_option(remesh_strategy_option);
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
    const Real epsilon = input.get_option("kernel_smoothing_parameter").get_real();
    auto plane = std::make_unique<Incompressible2D<seed_type>>(mesh_params, coriolis, epsilon);

    // set problem initial conditions
    vorticity_type vorticity;
    plane->init_vorticity(vorticity);
    plane->init_direct_sums();

    const auto vel_range = plane->velocity_active.range(plane->mesh.n_faces_host());
    const Real cr = vel_range.second * dt / plane->mesh.appx_mesh_size();
    logger.info("velocity magnitude (min, max) = ({}, {}); approximate Courant number = {}",
      vel_range.first, vel_range.second, cr);
    logger.info("mesh initialized");
    logger.info(plane->info_string());
    if (cr > 0.5) {
        logger.warn("Courant number {} may be too high.", cr);
    }
    const Int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy = input.get_option("remesh_strategy").get_str();

#ifdef LPM_USE_VTK
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int());
    const std::string remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    const std::string vtk_file_root = input.get_option("output_file_directory").get_str() +
       "/" + input.get_option("output_file_root").get_str() +
       "_" + seed_type::id_string() + resolution_str + "_" + remesh_str + "_";
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
    auto solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *plane);
    logger.info(solver->info_string());
    int remesh_counter = 0;

    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      plane->advance_timestep(*solver);
      logger.debug("t = {}", plane->t);

      if ((t_idx+1)%remesh_interval == 0) {
        logger.debug("remesh {} triggered by remesh interval", ++remesh_counter);

        auto new_plane = std::make_unique<Incompressible2D<seed_type>>(mesh_params, coriolis, epsilon);
        auto remesh = bivar_remesh(*new_plane, *plane);

        if (remesh_strategy == "direct") {
          remesh.uniform_direct_remesh();
        }
        else {
          remesh.uniform_indirect_remesh(vorticity, coriolis);
        }

        plane = std::move(new_plane);
        plane->update_device();
        auto new_solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *plane, solver->t_idx);
        solver = std::move(new_solver);

      }

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
