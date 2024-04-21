#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_rk2_impl.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_tracer_gallery.hpp"
#include "util/lpm_string_util.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

using namespace Lpm;

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("bve_gaussian_vortex", Log::level::debug, comm);
  using seed_type = CubedSphereSeed;
  using Coriolis = CoriolisSphere;
  using Vorticity = GaussianVortexSphere;
  using Ftle = FtleTracer<SphereGeometry>;
  using Lat0 = LatitudeTracer;
  using Solver = Incompressible2DRK2<seed_type>;

  Kokkos::initialize(argc, argv);
  {
    user::Input input("bve_gauss_vort");
    {
      user::Option tfinal_option("tfinal", "-tf", "--time-final", "time final", 0.5);
      input.add_option(tfinal_option);

      user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of time steps", 5);
      input.add_option(nsteps_option);

      user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree initial uniform depth", 4);
      input.add_option(tree_depth_option);

      user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer", "amr memory buffer", 0);
      input.add_option(amr_refinement_buffer_option);

      user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit", "amr refinement limit", 0);
      input.add_option(amr_refinement_limit_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("gauss_vort"));
      input.add_option(output_file_root_option);
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
    logger.info("dt = {}", dt);

    /**
    Build the particle/panel mesh
    */
    constexpr Real sphere_radius = 1;
    PolyMeshParameters<seed_type> mesh_params(input.get_option("tree_depth").get_int(),
      sphere_radius, input.get_option("amr_buffer").get_int());

    Coriolis coriolis;
    Vorticity gauss_vort;
    auto sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
      coriolis, input.get_option("kernel_smoothing_parameter").get_real());
    sphere->init_vorticity(gauss_vort);
    sphere->init_direct_sums();

    Ftle ftle;
    Lat0 lat0;
    sphere->init_tracer(ftle, "ftle");
    sphere->init_tracer(lat0, "initial_latitude");

    const auto vel_range = sphere->velocity_active.range(sphere->mesh.n_faces_host());
    const Real cr = vel_range.second * dt / sphere->mesh.appx_mesh_size();
    logger.info("velocity magnitude (min, max) = ({}, {}); approximate Courant number = {}",
      vel_range.first, vel_range.second, cr);
    if constexpr (std::is_same<Solver, Incompressible2DRK2<seed_type>>::value) {
      if (cr > 0.5) {
        logger.warn("Courant number {} is likely too high.", cr);
      }
    }
    auto solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *sphere);

#ifdef LPM_USE_VTK
    const std::string resolution_str =
      std::to_string(input.get_option("tree_depth").get_int()) + dt_str(dt);
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + "_";
    {
      sphere->update_host();
      auto vtk = vtk_mesh_interface(*sphere);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file {}", sphere->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif

    /**
    time stepping
    */
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


  } // kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}
