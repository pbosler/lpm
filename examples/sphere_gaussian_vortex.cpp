#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "lpm_incompressible2d_rk2_impl.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_tracer_gallery.hpp"
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"
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

      user::Option max_circulation_option("max_circulation_tol", "-c", "--circuluation-max", "amr max circulation tolerance", std::numeric_limits<Real>::max());
      input.add_option(max_circulation_option);

      user::Option amr_both_option("amr_both", "-amr", "--amr-both", "both amr buffer and limit values", LPM_NULL_IDX);
      input.add_option(amr_both_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("gauss_vort"));
      input.add_option(output_file_root_option);

      user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", std::numeric_limits<int>::max());
      input.add_option(remesh_interval_option);

      user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
      input.add_option(remesh_strategy_option);

      user::Option remesh_interpolation_order("remesh_interpolation_order", "-ro", "--remesh-order", "polynomial order for gmls-based remesh interpolation", 4);
      input.add_option(remesh_interpolation_order);
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
      AMR
    */
    Real max_circ_tol = input.get_option("max_circulation_tol").get_real();
    Int amr_buffer = input.get_option("amr_buffer").get_int();
    Int amr_limit = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit = input.get_option("amr_both").get_int();
    }
    const bool amr = (amr_buffer > 0 and amr_limit > 0);

    /**
    Build the particle/panel mesh
    */
    constexpr Real sphere_radius = 1;
    PolyMeshParameters<seed_type> mesh_params(
        input.get_option("tree_depth").get_int(),
        sphere_radius,
        amr_buffer,
        amr_limit);

    Coriolis coriolis;
    Vorticity gauss_vort;
    auto sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
      coriolis, input.get_option("kernel_smoothing_parameter").get_real());
    sphere->init_vorticity(gauss_vort);

    if (amr) {
      Refinement<seed_type> refiner(sphere->mesh);
      ScalarIntegralFlag max_circulation_flag(refiner.flags,
        sphere->rel_vort_active.view,
        sphere->mesh.faces.area,
        sphere->mesh.faces.mask,
        sphere->mesh.n_faces_host(),
        max_circ_tol);

        max_circulation_flag.set_tol_from_relative_value();
        max_circ_tol = max_circulation_flag.tol;

        logger.info("amr is enabled with limit {}, max_circ_tol = {}",
          amr_limit, max_circ_tol);

        Index face_start_idx = 0;
        for (int i=0; i<amr_limit; ++i) {
          const Index face_end_idx = sphere->mesh.n_faces_host();
          refiner.iterate(face_start_idx, face_end_idx, max_circulation_flag);

          logger.info("amr iteration {}: initial circulation refinement count = {}",
            i, refiner.count[0]);

          sphere->mesh.divide_flagged_faces(refiner.flags, logger);
          sphere->update_device();
          sphere->init_vorticity(gauss_vort);

          face_start_idx = face_end_idx;
        }
    }
    else {
      logger.info("amr is not enabled; using uniform meshes.");
    }
    sphere->init_direct_sums();

    Ftle ftle;
    Lat0 lat0;
    sphere->init_tracer(ftle);
    sphere->init_tracer(lat0);

    const auto vel_range = sphere->velocity_active.range(sphere->mesh.n_faces_host());
    const Real cr = vel_range.second * dt / sphere->mesh.appx_mesh_size();
    logger.info("velocity magnitude (min, max) = ({}, {}); approximate Courant number = {}",
      vel_range.first, vel_range.second, cr);
    if constexpr (std::is_same<Solver, Incompressible2DRK2<seed_type>>::value) {
      if (cr > 0.5) {
        logger.warn("Courant number {} is likely too high.", cr);
      }
    }
    const int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy = input.get_option("remesh_strategy").get_str();
    auto solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *sphere);
    gmls::Params gmls_params(input.get_option("remesh_interpolation_order").get_int());

#ifdef LPM_USE_VTK
    std::string amr_str = "_";
    if (amr) {
      amr_str = "amr" + std::to_string(amr_limit) + "_";
      if (max_circ_tol < 1) {
        amr_str += "gamma_tol" + float_str(max_circ_tol);
      }
      amr_str += "_";
    }
    const std::string resolution_str =
      std::to_string(input.get_option("tree_depth").get_int()) + dt_str(dt);
    const std::string remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + "_" + remesh_str + amr_str;
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
    int rm_counter = 0;
    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      if ( (t_idx+1)%remesh_interval == 0 ) {
        logger.debug("remesh {} triggered by remesh interval", ++rm_counter);

        auto new_sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
           coriolis, input.get_option("kernel_smoothing_parameter").get_real());
        new_sphere->t = sphere->t;
        new_sphere->allocate_tracer(ftle);
        new_sphere->allocate_tracer(lat0);

        auto remesh = compadre_remesh(*new_sphere, *sphere, gmls_params);
        if (amr) {
          Refinement<seed_type> refiner(new_sphere->mesh);
            ScalarIntegralFlag max_circulation_flag(refiner.flags,
              new_sphere->rel_vort_active.view,
              new_sphere->mesh.faces.area,
              new_sphere->mesh.faces.mask,
              new_sphere->mesh.n_faces_host(),
              max_circ_tol);
          if (remesh_strategy == "direct") {
            remesh.adaptive_direct_remesh(refiner, max_circulation_flag);
          }
          else {
            remesh.adaptive_indirect_remesh(refiner, max_circulation_flag,
              gauss_vort, coriolis, lat0, ftle);
          }
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            remesh.uniform_indirect_remesh(gauss_vort, coriolis, lat0, ftle);
          }
        }

        sphere = std::move(new_sphere);
        solver.reset(new Incompressible2DRK2<seed_type>(dt, *sphere, solver->t_idx));
      }

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
