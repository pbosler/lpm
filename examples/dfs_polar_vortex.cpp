#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "dfs/lpm_compadre_dfs_remesh.hpp"
#include "dfs/lpm_compadre_dfs_remesh_impl.hpp"
#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_ftle.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <cstdio>
#include <iomanip>
#include <iostream>

using namespace Lpm;
using namespace Lpm::DFS;

void init_input(user::Input& input);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("dfs_polar_vortex", Log::level::debug, comm);

  Kokkos::initialize(argc, argv);
  { // Kokkos scope

    user::Input input("dfs_polar_vortex");
    init_input(input);
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      logger.info(input.info_string());
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    logger.info(input.info_string());

    using SeedType = CubedSphereSeed;
    using Coriolis = CoriolisSphere;
    using Vorticity = JuckesMcIntyre87;
    using Solver = DFSRK4<SeedType>;
    const Int unif_mesh_depth = input.get_option("tree_depth").get_int();
    constexpr Real sphere_radius = 1;
    Int amr_buffer = input.get_option("amr_buffer").get_int();
    Int amr_limit = input.get_option("amr_limit").get_int();
    Real max_circ_tol = input.get_option("max_circulation_tol").get_real();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit = input.get_option("amr_both").get_int();
    }
    PolyMeshParameters<SeedType> mesh_params(unif_mesh_depth, sphere_radius, amr_buffer, amr_limit);
    const bool amr = (mesh_params.is_adaptive() and max_circ_tol < 1);
    const Int nlon = input.get_option("nlon").get_int();
    const Int gmls_order = input.get_option("gmls_interpolation_order").get_int();
    gmls::Params gmls_params(gmls_order);

    // DFSBVE initialization
    Vorticity vorticity(input.get_option("forcing_F0").get_real(),
      input.get_option("forcing_tp").get_real(),
      input.get_option("forcing_tf").get_real(),
      input.get_option("forcing_beta").get_real(),
      input.get_option("forcing_theta0").get_real(),
      input.get_option("forcing_theta1").get_real());
    auto sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
    const Real ftle_tol = input.get_option("ftle_tol").get_real();
    sphere->init_vorticity(vorticity);

    logger.info("initial mesh ready: {}", sphere->info_string());
    logger.info("initial uniform relative vorticity: {}", sphere->rel_vort_passive.info_string());
    logger.info("initial uniform relative vorticity: {}", sphere->rel_vort_active.info_string());

    /**
      Initial adaptive refinement
    */

    if (amr) {
      Refinement<SeedType> refiner(sphere->mesh);
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
        sphere->init_vorticity(vorticity);

        face_start_idx = face_end_idx;
      }
      sphere->set_mesh_ready();
      sphere->init_mesh_grid_coupling();
      logger.info(sphere->info_string());
      logger.info("initial adaptive relative vorticity: {}", sphere->rel_vort_passive.info_string());
      logger.info("initial adaptive relative vorticity: {}", sphere->rel_vort_active.info_string());
    }
    else {
      logger.info("amr is not enabled; using uniform meshes.");
    }
    sphere->init_velocity_from_vorticity();

    /**
      Solver setup
    */
    using SolverType = DFS::DFSRK4<SeedType>;
    const Real tfinal = input.get_option("tfinal").get_real();
    const auto vel_range = sphere->velocity_active.range(sphere->mesh.n_faces_host());
    Real dt;
    Int nsteps;
    if (input.get_option("use_dt").get_bool()) {
      if (input.get_option("use_nsteps").get_bool()) {
        logger.error("cannot use both dt and nsteps to determine time step size; choose one.  Defaulting to dt.");
      }
      dt = input.get_option("dt").get_real();
      nsteps = int(tfinal / dt);
    }
    else {
      nsteps = input.get_option("nsteps").get_int();
      dt = tfinal / nsteps;
    }
    const Real cr = vel_range.second * dt / sphere->mesh.appx_mesh_size();
    logger.info("dt = {}, cr = {}", dt, cr);

    const Int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy = input.get_option("remesh_strategy").get_str();
    const bool use_ftle = (input.get_option("remesh_trigger").get_str() == "ftle");

    auto solver = std::make_unique<SolverType>(dt, *sphere);
    std::vector<Real> total_vorticity(nsteps+1);
    std::vector<Real> total_kinetic_energy(nsteps+1);
    std::vector<Real> total_enstrophy(nsteps+1);
    std::vector<Real> ftle_max(nsteps+1);
    std::vector<Real> time(nsteps+1);
    ftle_max[0] = 0;
    total_vorticity[0] = sphere->total_vorticity();
    total_kinetic_energy[0] = sphere->total_kinetic_energy();
    total_enstrophy[0] = sphere->total_enstrophy();

    logger.info("initial integrals: relative vorticity: {}", sphere->rel_vort_passive.info_string());
    logger.info("initial integrals: relative vorticity: {}", sphere->rel_vort_active.info_string());
    logger.info("initial total vorticity, kinetic energy, and enstrophy = ({}, {}, {})",
      total_vorticity[0], total_kinetic_energy[0], total_enstrophy[0]);

    std::string amr_str = "_";
    if (amr) {
      amr_str = "amr" + std::to_string(amr_limit) + "_";
      if (max_circ_tol < 1) {
        amr_str += "gamma_tol" + float_str(max_circ_tol);
      }
      amr_str += "_";
    }
    const std::string resolution_str =  std::to_string(unif_mesh_depth) + dt_str(dt);
    std::string remesh_str;
    if (use_ftle) {
      remesh_str = remesh_strategy + "ftle" + float_str(ftle_tol,3);
    }
    else {
      remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    }
    const std::string ofile_root = input.get_option("output_file_root").get_str() + "_" + SeedType::id_string() + resolution_str + "_" + remesh_str + amr_str;
    const std::string vtk_file_root = ofile_root;
    int vtk_counter = 0;
    const int write_frequency = input.get_option("output_write_frequency").get_int();

    logger.info("output files will have filename root: {}", ofile_root);

    {
      /** output initial conditions to mesh/grid files */
      auto vtk_mesh = vtk_mesh_interface(*sphere);
      auto vtk_grid = vtk_grid_interface(*sphere);
      const std::string mesh_vtk_file = vtk_file_root + zero_fill_str(vtk_counter) + vtp_suffix();
      const std::string grid_vtk_file = vtk_file_root + zero_fill_str(vtk_counter) + vts_suffix();
      vtk_mesh.write(mesh_vtk_file);
      vtk_grid.write(grid_vtk_file);
      ++vtk_counter;
    }

    /**
      time stepping
    */
    int rm_counter = 0;
    Real tref = 0;
    Real max_ftle = 0;
    for (int t_idx = 0; t_idx <nsteps; ++t_idx) {
      max_ftle = get_max_ftle(sphere->ftle.view, sphere->mesh.faces.mask, sphere->mesh.n_faces_host());
      logger.debug("t = {}, max_ftle = {}", sphere->t, max_ftle);

      const bool ftle_trigger = (use_ftle and max_ftle > ftle_tol);
      const bool interval_trigger = ((t_idx+1)%remesh_interval == 0);
      const bool do_remesh = (ftle_trigger or interval_trigger);

       if (do_remesh) {
        ++rm_counter;
        if (interval_trigger) {
          logger.debug("remesh {} triggered by remesh interval", rm_counter);
        }
        else {
          logger.info("remesh {} triggered by ftle", rm_counter);
        }

        auto new_sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
        new_sphere->allocate_tracer(lat0);
        new_sphere->t = sphere->t;

        auto remesh = compadre_dfs_remesh(*new_sphere, *sphere, gmls_params);
        if (amr) {
          logger.error("AMR not yet implemented for DFS; skipping remesh step.");
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            logger.error("indirect remesh not implemented for DFS; skipping remesh step.");
          }
        }
        logger.info(remesh.info_string());

        tref = sphere->t;
        sphere = std::move(new_sphere);
        sphere->sync_solver_views();
        sphere->t_ref = tref;
        solver.reset(new SolverType(dt, *sphere, solver->t_idx));

      }


      sphere->advance_timestep(*solver);

      Kokkos::parallel_for(sphere->mesh.n_faces_host(),
        ComputeFTLE<SeedType>(sphere->ftle.view,
          sphere->mesh.vertices.phys_crds.view,
          sphere->ref_crds_passive.view,
          sphere->mesh.faces.phys_crds.view,
          sphere->ref_crds_active.view,
          sphere->mesh.faces.verts,
          sphere->mesh.faces.mask,
          sphere->t - sphere->t_ref));

      time[t_idx+1] = (t_idx+1) * dt;
      ftle_max[t_idx+1] = max_ftle;
      total_vorticity[t_idx+1] = sphere->total_vorticity();
      total_kinetic_energy[t_idx+1] = sphere->total_kinetic_energy();
      total_enstrophy[t_idx+1] = sphere->total_enstrophy();

      if ((t_idx+1)%write_frequency == 0) {
        auto vtk_mesh = vtk_mesh_interface(*sphere);
        auto vtk_grid = vtk_grid_interface(*sphere);
        const std::string mesh_vtk_file = vtk_file_root + zero_fill_str(vtk_counter) + vtp_suffix();
        const std::string grid_vtk_file = vtk_file_root + zero_fill_str(vtk_counter) + vts_suffix();
        vtk_mesh.write(mesh_vtk_file);
        vtk_grid.write(grid_vtk_file);
        ++vtk_counter;
      }
      const auto rel_vort_range = sphere->rel_vort_passive.range(sphere->mesh.vertices.nh());
      logger.info("t = {}, rel. vort passive range : ({}, {})", (t_idx+1)*dt, rel_vort_range.first, rel_vort_range.second);
    }

  } // Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}

void init_input(user::Input& input) {
  user::Option nlon_option("nlon", "-nlon", "--n_lon", "number of longitude points in DFS grid", 20);
  input.add_option(nlon_option);

  user::Option omega_option("omega", "-omg", "--omega", "background rotation rate of the sphere", 2*constants::PI);
  input.add_option(omega_option);

  user::Option tfinal_option("tfinal", "-tf", "--time-final", "time final", 0.025);
  input.add_option(tfinal_option);

  user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of time steps", 1);
  input.add_option(nsteps_option);

  user::Option dt_option("dt", "-dt", "--time-step", "time step size", 0.025);
  input.add_option(dt_option);

  user::Option use_dt_option("use_dt", "-udt", "--use-dt", "use dt for time step size", true);
  input.add_option(use_dt_option);

  user::Option use_nsteps_option("use_nsteps", "-un", "--use-nsteps", "use nsteps for time step size", false);
  input.add_option(use_nsteps_option);

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

  user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("dfs_polar_vortex"));
  input.add_option(output_file_root_option);

  user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", std::numeric_limits<int>::max());
  input.add_option(remesh_interval_option);

  user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
  input.add_option(remesh_strategy_option);

  user::Option gmls_interpolation_order("gmls_interpolation_order", "-g", "--gmls-order", "polynomial order for gmls-based interpolation", 6);
  input.add_option(gmls_interpolation_order);

  user::Option remesh_trigger_option("remesh_trigger", "-rt", "--remesh-trigger", "trigger for a remeshing : ftle or an interval", std::string("interval"), std::set<std::string>({"interval", "ftle"}));
  input.add_option(remesh_trigger_option);

  user::Option ftle_tolerance_option("ftle_tol", "-ftle", "--ftle-tol", "max value for ftle before remesh", 2.0);
  input.add_option(ftle_tolerance_option);

  user::Option pv_forcing_tp("forcing_tp", "-tp", "--forcing-tp", "ramp-up time for forcing amplitude", 4.0);
  input.add_option(pv_forcing_tp);

  user::Option pv_forcing_tf("forcing_tf", "-pvtf", "--forcing-tf", "end time for forcing", 15.0);
  input.add_option(pv_forcing_tf);

  user::Option pv_forcing_theta1("forcing_theta1", "-th1", "--forcing-theta1", "central latitude of forcing", constants::PI/3);
  input.add_option(pv_forcing_theta1);

  user::Option pv_forcing_theta0("forcing_theta0", "-th0", "--forcing-theta0", "central latitude of initial jet", 15*constants::PI / 32);
  input.add_option(pv_forcing_theta0);

  user::Option pv_forcing_beta("forcing_beta", "-beta", "--forcing-beta", "shape parameter of polar jet's meridional profile", 1.5);
  input.add_option(pv_forcing_beta);

  user::Option pv_f0("forcing_F0", "-f0", "--forcing-f0", "maximum forcing", 1.2 * constants::PI);
  input.add_option(pv_f0);
}
