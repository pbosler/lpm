#include "LpmConfig.h"
#include "dfs/lpm_compadre_dfs_remesh.hpp"
#include "dfs/lpm_compadre_dfs_remesh_impl.hpp"
#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
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

void set_exact_vorticity(DFS::DFSBVE<CubedSphereSeed>& sphere, const RossbyHaurwitz54& vorticity);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("dfs_bve_rh54", Log::level::info, comm);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
    /**
      program run
    */
    /**
      initialize problem
    */
    user::Input input("dfs_bve_rh54");
    init_input(input);
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      logger.info(input.info_string());
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    logger.info(input.info_string());
    // problem types: velocity and vorticity
    using Coriolis = CoriolisSphere;
    using Lat0     = LatitudeTracer;
    Coriolis coriolis(input.get_option("omega").get_real());
    RossbyWave54Velocity velocity(constants::PI / 7);
    RossbyHaurwitz54 vorticity(constants::PI / 7);
    if (vorticity.is_stationary()) {
      logger.info("stationary wave speed selected; RH54 u0 = {}, rh54_amplitude = {}", vorticity.u0, vorticity.rh54_amplitude);
      if (!velocity.is_stationary()) {
        logger.error("velocity parameters do not match vorticity parameters.");
      }
    }
    else {
      logger.info("non-stationary wave speed selected; RH54 u0 = {}, wave_amp = {}", vorticity.u0, vorticity.rh54_amplitude);
      logger.warn("non-stationary vorticity error not implemented yet.");
      // TODO: check for mismatch between general velocity/vorticity parameters
    }

    //  particle/panel/grid initialization
    using SeedType               = CubedSphereSeed;
    using Coriolis               = CoriolisSphere;
    const Int mesh_depth         = input.get_option("tree_depth").get_int();
    constexpr Real sphere_radius = 1;
    Int amr_buffer               = input.get_option("amr_buffer").get_int();
    Int amr_limit                = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit  = input.get_option("amr_both").get_int();
    }
    const bool amr = (amr_buffer > 0 and amr_limit > 0);
    PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius,
                                             amr_buffer, amr_limit);
    const Int nlon = input.get_option("nlon").get_int();
    const Int gmls_order =
        input.get_option("gmls_interpolation_order").get_int();
    gmls::Params gmls_params(gmls_order);
    gmls_params.eps_multiplier =
        input.get_option("gmls_eps_multiplier").get_real();

    // DFS initialization
    auto sphere =
        std::make_unique<DFS::DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
    const Real ftle_tol = input.get_option("ftle_tol").get_real();
    sphere->init_vorticity(vorticity);
    sphere->finalize_mesh_to_grid_coupling();
    sphere->init_velocity(velocity);
    /**
      Initial adaptive refinement
    */
    if (amr) {
      logger.warn("AMR not implemented yet; proceeding with a uniform mesh.");
    }
    Lat0 lat0;
    sphere->init_tracer(lat0);
    sphere->init_tracer(vorticity, "exact_vorticity");
    sphere->allocate_tracer("vorticity_error");
    logger.info(sphere->info_string());

    // Solver initialization : choose solver type at compile time
    //     using SolverType = DFS::DFSRK2<SeedType>;
    using SolverType = DFS::DFSRK3<SeedType>;
    //     using SolverType  = DFS::DFSRK4<SeedType>;
    const Real tfinal = input.get_option("tfinal").get_real();
    const auto vel_range =
        sphere->velocity_active.range(sphere->mesh.n_faces_host());
    Real dt;
    Int nsteps;
    if (input.get_option("use_dt").get_bool()) {
      if (input.get_option("use_nsteps").get_bool()) {
        logger.error(
            "cannot use both dt and nsteps to determine time step size; choose "
            "one.  Defaulting to dt.");
      }
      dt     = input.get_option("dt").get_real();
      nsteps = int(tfinal / dt);
    } else {
      nsteps = input.get_option("nsteps").get_int();
      dt     = tfinal / nsteps;
    }

    const Real cr = vel_range.second * dt / sphere->mesh.appx_mesh_size();
    logger.info("dt = {}, cr = {}", dt, cr);

    const Int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy =
        input.get_option("remesh_strategy").get_str();
    const bool use_ftle =
        (input.get_option("remesh_trigger").get_str() == "ftle");

    auto solver = std::make_unique<SolverType>(dt, *sphere);
    std::vector<Real> zeta_l1err(nsteps+1);
    std::vector<Real> zeta_l2err(nsteps+1);
    std::vector<Real> zeta_linferr(nsteps+1);
    std::vector<Real> total_vorticity(nsteps + 1);
    std::vector<Real> total_kinetic_energy(nsteps + 1);
    std::vector<Real> total_enstrophy(nsteps + 1);
    std::vector<Real> ftle_max(nsteps + 1);
    std::vector<Real> time(nsteps + 1);
    total_vorticity[0]      = sphere->total_vorticity();
    total_kinetic_energy[0] = sphere->total_kinetic_energy();
    total_enstrophy[0]      = sphere->total_enstrophy();

    logger.info(
        "initial total vorticity, kinetic energy, and enstrophy = ({}, {}, {})",
        total_vorticity[0], total_kinetic_energy[0], total_enstrophy[0]);

    std::string amr_str = "_";
    if (amr) {
      logger.warn("AMR not implemented yet.");
    }
    const std::string resolution_str = std::to_string(mesh_depth) + dt_str(dt);
    std::string remesh_str;
    if (use_ftle) {
      remesh_str = remesh_strategy + "ftle" + float_str(ftle_tol, 3);
    } else {
      remesh_str =
          (remesh_interval < nsteps
               ? remesh_strategy + "rm" + std::to_string(remesh_interval)
               : "no_rm");
    }
    const std::string ofile_root =
        input.get_option("output_file_root").get_str() + "_" +
        SeedType::id_string() + resolution_str + "_nlon" +
        std::to_string(nlon) + "_" + remesh_str + amr_str;
    const std::string vtk_file_root = ofile_root;
    logger.info("output will be written to files beginning: {}", ofile_root);

    int vtk_counter                 = 0;
    const int write_frequency =
        input.get_option("output_write_frequency").get_int();
    {
      /** output initial conditions to mesh/grid files */
      auto vtk_mesh = vtk_mesh_interface(*sphere);
      auto vtk_grid = vtk_grid_interface(*sphere);
      const std::string mesh_vtk_file =
          vtk_file_root + zero_fill_str(vtk_counter) + vtp_suffix();
      const std::string grid_vtk_file =
          vtk_file_root + zero_fill_str(vtk_counter) + vts_suffix();
      vtk_mesh.write(mesh_vtk_file);
      vtk_grid.write(grid_vtk_file);
      ++vtk_counter;
    }
    /**
    time stepping
    */
    int rm_counter = 0;
    Real tref      = 0;
    Real max_ftle  = 0;
    for (int t_idx = 0; t_idx < nsteps; ++t_idx) {
      max_ftle = get_max_ftle(sphere->ftle_active.view, sphere->mesh.faces.mask,
                              sphere->mesh.n_faces_host());
      logger.debug("t = {}, max_ftle = {}", sphere->t, max_ftle);

      const bool ftle_trigger     = (use_ftle and max_ftle > ftle_tol);
      const bool interval_trigger = ((t_idx + 1) % remesh_interval == 0);
      const bool do_remesh        = (ftle_trigger or interval_trigger);
      if (do_remesh) {
        ++rm_counter;
        if (interval_trigger) {
          logger.debug("remesh {} triggered by remesh interval", rm_counter);
        } else {
          logger.info("remesh {} triggered by ftle", rm_counter);
        }

        auto new_sphere =
            std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
        new_sphere->allocate_tracer(lat0.name());
        new_sphere->t = sphere->t;

        auto remesh = compadre_dfs_remesh(*new_sphere, *sphere, gmls_params);
        if (amr) {
          logger.error(
              "AMR not yet implemented for DFS; skipping remesh step.");
        } else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          } else {
            logger.error(
                "indirect remesh not implemented for DFS; skipping remesh "
                "step.");
          }
        }
        logger.info(remesh.info_string());

        tref   = sphere->t;
        sphere = std::move(new_sphere);

        sphere->finalize_mesh_to_grid_coupling();
        sphere->t_ref = tref;
        solver.reset(new SolverType(dt, *sphere, solver->t_idx));
      }

      sphere->advance_timestep(*solver);
      set_exact_vorticity(*sphere, vorticity);
      compute_error(sphere->tracer_passive.at("vorticity_error").view, sphere->rel_vort_passive.view, sphere->tracer_passive.at("exact_vorticity").view);
      compute_error(sphere->tracer_active.at("vorticity_error").view, sphere->rel_vort_active.view, sphere->tracer_active.at("exact_vorticity").view);
      const auto zeta_error_scalars = reduce_error(sphere->tracer_active.at("vorticity_error").view,
                                     sphere->tracer_active.at("exact_vorticity").view,
                                     sphere->mesh.faces.area);
      const ErrNorms zeta_error(zeta_error_scalars);

      time[t_idx + 1]                 = (t_idx + 1) * dt;
      zeta_l1err[t_idx+1]             = zeta_error.l1;
      zeta_l2err[t_idx+1]             = zeta_error.l2;
      zeta_linferr[t_idx+1]           = zeta_error.linf;
      ftle_max[t_idx + 1]             = max_ftle;
      total_vorticity[t_idx + 1]      = sphere->total_vorticity();
      total_kinetic_energy[t_idx + 1] = sphere->total_kinetic_energy();
      total_enstrophy[t_idx + 1]      = sphere->total_enstrophy();

      if ((t_idx + 1) % write_frequency == 0) {
        auto vtk_mesh = vtk_mesh_interface(*sphere);
        sphere->update_grid_velocity();
        auto vtk_grid = vtk_grid_interface(*sphere);
        const std::string mesh_vtk_file =
            vtk_file_root + zero_fill_str(vtk_counter) + vtp_suffix();
        const std::string grid_vtk_file =
            vtk_file_root + zero_fill_str(vtk_counter) + vts_suffix();
        vtk_mesh.write(mesh_vtk_file);
        vtk_grid.write(grid_vtk_file);
        ++vtk_counter;
      }
      const auto rel_vort_range =
          sphere->rel_vort_passive.range(sphere->mesh.vertices.nh());
      logger.info("t = {}, rel. vort passive range : ({}, {})",
                  (t_idx + 1) * dt, rel_vort_range.first,
                  rel_vort_range.second);
    }
    const std::string matlab_file = ofile_root + ".m";
    std::ofstream ofile(matlab_file);
    write_vector_matlab(ofile, "time", time);
    write_vector_matlab(ofile, "zeta_l1err", zeta_l1err);
    write_vector_matlab(ofile, "zeta_l2err", zeta_l2err);
    write_vector_matlab(ofile, "zeta_linferr", zeta_linferr);
    write_vector_matlab(ofile, "total_vorticity", total_vorticity);
    write_vector_matlab(ofile, "total_kinetic_energy", total_kinetic_energy);
    write_vector_matlab(ofile, "total_enstrophy", total_enstrophy);
    write_vector_matlab(ofile, "max_ftle", ftle_max);
    ofile.close();
    logger.info("output can be found in files beginning: {}", ofile_root);
  }  // Kokkos scope
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
}

void set_exact_vorticity(DFS::DFSBVE<CubedSphereSeed>& sphere, const RossbyHaurwitz54& vorticity) {

    LPM_REQUIRE_MSG(vorticity.is_stationary(), "Vorticity error assumes stationary RH wave");

    const auto vert_crds = sphere.mesh.vertices.phys_crds.view;
    const auto vert_exact_zeta = sphere.tracer_passive.at("exact_vorticity").view;
    const auto face_crds = sphere.mesh.faces.phys_crds.view;
    const auto face_exact_zeta = sphere.tracer_active.at("exact_vorticity").view;

    Kokkos::parallel_for(sphere.mesh.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mcrd = Kokkos::subview(vert_crds, i, Kokkos::ALL);
        vert_exact_zeta(i) = vorticity(mcrd[0], mcrd[1], mcrd[2]);
      });
    Kokkos::parallel_for(sphere.mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mcrd = Kokkos::subview(face_crds, i, Kokkos::ALL);
        face_exact_zeta(i) = vorticity(mcrd[0], mcrd[1], mcrd[2]);
      });
}

void init_input(user::Input& input) {
  user::Option nlon_option("nlon", "-nlon", "--n_lon",
                           "number of longitude points in DFS grid", 20);
  input.add_option(nlon_option);

  user::Option omega_option("omega", "-omg", "--omega",
                            "background rotation rate of the sphere",
                            2 * constants::PI);
  input.add_option(omega_option);

  user::Option tfinal_option("tfinal", "-tf", "--time-final", "time final",
                             0.025);
  input.add_option(tfinal_option);

  user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of time steps",
                             1);
  input.add_option(nsteps_option);

  user::Option dt_option("dt", "-dt", "--time-step", "time step size", 0.025);
  input.add_option(dt_option);

  user::Option use_dt_option("use_dt", "-udt", "--use-dt",
                             "use dt for time step size", true);
  input.add_option(use_dt_option);

  user::Option use_nsteps_option("use_nsteps", "-un", "--use-nsteps",
                                 "use nsteps for time step size", false);
  input.add_option(use_nsteps_option);

  user::Option tree_depth_option("tree_depth", "-d", "--depth",
                                 "mesh tree initial uniform depth", 4);
  input.add_option(tree_depth_option);

  user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer",
                                            "amr memory buffer", 0);
  input.add_option(amr_refinement_buffer_option);

  user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit",
                                           "amr refinement limit", 0);
  input.add_option(amr_refinement_limit_option);

  user::Option max_circulation_option(
      "max_circulation_tol", "-c", "--circuluation-max",
      "amr max circulation tolerance", std::numeric_limits<Real>::max());
  input.add_option(max_circulation_option);

  user::Option amr_both_option("amr_both", "-amr", "--amr-both",
                               "both amr buffer and limit values",
                               LPM_NULL_IDX);
  input.add_option(amr_both_option);

  user::Option output_write_frequency_option("output_write_frequency", "-of",
                                             "--output-frequency",
                                             "output write frequency", 1);
  input.add_option(output_write_frequency_option);

  user::Option output_file_root_option("output_file_root", "-o",
                                       "--output-file-root", "output file root",
                                       std::string("dfs_rh54"));
  input.add_option(output_file_root_option);

  user::Option remesh_interval_option(
      "remesh_interval", "-rm", "--remesh-interval",
      "number of timesteps allowed between remesh interpolations",
      std::numeric_limits<int>::max());
  input.add_option(remesh_interval_option);

  user::Option remesh_strategy_option(
      "remesh_strategy", "-rs", "--remesh-strategy",
      "direct or indirect remeshing strategy", std::string("direct"),
      std::set<std::string>({"direct", "indirect"}));
  input.add_option(remesh_strategy_option);

  user::Option gmls_interpolation_order(
      "gmls_interpolation_order", "-g", "--gmls-order",
      "polynomial order for gmls-based interpolation", 6);
  input.add_option(gmls_interpolation_order);

  user::Option gmls_eps_multiplier(
      "gmls_eps_multiplier", "-eps", "--gmls-eps-multiplier",
      "gmls neighborhood radius multiplier (default 2)", 2.0);
  input.add_option(gmls_eps_multiplier);

  user::Option remesh_trigger_option(
      "remesh_trigger", "-rt", "--remesh-trigger",
      "trigger for a remeshing : ftle or an interval", std::string("interval"),
      std::set<std::string>({"interval", "ftle"}));
  input.add_option(remesh_trigger_option);

  user::Option ftle_tolerance_option("ftle_tol", "-ftle", "--ftle-tol",
                                     "max value for ftle before remesh", 2.0);
  input.add_option(ftle_tolerance_option);
}
