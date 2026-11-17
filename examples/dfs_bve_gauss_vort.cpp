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
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_ftle.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

/** Gaussian vortex on a rotating sphere (barotropic vorticity equation).

  The vortex drifts northwest and radiates Rossby waves that roll up into
  thin vorticity filaments, so the Lagrangian mesh distorts in ways that
  matter for accuracy. There is no exact solution: runs are compared against
  a high-resolution reference through snapshots of the DFS grid vorticity
  (written to the .m file; DFS grids are nested when nlon doubles).

  Intended for studying remeshing (none, fixed interval, or FTLE-triggered)
  and adaptive vs. uniform remeshing. Adaptive refinement criteria, each off
  unless its tolerance is given:
    -c   circulation |zeta| * area, relative to the max on the initial mesh
    -zv  vorticity variation within a panel, relative to the max on the
         initial mesh
    -fs  FTLE (absolute), at remesh only
  The absolute circulation and variation tolerances are fixed from the
  initial uniform mesh and reused at every remesh.

  Time stepping is RK4.
*/

using namespace Lpm;
using namespace Lpm::DFS;

void init_input(user::Input& input);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("dfs_bve_gauss", Log::level::info, comm);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
    /**
      program run
    */
    /**
      initialize problem
    */
    user::Input input("dfs_bve_gauss");
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
    using Lat0                        = LatitudeTracer;
    const Real Omega                  = input.get_option("omega").get_real();
    constexpr Real vortex_strength    = 4 * constants::PI;
    constexpr Real vortex_shape_param = 4.0;
    constexpr Real vortex_init_lon    = constants::PI;
    constexpr Real vortex_init_lat    = constants::PI / 10;
    GaussianVortexSphere gauss_vort(vortex_strength, vortex_shape_param,
                                    vortex_init_lon, vortex_init_lat);
    // exact: the same initial condition at every resolution and refinement;
    // discrete: zero total vorticity on this mesh (the pre-2026-11-17 default)
    const bool exact_gauss_const =
        (input.get_option("gauss_const").get_str() == "exact");
    if (exact_gauss_const) {
      gauss_vort.set_exact_gauss_const();
    }

    //  particle/panel/grid initialization
    using SeedType               = CubedSphereSeed;
    const Int mesh_depth         = input.get_option("tree_depth").get_int();
    constexpr Real sphere_radius = 1;
    Int amr_buffer               = input.get_option("amr_buffer").get_int();
    Int amr_limit                = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit  = input.get_option("amr_both").get_int();
    }
    const bool amr = (amr_buffer > 0 and amr_limit > 0);
    constexpr Real no_tol = std::numeric_limits<Real>::max();
    const Real rel_circ_tol =
        input.get_option("max_circulation_tol").get_real();
    const Real rel_zeta_var_tol =
        input.get_option("vorticity_variation_tol").get_real();
    const Real ftle_space_tol = input.get_option("ftle_space_tol").get_real();
    const bool use_circ_flag  = (rel_circ_tol < no_tol);
    const bool use_var_flag   = (rel_zeta_var_tol < no_tol);
    const bool use_ftle_flag  = (ftle_space_tol < no_tol);
    if (amr) {
      LPM_REQUIRE_MSG(use_circ_flag or use_var_flag or use_ftle_flag,
                      "amr needs at least one of -c, -zv, -fs.");
    }
    /**
      Build the Lagrangian particle/panel mesh and DFS grid
    */
    PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius,
                                             amr_buffer, amr_limit);
    const Int nlon = input.get_option("nlon").get_int();
    const Int gmls_order =
        input.get_option("gmls_interpolation_order").get_int();
    gmls::Params gmls_params(gmls_order);
    gmls_params.eps_multiplier =
        input.get_option("gmls_eps_multiplier").get_real();
    // the remesh may use larger neighborhoods than the mesh-to-grid
    // interpolation done at every RK stage
    gmls::Params remesh_gmls_params(gmls_params);
    remesh_gmls_params.eps_multiplier =
        input.get_option("remesh_eps_multiplier").get_real();
    LPM_REQUIRE_MSG(gmls_params.eps_multiplier > 0,
                    "-eps (" << gmls_params.eps_multiplier
                             << ") must be positive.");
    LPM_REQUIRE_MSG(remesh_gmls_params.eps_multiplier > 0,
                    "-rmeps (" << remesh_gmls_params.eps_multiplier
                               << ") must be positive.");

    // DFS initialization
    auto sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon,
                                                     gmls_params, Omega);

    /**
      initial vorticity
    */
    sphere->init_vorticity(gauss_vort);
    if (not exact_gauss_const) {
      gauss_vort.set_gauss_const(sphere->total_vorticity());
      sphere->init_vorticity(gauss_vort);
    }
    logger.info("gauss_const = {} ({}), total vorticity = {}",
                gauss_vort.gauss_const,
                input.get_option("gauss_const").get_str(),
                sphere->total_vorticity());

    auto rel_vort_range =
        sphere->rel_vort_active.range(sphere->mesh.n_faces_host());
    logger.info(
        "uniform mesh has (min, max) vorticity (active) = ({}, {}), max. circ. "
        "appx {}",
        rel_vort_range.first, rel_vort_range.second,
        rel_vort_range.second * square(sphere->mesh.appx_mesh_size()));

    /**
      Initial adaptive refinement

      Relative tolerances from the input become absolute tolerances on the
      initial uniform mesh; the remesh reuses those absolute values.
    */
    Real circ_tol     = no_tol;
    Real zeta_var_tol = no_tol;
    if (amr) {
      Refinement<SeedType> refiner(sphere->mesh);

      ScalarIntegralFlag circ_flag(
          refiner.flags, sphere->rel_vort_active.view, sphere->mesh.faces.area,
          sphere->mesh.faces.mask, sphere->mesh.n_faces_host(), rel_circ_tol);
      if (use_circ_flag) {
        circ_flag.set_tol_from_relative_value();
        logger.info(
            "relative max. circulation tol {} converts to absolute circulation "
            "tol {}",
            rel_circ_tol, circ_flag.tol);
      }
      circ_tol = circ_flag.tol;

      ScalarVariationFlag var_flag(
          refiner.flags, sphere->rel_vort_active.view,
          sphere->rel_vort_passive.view, sphere->mesh.faces.verts,
          sphere->mesh.faces.mask, sphere->mesh.n_faces_host(),
          rel_zeta_var_tol);
      if (use_var_flag) {
        var_flag.set_tol_from_relative_value();
        logger.info(
            "relative vorticity variation tol {} converts to absolute "
            "variation tol {}",
            rel_zeta_var_tol, var_flag.tol);
      }
      zeta_var_tol = var_flag.tol;

      Index face_start_idx = 0;
      for (int i = 0; i < amr_limit; ++i) {
        const Index face_end_idx = sphere->mesh.n_faces_host();

        refiner.iterate(face_start_idx, face_end_idx, circ_flag, var_flag);
        logger.info("amr iteration {}: {}", i, refiner.info_string());
        if (refiner.total() == 0) {
          break;
        }

        sphere->mesh.divide_flagged_faces(refiner.flags, logger);
        face_start_idx = face_end_idx;
        sphere->init_vorticity(gauss_vort);
      }  // amr iterations
      if (not exact_gauss_const) {
        gauss_vort.update_gauss_const(sphere->total_vorticity());
        sphere->init_vorticity(gauss_vort);
      }

      rel_vort_range =
          sphere->rel_vort_active.range(sphere->mesh.n_faces_host());
      logger.info(
          "adaptively refined mesh has {} leaf faces, (min, max) vorticity "
          "(active) = ({}, {}), total vorticity = {}",
          sphere->mesh.faces.n_leaves_host(), rel_vort_range.first,
          rel_vort_range.second, sphere->total_vorticity());
    }  // if (amr)

    /**
      With vorticity initialized, we can compute the initial velocity
    */
    sphere->finalize_mesh_to_grid_coupling();
    sphere->init_velocity_from_vorticity();

    Lat0 lat0;
    sphere->init_tracer(lat0);
    logger.info(sphere->info_string());

    // Solver initialization
    using SolverType  = DFSRK4<SeedType>;
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
    logger.info("dt = {}, nsteps = {}, cr = {}", dt, nsteps, cr);

    const Int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy =
        input.get_option("remesh_strategy").get_str();
    LPM_REQUIRE_MSG(remesh_strategy == "direct",
                    "only direct remeshing is implemented for DFS.");
    const bool use_ftle =
        (input.get_option("remesh_trigger").get_str() == "ftle");
    const Real ftle_tol = input.get_option("ftle_tol").get_real();

    auto solver = std::make_unique<SolverType>(dt, *sphere);
    std::vector<Real> total_vorticity(nsteps + 1);
    std::vector<Real> total_kinetic_energy(nsteps + 1);
    std::vector<Real> total_enstrophy(nsteps + 1);
    std::vector<Real> ftle_max(nsteps + 1);
    std::vector<Real> n_vertices(nsteps + 1);
    std::vector<Real> n_leaf_faces(nsteps + 1);
    std::vector<Real> time(nsteps + 1);
    std::vector<Real> remesh_times;
    total_vorticity[0]      = sphere->total_vorticity();
    total_kinetic_energy[0] = sphere->total_kinetic_energy();
    total_enstrophy[0]      = sphere->total_enstrophy();
    n_vertices[0]           = sphere->mesh.n_vertices_host();
    n_leaf_faces[0]         = sphere->mesh.faces.n_leaves_host();
    const bool grid_integrals = input.get_option("grid_integrals").get_bool();
    std::vector<Real> total_vorticity_grid(grid_integrals ? nsteps + 1 : 0);
    std::vector<Real> total_kinetic_energy_grid(grid_integrals ? nsteps + 1
                                                               : 0);
    std::vector<Real> total_enstrophy_grid(grid_integrals ? nsteps + 1 : 0);
    if (grid_integrals) {
      sphere->update_grid_velocity();
      total_vorticity_grid[0]      = sphere->total_vorticity_grid();
      total_kinetic_energy_grid[0] = sphere->total_kinetic_energy_grid();
      total_enstrophy_grid[0]      = sphere->total_enstrophy_grid();
      logger.info(
          "initial grid total vorticity, kinetic energy, and enstrophy = ({}, "
          "{}, {})",
          total_vorticity_grid[0], total_kinetic_energy_grid[0],
          total_enstrophy_grid[0]);
    }

    logger.info(
        "initial total vorticity, kinetic energy, and enstrophy = ({}, {}, {})",
        total_vorticity[0], total_kinetic_energy[0], total_enstrophy[0]);

    /**
      Grid vorticity snapshots, for comparison against a reference run:
      zeta_grid holds the nlat x nlon grid (row i = colatitude i) of each
      snapshot in turn.
    */
    const Int grid_output_frequency =
        input.get_option("grid_output_frequency").get_int();
    std::vector<Real> zeta_grid_times;
    std::vector<Real> zeta_grid;
    const auto record_zeta_grid = [&](const Real t) {
      sphere->rel_vort_grid.update_host();
      for (Index i = 0; i < sphere->grid.size(); ++i) {
        zeta_grid.push_back(sphere->rel_vort_grid.hview(i));
      }
      zeta_grid_times.push_back(t);
    };
    sphere->interpolate_vorticity_from_mesh_to_grid();
    record_zeta_grid(0);

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
    if (remesh_str != "no_rm") {
      LPM_REQUIRE_MSG(
          remesh_gmls_params.eps_multiplier >= gmls_params.eps_multiplier,
          "-rmeps (" << remesh_gmls_params.eps_multiplier
                     << ") must be >= -eps (" << gmls_params.eps_multiplier
                     << ").");
    }
    if (remesh_str != "no_rm" and
        remesh_gmls_params.eps_multiplier != gmls_params.eps_multiplier) {
      remesh_str +=
          "_rmeps" + float_str(remesh_gmls_params.eps_multiplier, 3);
    }
    const std::string amr_str =
        (amr ? "_amr" + std::to_string(amr_limit) : std::string());
    const std::string ofile_root =
        input.get_option("output_file_root").get_str() + "_" +
        SeedType::id_string() + resolution_str + "_nlon" +
        std::to_string(nlon) + "_" + remesh_str + amr_str + "_";
    const std::string vtk_file_root = ofile_root;
    logger.info("output will be written to files beginning: {}", ofile_root);

    int vtk_counter = 0;
    const int write_frequency =
        input.get_option("output_write_frequency").get_int();
#ifdef LPM_USE_VTK
    if (write_frequency > 0) {
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
#endif
    /**
    time stepping
    */
    int rm_counter = 0;
    Real tref      = 0;
    Real max_ftle  = 0;
    for (int t_idx = 0; t_idx < nsteps; ++t_idx) {
      /**
        check: remesh before this step?
        max_ftle is the deformation accumulated since the last remesh.
      */
      const bool ftle_trigger = (use_ftle and max_ftle > ftle_tol);
      const bool interval_trigger =
          (t_idx > 0 and t_idx % remesh_interval == 0);
      const bool do_remesh = (ftle_trigger or interval_trigger);
      if (do_remesh) {
        ++rm_counter;
        if (interval_trigger) {
          logger.debug("remesh {} triggered by remesh interval at t = {}",
                       rm_counter, sphere->t);
        } else {
          logger.info("remesh {} triggered by ftle = {} at t = {}",
                      rm_counter, max_ftle, sphere->t);
        }

        auto new_sphere = std::make_unique<DFSBVE<SeedType>>(
            mesh_params, nlon, gmls_params, Omega);
        // the remesh interpolates every tracer carried by the old sphere, so
        // the new sphere must allocate all of them
        for (const auto& tracer : sphere->tracer_passive) {
          new_sphere->allocate_tracer(tracer.first);
        }
        new_sphere->t = sphere->t;

        auto remesh =
            compadre_dfs_remesh(*new_sphere, *sphere, remesh_gmls_params);
        if (amr) {
          Refinement<SeedType> refiner(new_sphere->mesh);

          ScalarIntegralFlag circ_flag(
              refiner.flags, new_sphere->rel_vort_active.view,
              new_sphere->mesh.faces.area, new_sphere->mesh.faces.mask,
              new_sphere->mesh.n_faces_host(), rel_circ_tol);
          circ_flag.tol = circ_tol;

          ScalarVariationFlag var_flag(
              refiner.flags, new_sphere->rel_vort_active.view,
              new_sphere->rel_vort_passive.view, new_sphere->mesh.faces.verts,
              new_sphere->mesh.faces.mask, new_sphere->mesh.n_faces_host(),
              rel_zeta_var_tol);
          var_flag.tol = zeta_var_tol;

          ScalarMaxFlag ftle_flag(
              refiner.flags, new_sphere->ftle_active.view,
              new_sphere->mesh.faces.mask, new_sphere->mesh.n_faces_host(),
              ftle_space_tol);

          remesh.adaptive_direct_remesh(refiner, circ_flag, var_flag,
                                        ftle_flag);
        } else {
          remesh.uniform_direct_remesh();
        }
        logger.info(remesh.info_string());

        tref   = sphere->t;
        sphere = std::move(new_sphere);

        sphere->finalize_mesh_to_grid_coupling();
        // the first RK stage reads the gathered velocity; solve it from the
        // remeshed vorticity, as at the end of every step, instead of using
        // the velocity interpolated from the old particles
        sphere->interpolate_vorticity_from_mesh_to_grid();
        sphere->init_velocity_from_vorticity();
        sphere->t_ref = tref;
        solver.reset(new SolverType(dt, *sphere, solver->t_idx));
        sphere->reset_ftle();
        remesh_times.push_back(tref);
        logger.info("remesh {} at t = {}: {} vertices, {} leaf faces",
                    rm_counter, tref, sphere->mesh.n_vertices_host(),
                    sphere->mesh.faces.n_leaves_host());
      }

      /**
        step forward
      */
      sphere->advance_timestep(*solver);
      max_ftle = get_max_ftle(sphere->ftle_active.view, sphere->mesh.faces.mask,
                              sphere->mesh.n_faces_host());

      time[t_idx + 1]                 = (t_idx + 1) * dt;
      ftle_max[t_idx + 1]             = max_ftle;
      total_vorticity[t_idx + 1]      = sphere->total_vorticity();
      total_kinetic_energy[t_idx + 1] = sphere->total_kinetic_energy();
      total_enstrophy[t_idx + 1]      = sphere->total_enstrophy();
      n_vertices[t_idx + 1]           = sphere->mesh.n_vertices_host();
      n_leaf_faces[t_idx + 1]         = sphere->mesh.faces.n_leaves_host();
      if (grid_integrals) {
        sphere->update_grid_velocity();
        total_vorticity_grid[t_idx + 1] = sphere->total_vorticity_grid();
        total_kinetic_energy_grid[t_idx + 1] =
            sphere->total_kinetic_energy_grid();
        total_enstrophy_grid[t_idx + 1] = sphere->total_enstrophy_grid();
      }
      // the solver leaves rel_vort_grid interpolated from the particles'
      // positions at the end of the step
      if (t_idx + 1 == nsteps or (grid_output_frequency > 0 and
                                  (t_idx + 1) % grid_output_frequency == 0)) {
        record_zeta_grid((t_idx + 1) * dt);
      }

#ifdef LPM_USE_VTK
      if (write_frequency > 0 and (t_idx + 1) % write_frequency == 0) {
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
#endif
      const auto rel_vort_range =
          sphere->rel_vort_passive.range(sphere->mesh.vertices.nh());
      logger.info(
          "t = {}, max. ftle = {}, rel. vort passive range : ({}, {})",
          (t_idx + 1) * dt, max_ftle, rel_vort_range.first,
          rel_vort_range.second);
    }
    logger.info("{} remeshing steps over {} time steps.", rm_counter, nsteps);

    const std::string matlab_file = ofile_root + ".m";
    std::ofstream ofile(matlab_file);
    ofile << std::setprecision(17);
    write_vector_matlab(ofile, "time", time);
    write_vector_matlab(ofile, "total_vorticity", total_vorticity);
    write_vector_matlab(ofile, "total_kinetic_energy", total_kinetic_energy);
    write_vector_matlab(ofile, "total_enstrophy", total_enstrophy);
    if (grid_integrals) {
      write_vector_matlab(ofile, "total_vorticity_grid", total_vorticity_grid);
      write_vector_matlab(ofile, "total_kinetic_energy_grid",
                          total_kinetic_energy_grid);
      write_vector_matlab(ofile, "total_enstrophy_grid", total_enstrophy_grid);
    }
    write_vector_matlab(ofile, "max_ftle", ftle_max);
    write_vector_matlab(ofile, "remesh_times", remesh_times);
    write_vector_matlab(ofile, "n_vertices", n_vertices);
    write_vector_matlab(ofile, "n_leaf_faces", n_leaf_faces);
    write_vector_matlab(ofile, "grid_nlat",
                        std::vector<Real>(1, sphere->grid.nlat));
    write_vector_matlab(ofile, "grid_nlon",
                        std::vector<Real>(1, sphere->grid.nlon));
    write_vector_matlab(ofile, "zeta_grid_times", zeta_grid_times);
    write_vector_matlab(ofile, "zeta_grid", zeta_grid);
    ofile.close();
    logger.info("output can be found in files beginning: {}", ofile_root);
  }  // Kokkos scope
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
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

  user::Option amr_both_option("amr_both", "-amr", "--amr-both",
                               "both amr buffer and limit values",
                               LPM_NULL_IDX);
  input.add_option(amr_both_option);

  user::Option max_circulation_option(
      "max_circulation_tol", "-c", "--circuluation-max",
      "amr max circulation tolerance, relative to the initial mesh (default: "
      "off)",
      std::numeric_limits<Real>::max());
  input.add_option(max_circulation_option);

  user::Option vorticity_variation_option(
      "vorticity_variation_tol", "-zv", "--vorticity-variation",
      "amr vorticity variation tolerance, relative to the initial mesh "
      "(default: off)",
      std::numeric_limits<Real>::max());
  input.add_option(vorticity_variation_option);

  user::Option ftle_space_tolerance_option(
      "ftle_space_tol", "-fs", "--ftle-space-tol",
      "amr ftle tolerance, applied at remesh (default: off)",
      std::numeric_limits<Real>::max());
  input.add_option(ftle_space_tolerance_option);

  user::Option gauss_const_option(
      "gauss_const", "-gc", "--gauss-const",
      "gaussian vortex offset: exact (zero exact total vorticity) or discrete "
      "(zero total vorticity on the initial mesh)",
      std::string("exact"), std::set<std::string>({"exact", "discrete"}));
  input.add_option(gauss_const_option);

  user::Option output_write_frequency_option(
      "output_write_frequency", "-of", "--output-frequency",
      "vtk output write frequency (0 disables vtk output)", 1);
  input.add_option(output_write_frequency_option);

  user::Option grid_output_frequency_option(
      "grid_output_frequency", "-gf", "--grid-output-frequency",
      "steps between grid vorticity snapshots in the .m file (0: initial and "
      "final only)",
      0);
  input.add_option(grid_output_frequency_option);

  user::Option output_file_root_option("output_file_root", "-o",
                                       "--output-file-root", "output file root",
                                       std::string("dfs_gauss"));
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

  user::Option remesh_eps_multiplier(
      "remesh_eps_multiplier", "-rmeps", "--remesh-eps-multiplier",
      "gmls neighborhood radius multiplier for remeshing (default 2)", 2.0);
  input.add_option(remesh_eps_multiplier);

  user::Option remesh_trigger_option(
      "remesh_trigger", "-rt", "--remesh-trigger",
      "trigger for a remeshing : ftle or an interval", std::string("interval"),
      std::set<std::string>({"interval", "ftle"}));
  input.add_option(remesh_trigger_option);

  user::Option ftle_tolerance_option("ftle_tol", "-ftle", "--ftle-tol",
                                     "max value for ftle before remesh", 2.0);
  input.add_option(ftle_tolerance_option);

  user::Option grid_integrals_option(
      "grid_integrals", "-gi", "--grid-integrals",
      "also compute conservation integrals on the DFS grid", false);
  input.add_option(grid_integrals_option);
}
