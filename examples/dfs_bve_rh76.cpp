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

/** Rossby-Haurwitz wave RH76 (spherical harmonic Y_7^6 plus a solid-body
  background flow) on a rotating sphere.

  The background flow is set so that the wave is stationary; the exact
  solution is the initial condition, so vorticity error is computed against
  the initial vorticity at each particle's current position.

  Intended for studying the effect of remeshing (none, fixed interval, or
  FTLE-triggered) on vorticity error and on the drift of the conserved
  kinetic energy and enstrophy integrals.  Time stepping is RK4.
*/

using namespace Lpm;
using namespace Lpm::DFS;

void init_input(user::Input& input);

template <typename SeedType>
void set_exact_vorticity(DFSBVE<SeedType>& sphere,
                         const RossbyHaurwitzR& vorticity);

template <typename SeedType>
ErrNorms compute_vorticity_error(DFSBVE<SeedType>& sphere,
                                 const RossbyHaurwitzR& vorticity);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("dfs_bve_rh76", Log::level::info, comm);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
    /**
      program run
    */
    /**
      initialize problem
    */
    user::Input input("dfs_bve_rh76");
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
    using Lat0         = LatitudeTracer;
    const Real Omega   = input.get_option("omega").get_real();
    constexpr Int R    = 6;
    const Real rh_amp  = input.get_option("wave_amplitude").get_real();
    RossbyHaurwitzR vorticity(0, rh_amp, R);
    vorticity.set_stationary_wave_speed(Omega);
    RossbyHaurwitzRVelocity velocity(vorticity);
    logger.info("stationary RH76 wave: u0 = {}, amplitude = {}, Omega = {}",
                vorticity.u0, vorticity.amplitude, Omega);

    //  particle/panel/grid initialization
    using SeedType               = CubedSphereSeed;
    const Int mesh_depth         = input.get_option("tree_depth").get_int();
    constexpr Real sphere_radius = 1;
    PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius);
    const Int nlon = input.get_option("nlon").get_int();
    const Int gmls_order =
        input.get_option("gmls_interpolation_order").get_int();
    gmls::Params gmls_params(gmls_order);
    gmls_params.eps_multiplier =
        input.get_option("gmls_eps_multiplier").get_real();

    // DFS initialization
    auto sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon,
                                                     gmls_params, Omega);
    const Real ftle_tol = input.get_option("ftle_tol").get_real();
    sphere->init_vorticity(vorticity);
    sphere->finalize_mesh_to_grid_coupling();
    sphere->init_velocity(velocity);
    Lat0 lat0;
    sphere->init_tracer(lat0);
    sphere->init_tracer(vorticity, "exact_vorticity");
    sphere->allocate_tracer("vorticity_error");
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

    auto solver = std::make_unique<SolverType>(dt, *sphere);
    std::vector<Real> zeta_l1err(nsteps + 1);
    std::vector<Real> zeta_l2err(nsteps + 1);
    std::vector<Real> zeta_linferr(nsteps + 1);
    std::vector<Real> total_vorticity(nsteps + 1);
    std::vector<Real> total_kinetic_energy(nsteps + 1);
    std::vector<Real> total_enstrophy(nsteps + 1);
    std::vector<Real> ftle_max(nsteps + 1);
    std::vector<Real> time(nsteps + 1);
    std::vector<Real> remesh_times;
    {
      const auto zeta_error = compute_vorticity_error(*sphere, vorticity);
      zeta_l1err[0]   = zeta_error.l1;
      zeta_l2err[0]   = zeta_error.l2;
      zeta_linferr[0] = zeta_error.linf;
    }
    total_vorticity[0]      = sphere->total_vorticity();
    total_kinetic_energy[0] = sphere->total_kinetic_energy();
    total_enstrophy[0]      = sphere->total_enstrophy();
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
        std::to_string(nlon) + "_" + remesh_str + "_";
    const std::string vtk_file_root = ofile_root;
    logger.info("output will be written to files beginning: {}", ofile_root);

    int vtk_counter = 0;
    const int write_frequency =
        input.get_option("output_write_frequency").get_int();
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

        auto remesh = compadre_dfs_remesh(*new_sphere, *sphere, gmls_params);
        remesh.uniform_direct_remesh();
        logger.debug(remesh.info_string());

        tref   = sphere->t;
        sphere = std::move(new_sphere);

        sphere->finalize_mesh_to_grid_coupling();
        sphere->t_ref = tref;
        solver.reset(new SolverType(dt, *sphere, solver->t_idx));
        sphere->reset_ftle();
        remesh_times.push_back(tref);
      }

      /**
        step forward
      */
      sphere->advance_timestep(*solver);
      max_ftle = get_max_ftle(sphere->ftle_active.view, sphere->mesh.faces.mask,
                              sphere->mesh.n_faces_host());
      const auto zeta_error = compute_vorticity_error(*sphere, vorticity);

      time[t_idx + 1]                 = (t_idx + 1) * dt;
      zeta_l1err[t_idx + 1]           = zeta_error.l1;
      zeta_l2err[t_idx + 1]           = zeta_error.l2;
      zeta_linferr[t_idx + 1]         = zeta_error.linf;
      ftle_max[t_idx + 1]             = max_ftle;
      total_vorticity[t_idx + 1]      = sphere->total_vorticity();
      total_kinetic_energy[t_idx + 1] = sphere->total_kinetic_energy();
      total_enstrophy[t_idx + 1]      = sphere->total_enstrophy();
      if (grid_integrals) {
        sphere->update_grid_velocity();
        total_vorticity_grid[t_idx + 1] = sphere->total_vorticity_grid();
        total_kinetic_energy_grid[t_idx + 1] =
            sphere->total_kinetic_energy_grid();
        total_enstrophy_grid[t_idx + 1] = sphere->total_enstrophy_grid();
      }

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
      const auto rel_vort_range =
          sphere->rel_vort_passive.range(sphere->mesh.vertices.nh());
      logger.info(
          "t = {}, max. ftle = {}, l2 vort. err = {}, rel. vort passive range "
          ": ({}, {})",
          (t_idx + 1) * dt, max_ftle, zeta_error.l2, rel_vort_range.first,
          rel_vort_range.second);
    }
    logger.info("{} remeshing steps over {} time steps.", rm_counter, nsteps);

    const std::string matlab_file = ofile_root + ".m";
    std::ofstream ofile(matlab_file);
    ofile << std::setprecision(17);
    write_vector_matlab(ofile, "time", time);
    write_vector_matlab(ofile, "zeta_l1err", zeta_l1err);
    write_vector_matlab(ofile, "zeta_l2err", zeta_l2err);
    write_vector_matlab(ofile, "zeta_linferr", zeta_linferr);
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
    ofile.close();
    logger.info("output can be found in files beginning: {}", ofile_root);
  }  // Kokkos scope
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
}

template <typename SeedType>
void set_exact_vorticity(DFSBVE<SeedType>& sphere,
                         const RossbyHaurwitzR& vorticity) {
  LPM_REQUIRE_MSG(vorticity.is_stationary(sphere.Omega()),
                  "Vorticity error assumes stationary RH wave");

  const auto vert_crds       = sphere.mesh.vertices.phys_crds.view;
  const auto vert_exact_zeta = sphere.tracer_passive.at("exact_vorticity").view;
  const auto face_crds       = sphere.mesh.faces.phys_crds.view;
  const auto face_exact_zeta = sphere.tracer_active.at("exact_vorticity").view;
  // non-leaf (divided) panels are not part of the solution; zero them so they
  // do not enter the error norms
  const auto face_mask = sphere.mesh.faces.mask;

  Kokkos::parallel_for(
      sphere.mesh.n_vertices_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd    = Kokkos::subview(vert_crds, i, Kokkos::ALL);
        vert_exact_zeta(i) = vorticity(mcrd);
      });
  Kokkos::parallel_for(
      sphere.mesh.n_faces_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd    = Kokkos::subview(face_crds, i, Kokkos::ALL);
        face_exact_zeta(i) = (face_mask(i) ? 0 : vorticity(mcrd));
      });
}

/** Sets the exact vorticity and vorticity error tracers at the current
  particle positions and returns the error norms over the active panels.
*/
template <typename SeedType>
ErrNorms compute_vorticity_error(DFSBVE<SeedType>& sphere,
                                 const RossbyHaurwitzR& vorticity) {
  set_exact_vorticity(sphere, vorticity);
  compute_error(sphere.tracer_passive.at("vorticity_error").view,
                sphere.rel_vort_passive.view,
                sphere.tracer_passive.at("exact_vorticity").view);
  compute_error(sphere.tracer_active.at("vorticity_error").view,
                sphere.rel_vort_active.view,
                sphere.tracer_active.at("exact_vorticity").view,
                sphere.mesh.faces.mask);
  const auto zeta_error_scalars =
      reduce_error(sphere.tracer_active.at("vorticity_error").view,
                   sphere.tracer_active.at("exact_vorticity").view,
                   sphere.mesh.faces.area);
  return ErrNorms(zeta_error_scalars);
}

void init_input(user::Input& input) {
  user::Option nlon_option("nlon", "-nlon", "--n_lon",
                           "number of longitude points in DFS grid", 20);
  input.add_option(nlon_option);

  user::Option omega_option("omega", "-omg", "--omega",
                            "background rotation rate of the sphere",
                            2 * constants::PI);
  input.add_option(omega_option);

  user::Option wave_amplitude_option("wave_amplitude", "-amp",
                                     "--wave-amplitude",
                                     "amplitude of the RH76 wave", 1.0);
  input.add_option(wave_amplitude_option);

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

  user::Option output_write_frequency_option(
      "output_write_frequency", "-of", "--output-frequency",
      "vtk output write frequency (0 disables vtk output)", 1);
  input.add_option(output_write_frequency_option);

  user::Option output_file_root_option("output_file_root", "-o",
                                       "--output-file-root", "output file root",
                                       std::string("dfs_rh76"));
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

  user::Option grid_integrals_option(
      "grid_integrals", "-gi", "--grid-integrals",
      "also compute conservation integrals on the DFS grid", false);
  input.add_option(grid_integrals_option);
}
