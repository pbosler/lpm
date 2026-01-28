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
#include "dfs/lpm_dfs_polar_vortex_solver.hpp"
#include "dfs/lpm_dfs_polar_vortex_solver_impl.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_ftle.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include <cstdio>
#include <iomanip>
#include <iostream>

using namespace Lpm;
using namespace Lpm::DFS;

void init_input(user::Input& input);
void init_amr(bool& do_amr, int& amr_limit, int& amr_buffer, const user::Input& input);
template <typename SeedType, typename VorticityType, typename LoggerType>
void initial_refinement(DFSBVE<SeedType>& sphere, Real& circ_tol, const VorticityType& vorticity, const user::Input& input, LoggerType& logger);

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
  /**
    Initialize the problem
  */
  using SeedType = CubedSphereSeed;
  constexpr Real sphere_radius = 1.0;

  bool amr = false;
  int amr_limit = 0;
  int amr_buffer = 0;
  init_amr(amr, amr_limit, amr_buffer, input);

  const Int mesh_depth = input.get_option("tree_depth").get_int();
  PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius, amr_buffer, amr_limit);
  const Int nlon = input.get_option("nlon").get_int();
  const Int gmls_order = input.get_option("gmls_interpolation_order").get_int();
  gmls::Params gmls_params(gmls_order);
  const Real Omega = input.get_option("omega").get_real();

  auto sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params, Omega);

  /**
      initialize vorticity, step 1 of 2
  */
  const Real zeta0_max = input.get_option("vortex_strength").get_real();
  const Real zeta0_b = input.get_option("vortex_shape_parameter").get_real();
  const Real tfull = input.get_option("forcing_tfull").get_real();
  const Real tend = input.get_option("forcing_tend").get_real();
  const Real F0 = input.get_option("forcing_F0").get_real();
  const PolarVortexParams pv_params(zeta0_max, zeta0_b, tfull, tend, F0);

  JM86PolarVortex vorticity(pv_params);
  sphere->init_vorticity(vorticity);
  const Real total_vort_0 = sphere->total_vorticity();
  /**
      initialize vorticity, step 1 of 2
  */
  vorticity.set_gauss_const(total_vort_0);
  sphere->init_vorticity(vorticity);
  auto rel_vort_range = sphere->rel_vort_active.range(sphere->mesh.n_faces_host());
  logger.info("uniform mesh has active vorticity (min, max) = ({}, {}), for max. circulation of appx. {}, gauss.const = {}",
    rel_vort_range.first, rel_vort_range.second, rel_vort_range.second * sphere->mesh.avg_face_area(), vorticity.gauss_const);
  rel_vort_range = sphere->rel_vort_passive.range(sphere->mesh.n_faces_host());
  logger.debug("uniform mesh has passive vorticity (min, max) = ({}, {})",
    rel_vort_range.first, rel_vort_range.second);

  /**
      initialize refinement
  */
  Real circ_tol = std::numeric_limits<Real>::max();
  if (amr) {
    initial_refinement(*sphere, circ_tol, vorticity, input, logger);
    logger.info("relative max. circulation tol (from input, {}) converts to absolute circ_tol = {}",
      input.get_option("max_circulation_tol").get_real(), circ_tol);
  }
  sphere->finalize_mesh_to_grid_coupling();
  sphere->init_velocity_from_vorticity();
  logger.info(sphere->info_string());

  /**
    initialize time stepping
  */
  using SolverType = DFSPolarVortexRK4<SeedType>;
  TimestepParams timestep_params(input);
  const Real dt = timestep_params.dt;
  const Real tfinal = timestep_params.tfinal;
  const Int nsteps = timestep_params.nsteps;
  auto solver = std::make_unique<SolverType>(dt, *sphere, 0, pv_params);
  auto vel_range = sphere->velocity_active.range(sphere->mesh.n_faces_host());
  logger.info("timestepping ready: dt = {}, nsteps = {}, tfinal = {}, initial courant number = {}",
    dt, nsteps, tfinal, timestep_params.courant_number(sphere->mesh.appx_min_mesh_size(), vel_range.second));
  const Real ftle_tol = input.get_option("ftle_tol").get_real();
  const bool use_ftle = (input.get_option("remesh_trigger").get_str() == "ftle" and
    ftle_tol < 100);
  const Real ftle_space_tol = input.get_option("ftle_space_tol").get_real();
  const Int remesh_interval = input.get_option("remesh_interval").get_int();

  std::vector<Real> total_vorticity(nsteps+1);
  std::vector<Real> total_kinetic_energy(nsteps+1);
  std::vector<Real> total_enstrophy(nsteps+1);
  std::vector<Real> ftle_max(nsteps+1);
  std::vector<Real> time(nsteps+1);
  ftle_max[0] = 0;
  total_vorticity[0] = sphere->total_vorticity();
  total_kinetic_energy[0] = sphere->total_kinetic_energy();
  total_enstrophy[0] = sphere->total_enstrophy();

  /**
    problem initialization complete: write initial data
  */
  std::string resolution_str = SeedType::id_string() + std::to_string(mesh_depth);
  if (amr) {
    std::ostringstream ss;
    ss << "+amr" << amr_limit << "_circtol" << std::setprecision(6) << circ_tol;
    resolution_str += ss.str();
  }
  resolution_str += "_nlon" + std::to_string(nlon) + "_" + timestep_params.filename_piece();
  std::string remesh_str;
  if (use_ftle) {
    remesh_str = "ftle" + float_str(ftle_tol,3) + "ftle_sp" + float_str(ftle_space_tol,3);
  }
  else {
    remesh_str = (remesh_interval < nsteps ? "rm" + std::to_string(remesh_interval) : "no_rm");
  }
  const std::string ofile_root = input.get_option("output_file_root").get_str() + "_" +
    resolution_str + "_" + remesh_str + "_";
  int vtk_counter = 0;
  const int write_frequency = input.get_option("output_write_frequency").get_int();
  logger.info("output will be written to files beginning: {}", ofile_root);
  {
    auto vtk_mesh = vtk_mesh_interface(*sphere);
    auto vtk_grid = vtk_grid_interface(*sphere);
    const std::string mesh_vtk_file = ofile_root + zero_fill_str(vtk_counter) + vtp_suffix();
    const std::string grid_vtk_file = ofile_root + zero_fill_str(vtk_counter) + vts_suffix();
    vtk_mesh.write(mesh_vtk_file);
    vtk_grid.write(grid_vtk_file);
    ++vtk_counter;
  }

  /**
    Time stepping start
  */
  int rm_counter = 0;
  Real tref = 0;
  Real max_ftle = 0;
  logger.debug("starting timestep loop.");
  for (int t_idx = 0; t_idx <nsteps; ++t_idx) {

    /**
      check: remesh?
    */
    max_ftle = get_max_ftle(sphere->ftle_active.view, sphere->mesh.faces.mask, sphere->mesh.n_faces_host());
    const bool ftle_trigger = (use_ftle and max_ftle > ftle_tol);
    const bool interval_trigger = ((t_idx+1)%remesh_interval == 0);
    const bool do_remesh = (ftle_trigger or interval_trigger);
    if (do_remesh) {
      /**
        do remesh before time step
      */
      ++rm_counter;
      if (interval_trigger) {
        logger.debug("remesh {} triggered by remesh interval", rm_counter);
      }
      else {
        logger.info("remesh {} triggered by ftle", rm_counter);
      }

      auto new_sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
      new_sphere->t = sphere->t;

      auto remesh = compadre_dfs_remesh(*new_sphere, *sphere, gmls_params);
      if (amr) {
        Refinement<SeedType> refiner(new_sphere->mesh);

        ScalarIntegralFlag circ_flag(refiner.flags,
          new_sphere->rel_vort_active.view,
          new_sphere->mesh.faces.area,
          new_sphere->mesh.faces.mask,
          new_sphere->mesh.n_faces_host(),
          circ_tol);

        ScalarMaxFlag ftle_flag(refiner.flags,
          new_sphere->ftle_active.view,
          new_sphere->mesh.faces.mask,
          new_sphere->mesh.n_faces_host(),
          ftle_space_tol);

        remesh.adaptive_direct_remesh(refiner, circ_flag, ftle_flag);
      }
      else {
        remesh.uniform_direct_remesh();
      }
      logger.info(remesh.info_string());

      tref = sphere->t;
      sphere = std::move(new_sphere);
      sphere->finalize_mesh_to_grid_coupling();
      sphere->t_ref = tref;
      solver.reset(new SolverType(dt, *sphere, t_idx, pv_params));
      sphere->reset_ftle();
    }

    /**
      step forward
    */
    logger.debug("step forward");
    sphere->advance_timestep(*solver);
    logger.debug("step completed.");

    time[t_idx+1] = (t_idx+1) * dt;
    ftle_max[t_idx+1] = max_ftle;
    total_vorticity[t_idx+1] = sphere->total_vorticity();
    total_kinetic_energy[t_idx+1] = sphere->total_kinetic_energy();
    total_enstrophy[t_idx+1] = sphere->total_enstrophy();

    if ((t_idx+1)%write_frequency == 0) {
      auto vtk_mesh = vtk_mesh_interface(*sphere);
      auto vtk_grid = vtk_grid_interface(*sphere);
      const std::string mesh_vtk_file = ofile_root + zero_fill_str(vtk_counter) + vtp_suffix();
      const std::string grid_vtk_file = ofile_root + zero_fill_str(vtk_counter) + vts_suffix();
      vtk_mesh.write(mesh_vtk_file);
      vtk_grid.write(grid_vtk_file);
      ++vtk_counter;

      rel_vort_range = sphere->rel_vort_passive.range(sphere->mesh.n_vertices_host());
      logger.info("t = {:4.2f}, max. ftle = {:4.2f}, passive rel. vort. range = ({:4.2f}, {:4.2f})",
        (t_idx+1)*dt, max_ftle, rel_vort_range.first, rel_vort_range.second);
    }
  } // time stepping loop

  const std::string matlab_file = ofile_root + ".m";
  std::ofstream ofile(matlab_file);
  write_vector_matlab(ofile, "time", time);
  write_vector_matlab(ofile, "total_vorticity", total_vorticity);
  write_vector_matlab(ofile, "total_kinetic_energy", total_kinetic_energy);
  write_vector_matlab(ofile, "total_enstrophy", total_enstrophy);
  write_vector_matlab(ofile, "max_ftle", ftle_max);

  logger.info("output was written to files beginning: {}", ofile_root);

  } // Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
return 0;
} // main

void init_amr(bool& do_amr, int& amr_limit, int& amr_buffer, const user::Input& input) {
  amr_limit = input.get_option("amr_limit").get_int();
  amr_buffer = input.get_option("amr_buffer").get_int();
  if (input.get_option("amr_both").get_int() > 0) {
    amr_limit = input.get_option("amr_both").get_int();
    amr_buffer = amr_limit;
  }
  do_amr = (amr_buffer > 0  and amr_limit > 0);
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

  user::Option gmls_interpolation_order("gmls_interpolation_order", "-g", "--gmls-order", "polynomial order for gmls-based interpolation", 6);
  input.add_option(gmls_interpolation_order);

  user::Option remesh_trigger_option("remesh_trigger", "-rt", "--remesh-trigger", "trigger for a remeshing : ftle or an interval", std::string("interval"), std::set<std::string>({"interval", "ftle"}));
  input.add_option(remesh_trigger_option);

  user::Option ftle_tolerance_option("ftle_tol", "-ft", "--ftle-tol", "max value for ftle before remesh", std::numeric_limits<Real>::max());
  input.add_option(ftle_tolerance_option);

  user::Option ftle_space_tolerance_option("ftle_space_tol", "-fs", "--ftle-space-tol", "spatial amr ftle tolerance remesh", std::numeric_limits<Real>::max());
  input.add_option(ftle_space_tolerance_option);

  user::Option vortex_strength_option("vortex_strength", "-vs", "--vortex-strength", "maximum strength of initial relative vorticity", 4*constants::PI);
  input.add_option(vortex_strength_option);

  user::Option vortex_shape_option("vortex_shape_parameter", "-vb", "--vortex-b", "shape parameter for initial vorticity", 2.0);
  input.add_option(vortex_shape_option);

  user::Option wave_forcing_strength_option("forcing_F0", "-F", "--forcing-F0", "maximum strength of forcing", 6*constants::PI/5);
  input.add_option(wave_forcing_strength_option);

  user::Option wave_forcing_tfull_option("forcing_tfull", "-tfull", "--forcing-tfull", "time to reach full strength forcing", 4.0);
  input.add_option(wave_forcing_tfull_option);
  user::Option wave_forcing_tend_option("forcing_tend", "-tend", "--forcing-tend", "end time for forcing", 15.0);
  input.add_option(wave_forcing_tend_option);
}

template <typename SeedType, typename VorticityType, typename LoggerType>
void initial_refinement(DFSBVE<SeedType>& sphere, Real& circ_tol, const VorticityType& vorticity, const user::Input& input, LoggerType& logger) {
  Refinement<SeedType> refiner(sphere.mesh);

  const Real rel_circ_tol = input.get_option("max_circulation_tol").get_real();
  ScalarIntegralFlag circ_flag(refiner.flags,
    sphere.rel_vort_active.view,
    sphere.mesh.faces.area,
    sphere.mesh.faces.mask,
    sphere.mesh.n_faces_host(),
    rel_circ_tol);

  circ_flag.set_tol_from_relative_value();
  circ_tol = circ_flag.tol;

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<sphere.mesh.params.amr_limit; ++i) {
    Index face_end_idx = sphere.mesh.n_faces_host();
    Index vert_end_idx = sphere.mesh.n_vertices_host();

    refiner.iterate(face_start_idx, face_end_idx, circ_flag);
    logger.info("initial amr iteration {}: circulation refinement count = {}", i, refiner.count[0]);

    if (refiner.count[0] > 0) {
      sphere.mesh.divide_flagged_faces(refiner.flags, logger);

      Kokkos::deep_copy(refiner.flags, false);
      face_start_idx = face_end_idx;
      vert_start_idx = vert_end_idx;

      sphere.init_vorticity_from_lag_crds(vorticity, vert_start_idx, face_start_idx);
    }
    else {
      break;
    }
  }
}

