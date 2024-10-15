#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
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
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"
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

/** Computes vorticity error at each particle and panel.
*/
template <typename SeedType>
void compute_vorticity_error(scalar_view_type vert_err, scalar_view_type face_err,
  const DFS::DFSBVE<SeedType>& sph) {
  Kokkos::parallel_for("relative vorticity error (vertices)",
    sph.mesh.n_vertices_host(),
    RelVortError(vert_err,
      sph.abs_vort_passive.view, sph.rel_vort_passive.view,
      sph.mesh.vertices.phys_crds.view, sph.Omega));
  Kokkos::parallel_for("relative vorticity error (faces)",
    sph.mesh.n_faces_host(),
    RelVortError(face_err,
      sph.abs_vort_active.view, sph.rel_vort_active.view,
      sph.mesh.faces.phys_crds.view, sph.Omega));
}

/** Compute kernel for relative vorticity error.
*/
struct RelVortError {
  scalar_view_type rel_vort_error; /// relative vorticity error [output]
  scalar_view_type abs_vort; /// absolute vorticity [input]
  scalar_view_type rel_vort; /// computed relative vorticity [input]
  typename SphereGeometry::crd_view_type phys_crds_view; /// particle positions [input]
  Real Omega;

  /// constructor
  RelVortError(scalar_view_type err, const scalar_view_type omega,
    const scalar_view_type zeta, const typename SphereGeometry::crd_view_type pcrds,
    const Real Omg) :
    rel_vort_error(err),
    abs_vort(omega),
    rel_vort(zeta),
    phys_crds_view(pcrds),
    Omega(Omg) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    rel_vort_error(i) = abs_vort(i) - 2*Omega*phys_crds_view(i,2) - rel_vort(i);
  }
};

void init_input(user::Input& input);

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
    using Lat0 = LatitudeTracer;
    Coriolis coriolis(input.get_option("omega").get_real());
    RossbyWave54Velocity velocity(constants::PI/7);
    RossbyHaurwitz54 vorticity(constants::PI/7);

    //  particle/panel/grid initialization
    using SeedType = CubedSphereSeed;
    using Coriolis = CoriolisSphere;
    const Int mesh_depth = input.get_option("tree_depth").get_int();
    constexpr Real sphere_radius = 1;
    Int amr_buffer = input.get_option("amr_buffer").get_int();
    Int amr_limit = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit = input.get_option("amr_both").get_int();
    }
    const bool amr = (amr_buffer > 0 and amr_limit > 0);
    PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius, amr_buffer, amr_limit);
    const Int nlon = input.get_option("nlon").get_int();
    const Int gmls_order = input.get_option("gmls_interpolation_order").get_int();
    gmls::Params gmls_params(gmls_order);

    // DFS initialization
    auto sphere = std::make_unique<DFS::DFSBVE<SeedType>>(mesh_params, nlon, gmls_params);
    const Real ftle_tol = input.get_option("ftle_tol").get_real();
    sphere->init_vorticity(vorticity);
    sphere->init_velocity(velocity);
    /**
      Initial adaptive refinement
    */
    if (amr) {
      logger.warn("AMR not implemented for DFS yet.");
    }
    Lat0 lat0;
    sphere->init_tracer(lat0);
    logger.info(sphere->info_string());

    // Solver initialization
//     using SolverType = DFS::DFSRK2<SeedType>;
//     using SolverType = DFS::DFSRK3<SeedType>;
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

    logger.info("initial total vorticity, kinetic energy, and enstrophy = ({}, {}, {})",
      total_vorticity[0], total_kinetic_energy[0], total_enstrophy[0]);

    std::string amr_str = "_";
    if (amr) {
      logger.warn("AMR not implemented for DFS yet.");
    }
    const std::string resolution_str =  std::to_string(mesh_depth) + dt_str(dt);
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
    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      logger.debug("stepping time 1: idx {}", t_idx);
      sphere->advance_timestep(*solver);
      logger.debug("stepping time 2: idx {}", solver->t_idx);
      Kokkos::parallel_for(sphere->mesh.n_faces_host(),
        ComputeFTLE<SeedType>(sphere->ftle.view,
          sphere->mesh.vertices.phys_crds.view,
          sphere->ref_crds_passive.view,
          sphere->mesh.faces.phys_crds.view,
          sphere->ref_crds_active.view,
          sphere->mesh.faces.verts,
          sphere->mesh.faces.mask,
          sphere->t - sphere->t_ref));
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

        auto remesh = compadre_remesh(*new_sphere, *sphere, gmls_params);
        if (amr) {
          logger.error("AMR not yet implemented for DFS; skipping remesh step.");
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            remesh.uniform_indirect_remesh(vorticity, coriolis, lat0);
          }
        }
        tref = sphere->t;
        sphere = std::move(new_sphere);
        sphere->t_ref = tref;
        solver.reset(new SolverType(dt, *sphere, solver->t_idx));
      }

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

      logger.info("t = {}", (t_idx+1)*dt);
    }
    const std::string matlab_file = ofile_root + ".m";
    std::ofstream ofile(matlab_file);
    write_vector_matlab(ofile, "time", time);
    write_vector_matlab(ofile, "total_vorticity", total_vorticity);
    write_vector_matlab(ofile, "total_kinetic_energy", total_kinetic_energy);
    write_vector_matlab(ofile, "total_enstrophy", total_enstrophy);
    write_vector_matlab(ofile, "max_ftle", ftle_max);
    ofile.close();
  }  // Kokkos scope
  /**
    program finalize
  */
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

  user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("dfs_rh54"));
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
}
