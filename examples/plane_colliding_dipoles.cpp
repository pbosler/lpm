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
#include "mesh/lpm_ftle.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

using namespace Lpm;

void input_init(user::Input& input);

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("colliding_dipoles", Log::level::debug, comm);

  // compile-time settings
  using seed_type = QuadRectSeed;
  // plane gravity wave problem setup
  using vorticity_type = CollidingDipolePairPlane;
  using coriolis_type = CoriolisBetaPlane;
  using geo = PlaneGeometry;

  Kokkos::initialize(argc, argv);
  { // Kokkos scope
    user::Input input("colliding_dipoles");
    input_init(input);
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    Int amr_buffer = input.get_option("amr_buffer").get_int();
    Int amr_limit = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit = input.get_option("amr_both").get_int();
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
      amr_buffer, amr_limit);
    Real max_circ_tol = input.get_option("max_circulation_tol").get_real();
    Real flow_map_var_tol = input.get_option("flow_map_variation_tol").get_real();
    Real zeta_var_tol = input.get_option("vorticity_variation_tol").get_real();
    const bool amr = (mesh_params.amr_limit > 0 and
                      ( (max_circ_tol < 0.5 * std::numeric_limits<Real>::max() or
                         flow_map_var_tol < 0.5* std::numeric_limits<Real>::max()) or
                         zeta_var_tol < 0.5*std::numeric_limits<Real>::max()));

    coriolis_type coriolis(input.get_option("f-coriolis").get_real(),
      input.get_option("beta-coriolis").get_real());
    const Real epsilon = input.get_option("kernel_smoothing_parameter").get_real();
    auto plane = std::make_unique<Incompressible2D<seed_type>>(mesh_params, coriolis, epsilon);

    // set problem initial conditions
    vorticity_type vorticity;
    plane->init_vorticity(vorticity);

    if (amr) {
      Refinement<seed_type> refiner(plane->mesh);
      ScalarIntegralFlag max_circulation_flag(refiner.flags,
        plane->rel_vort_active.view,
        plane->mesh.faces.area,
        plane->mesh.faces.mask,
        plane->mesh.n_faces_host(),
        max_circ_tol);
      FlowMapVariationFlag<seed_type> flow_map_variation_flag(
        refiner.flags,
        plane->mesh,
        flow_map_var_tol);
      ScalarVariationFlag zeta_var_flag(refiner.flags,
        plane->rel_vort_active.view,
        plane->rel_vort_passive.view,
        plane->mesh.faces.verts,
        plane->mesh.faces.mask,
        plane->mesh.n_faces_host(),
        zeta_var_tol);


      max_circulation_flag.set_tol_from_relative_value();
      max_circ_tol = max_circulation_flag.tol;
      flow_map_variation_flag.set_tol_from_relative_value();
      flow_map_var_tol = flow_map_variation_flag.tol;
      zeta_var_flag.set_tol_from_relative_value();
      zeta_var_tol = zeta_var_flag.tol;

      logger.info("amr is enabled with limit {}, max_circ_tol = {}, flow_map_var_tol = {}, zeta_var_tol = {}",  mesh_params.amr_limit, max_circ_tol, flow_map_var_tol, zeta_var_tol);

      Index vert_start_idx = 0;
      Index face_start_idx = 0;
      for (int i=0; i<amr_limit; ++i) {
        const Index vert_end_idx = plane->mesh.n_vertices_host();
        const Index face_end_idx = plane->mesh.n_faces_host();
        refiner.iterate(face_start_idx, face_end_idx, max_circulation_flag, zeta_var_flag);

        logger.info("amr iteration {}: initial circulation refinement count = {}",
          i, refiner.count[0]);
        logger.info("amr iteration {}: vorticity variation refinement count = {}",
          i, refiner.count[1]);

        plane->mesh.divide_flagged_faces(refiner.flags, logger);
        plane->update_device();
        plane->init_vorticity(vorticity);

        vert_start_idx = vert_end_idx;
        face_start_idx = face_end_idx;
      }
    }
    else {
      logger.info("amr is not enabled; using uniform meshes.");
    }
    plane->init_direct_sums();

    const auto vel_range = plane->velocity_active.range(plane->mesh.n_faces_host());
    const Real cr = vel_range.second * dt / plane->mesh.appx_min_mesh_size();
    logger.info(plane->info_string());
    logger.info("velocity magnitude (min, max) = ({}, {}); approximate Courant number = {}",
      vel_range.first, vel_range.second, cr);
    if (cr > 0.5) {
        logger.warn("Courant number {} may be too high.", cr);
    }
    const Int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy = input.get_option("remesh_strategy").get_str();

#ifdef LPM_USE_VTK
    std::string amr_str = "_";
    if (amr) {
      amr_str = "amr" + std::to_string(amr_limit) + "_";
      if (max_circ_tol < 1) {
        amr_str += "gamma_tol" + float_str(max_circ_tol);
      }
      if (flow_map_var_tol < 10) {
        amr_str += "_fmap_tol" + float_str(flow_map_var_tol);
      }
      if (zeta_var_tol < 1) {
        amr_str += "_zeta_var" + float_str(zeta_var_tol);
      }
      amr_str += "_";
    }
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int()) + dt_str(dt);
    const std::string remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    const std::string vtk_file_root = input.get_option("output_file_directory").get_str() +
       "/" + input.get_option("output_file_root").get_str() +
       "_" + seed_type::id_string() + resolution_str + "_" + remesh_str + amr_str;
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

    /*
    * FTLE
    */
    const bool use_ftle = (input.get_option("remesh_trigger").get_str() == "ftle");
    const Real ftle_tol = input.get_option("ftle_tol").get_real();
    Real max_ftle = 0;
    Real tref = 0;

    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      const bool ftle_trigger = (use_ftle and max_ftle > ftle_tol);
      const bool interval_trigger = ((t_idx+1)%remesh_interval == 0);
      const bool do_remesh = (ftle_trigger or interval_trigger);
      if  (do_remesh) {
        ++remesh_counter;
        if (interval_trigger) {
          logger.debug("remesh {} triggered by remesh interval", remesh_counter);
        }
        else {
          logger.info("remesh {} triggered by ftle_tol {}", remesh_counter, ftle_tol);
        }

        auto new_plane = std::make_unique<Incompressible2D<seed_type>>(mesh_params, coriolis, epsilon);
        auto remesh = bivar_remesh(*new_plane, *plane);

        if (amr) {
          Refinement<seed_type> refiner(new_plane->mesh);
          ScalarIntegralFlag max_circulation_flag(refiner.flags,
            new_plane->rel_vort_active.view,
            new_plane->mesh.faces.area,
            new_plane->mesh.faces.mask,
            new_plane->mesh.n_faces_host(),
            max_circ_tol);

          FlowMapVariationFlag<seed_type> flow_map_variation_flag(
            refiner.flags,
            new_plane->mesh,
            flow_map_var_tol);

          ScalarVariationFlag zeta_var_flag(refiner.flags,
            new_plane->rel_vort_active.view,
            new_plane->rel_vort_passive.view,
            new_plane->mesh.faces.verts,
            new_plane->mesh.faces.mask,
            new_plane->mesh.n_faces_host(),
            zeta_var_tol);

          if (remesh_strategy == "direct") {
            remesh.adaptive_direct_remesh(refiner, max_circulation_flag, zeta_var_flag, flow_map_variation_flag);
          }
          else {
            remesh.adaptive_indirect_remesh(vorticity, coriolis, refiner, zeta_var_flag, max_circulation_flag, flow_map_variation_flag);
          }
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            remesh.uniform_indirect_remesh(vorticity, coriolis);
          }
        }
        tref = plane->t;
        plane = std::move(new_plane);
        plane->update_device();
        auto new_solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *plane, solver->t_idx);
        solver = std::move(new_solver);

      }

    plane->advance_timestep(*solver);
    logger.debug("t = {}", plane->t);
    Kokkos::parallel_for(plane->mesh.n_faces_host(),
      ComputeFTLE<seed_type>(plane->ftle.view,
        plane->mesh.vertices.phys_crds.view,
        plane->ref_crds_passive.view,
        plane->mesh.faces.phys_crds.view,
        plane->ref_crds_active.view,
        plane->mesh.faces.verts,
        plane->mesh.faces.mask,
        plane->t - tref));
    max_ftle = get_max_ftle(plane->ftle.view, plane->mesh.faces.mask, plane->mesh.n_faces_host());

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

void input_init(user::Input& input) {
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

  user::Option amr_both_option("amr_both", "-amr", "--amr-both", "both amr buffer and limit values", LPM_NULL_IDX);
  input.add_option(amr_both_option);

  user::Option max_circulation_option("max_circulation_tol", "-c", "--circuluation-max", "amr max circulation tolerance", std::numeric_limits<Real>::max());
  input.add_option(max_circulation_option);

  user::Option vorticity_variation_option("vorticity_variation_tol", "-z", "--zeta-variation", "amr max relative vorticity variation tolerance", std::numeric_limits<Real>::max());
  input.add_option(vorticity_variation_option);

  user::Option flow_map_variation_option("flow_map_variation_tol", "-fv", "--flow-map-variation", "amr max flow map variation tolerance", std::numeric_limits<Real>::max());
  input.add_option(flow_map_variation_option);

  user::Option output_file_directory_option("output_file_directory", "-odir", "--output-dir", "output file directory", std::string("."));
  input.add_option(output_file_directory_option);

  user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("colliding_dipoles"));
  input.add_option(output_file_root_option);

  user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
  input.add_option(output_write_frequency_option);

  user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
  input.add_option(kernel_smoothing_parameter_option);

  user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", std::numeric_limits<int>::max());
  input.add_option(remesh_interval_option);

  user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
  input.add_option(remesh_strategy_option);

  user::Option remesh_trigger_option("remesh_trigger", "-rt", "--remesh-trigger", "trigger for a remeshing : ftle or an interval", std::string("interval"), std::set<std::string>({"interval", "ftle"}));
  input.add_option(remesh_trigger_option);

  user::Option ftle_tolerance_option("ftle_tol", "-ftle", "--ftle-tol", "max value for ftle before remesh", 2.0);
  input.add_option(ftle_tolerance_option);
}
