#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "lpm_incompressible2d_rk2_impl.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe_kernels.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"
#include "mesh/lpm_ftle.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_tuple.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

using namespace Lpm;

void init_input(user::Input& input);

template <typename SeedType>
void exact_velocity(Kokkos::View<Real*[3]> passive_exact, Kokkos::View<Real*[3]> active_exact,
  Kokkos::View<Real*[3]> passive_error, Kokkos::View<Real*[3]> active_error,
  const RossbyWave54Velocity& rh54_velocity,
  const Incompressible2D<SeedType>& sphere);

template <typename SeedType>
void exact_vorticity(scalar_view_type zeta_passive,
  scalar_view_type zeta_active, const RossbyHaurwitz54& zeta,
  const Incompressible2D<SeedType>& sphere);

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("bve_rh54", Log::level::debug, comm);
  using seed_type = CubedSphereSeed;
  using Coriolis = CoriolisSphere;
  using Vorticity = RossbyHaurwitz54;
  using Velocity = RossbyWave54Velocity;
  using Lat0 = LatitudeTracer;
  using Solver = Incompressible2DRK2<seed_type>;
    Kokkos::initialize(argc, argv);
  {
    user::Input input("bve_rh54");
    init_input(input);
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
    std::vector<Real> total_vorticity(nsteps+1);
    std::vector<Real> total_kinetic_energy(nsteps+1);
    std::vector<Real> total_enstrophy(nsteps+1);
    std::vector<Real> zeta_l1(nsteps+1);
    std::vector<Real> zeta_l2(nsteps+1);
    std::vector<Real> zeta_linf(nsteps+1);
    std::vector<Real> vel_l1(nsteps+1);
    std::vector<Real> vel_l2(nsteps+1);
    std::vector<Real> vel_linf(nsteps+1);
    std::vector<Real> time(nsteps+1);
    std::vector<Real> ftle_max(nsteps+1);

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
    Vorticity vorticity;
    vorticity.set_stationary_wave_speed(coriolis.Omega);
    Velocity velocity(vorticity);
    constexpr Real velocity_eps = 0;

    auto sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
      coriolis, velocity_eps);
    sphere->init_vorticity(vorticity);

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
          sphere->init_vorticity(vorticity);

          face_start_idx = face_end_idx;
        }
    }
    else {
      logger.info("amr is not enabled; using uniform meshes.");
    }
    sphere->init_direct_sums();

    Kokkos::View<Real*[3]> velocity_exact_passive("velocity_exact", mesh_params.nmaxverts);
    Kokkos::View<Real*[3]> velocity_exact_active("velocity_exact", mesh_params.nmaxfaces);
    Kokkos::View<Real*[3]> vel_error_passive("vel_error", mesh_params.nmaxverts);
    Kokkos::View<Real*[3]> vel_error_active("vel_error", mesh_params.nmaxfaces);

    Lat0 lat0;
    sphere->init_tracer(lat0);
    sphere->allocate_tracer(std::string("exact_vorticity"));
    sphere->allocate_tracer(std::string("vorticity_error"));
    exact_vorticity(sphere->tracer_passive.at("exact_vorticity").view,
      sphere->tracer_active.at("exact_vorticity").view, vorticity,
      *sphere);


    logger.info(sphere->info_string());
    exact_velocity(velocity_exact_passive, velocity_exact_active,
                vel_error_passive, vel_error_active, velocity, *sphere);

    ErrNorms vel_err(vel_error_active, velocity_exact_active,
                          sphere->mesh.faces.area);
    logger.info("velocity error : {}", vel_err.info_string());
    vel_l1[0] = vel_err.l1;
    vel_l2[0] = vel_err.l2;
    vel_linf[0] = vel_err.linf;


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

    const bool use_ftle = (input.get_option("remesh_trigger").get_str() == "ftle");
    const Real ftle_tol = input.get_option("ftle_tol").get_real();

    total_vorticity[0] = sphere->total_vorticity();
    total_kinetic_energy[0] = sphere->total_kinetic_energy();
    total_enstrophy[0] = sphere->total_enstrophy();
    ftle_max[0] = 0;

    std::string amr_str = "_";
    if (amr) {
      amr_str = "amr" + std::to_string(amr_limit) + "_";
      if (max_circ_tol < 1) {
        amr_str += "gamma_tol" + float_str(max_circ_tol);
      }
      amr_str += "_";
    }
    const std::string eps_str = "eps"+float_str(sphere->eps);
    const std::string resolution_str =
      std::to_string(input.get_option("tree_depth").get_int()) + dt_str(dt);
    std::string remesh_str;
    if (use_ftle) {
      remesh_str = remesh_strategy + "ftle" + float_str(ftle_tol,3);
    }
    else {
     remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    }

    const std::string ofile_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + "_" + remesh_str + amr_str;
#ifdef LPM_USE_VTK
    const std::string vtk_file_root = ofile_root;
    {
      sphere->update_host();
      auto vtk = vtk_mesh_interface(*sphere);
      vtk.add_vector_point_data(vel_error_passive);
      vtk.add_vector_cell_data(vel_error_active);
      vtk.add_vector_point_data(velocity_exact_passive);
      vtk.add_vector_cell_data(velocity_exact_active);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file {}", sphere->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif

    /**************************************************
    time stepping
    ***************************************************/

    int rm_counter = 0;
    Real max_ftle = 0;
    Real tref = 0;
    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      const bool ftle_trigger = (use_ftle and max_ftle > ftle_tol);
      const bool interval_trigger = ((t_idx+1)%remesh_interval == 0);
      const bool do_remesh = (ftle_trigger or interval_trigger);
      if ( do_remesh ) {
        ++rm_counter;
        if (interval_trigger) {
          logger.debug("remesh {} triggered by remesh interval", rm_counter);
        }
        else {
          logger.info("remesh {} triggered by ftle", rm_counter);
        }

        auto new_sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
           coriolis, input.get_option("kernel_smoothing_parameter").get_real());
        new_sphere->t = sphere->t;
        new_sphere->allocate_tracer(lat0);
        new_sphere->allocate_tracer(std::string("exact_vorticity"));
        new_sphere->allocate_tracer(std::string("vorticity_error"));

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
              vorticity, coriolis, lat0);
          }
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            remesh.uniform_indirect_remesh(vorticity, coriolis, lat0);
          }
        }
        exact_vorticity(new_sphere->tracer_passive.at("exact_vorticity").view,
          new_sphere->tracer_active.at("exact_vorticity").view, vorticity,
          *new_sphere);

        sphere = std::move(new_sphere);
        solver.reset(new Incompressible2DRK2<seed_type>(dt, *sphere, solver->t_idx));
      }

      sphere->advance_timestep(*solver);
      exact_velocity(velocity_exact_passive, velocity_exact_active,
                vel_error_passive, vel_error_active, velocity, *sphere);
      vel_err = ErrNorms(vel_error_active, velocity_exact_active,
                          sphere->mesh.faces.area);

//       Kokkos::parallel_for(sphere->mesh.n_faces_host(),
//           ComputeFTLE<seed_type>(sphere->ftle.view,
//             sphere->mesh.vertices.phys_crds.view,
//             sphere->ref_crds_passive.view,
//             sphere->mesh.faces.phys_crds.view,
//             sphere->ref_crds_active.view,
//             sphere->mesh.faces.verts,
//             sphere->mesh.faces.mask,
//             sphere->t - sphere->t_ref));
      max_ftle = get_max_ftle(sphere->ftle.view, sphere->mesh.faces.mask, sphere->mesh.n_faces_host());

      exact_vorticity(sphere->tracer_passive.at("exact_vorticity").view,
        sphere->tracer_active.at("exact_vorticity").view, vorticity,
        *sphere);
      compute_error(sphere->tracer_passive.at("vorticity_error").view,
        sphere->rel_vort_passive.view, sphere->tracer_passive.at("exact_vorticity").view);
      ErrNorms zeta_err(sphere->tracer_active.at("vorticity_error").view,
        sphere->rel_vort_active.view, sphere->tracer_active.at("exact_vorticity").view,
        sphere->mesh.faces.area);


      time[t_idx+1] = (t_idx+1) * dt;
      total_vorticity[t_idx+1] = sphere->total_vorticity();
      total_kinetic_energy[t_idx+1] = sphere->total_kinetic_energy();
      total_enstrophy[t_idx+1] = sphere->total_enstrophy();
      zeta_l1[t_idx+1] = zeta_err.l1;
      zeta_l2[t_idx+1] = zeta_err.l2;
      zeta_linf[t_idx+1] = zeta_err.linf;
      vel_l1[t_idx+1] = vel_err.l1;
      vel_l2[t_idx+1] = vel_err.l2;
      vel_linf[t_idx+1] = vel_err.linf;
      ftle_max[t_idx+1] = max_ftle;

      logger.info("t = {}: relvort err (l1, l2, linf) = ({}, {}, {}), vel_err = ({}, {}, {})", sphere->t,
        zeta_err.l1, zeta_err.l2, zeta_err.linf, vel_err.l1,
        vel_err.l2, vel_err.linf);

    #ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {
        sphere->update_host();
        auto vtk = vtk_mesh_interface(*sphere);
        vtk.add_vector_point_data(vel_error_passive);
        vtk.add_vector_cell_data(vel_error_active);
        vtk.add_vector_point_data(velocity_exact_passive);
        vtk.add_vector_cell_data(velocity_exact_active);
        auto ctr_str = zero_fill_str(++frame_counter);
        const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
        logger.info("writing output at t = {} to file: {}", sphere->t, vtk_fname);
        vtk.write(vtk_fname);
      }
    #endif
    } // time stepping loop

    const std::string matlab_file = ofile_root + ".m";
    std::ofstream ofile(matlab_file);
    write_vector_matlab(ofile, "time", time);
    write_vector_matlab(ofile, "total_vorticity", total_vorticity);
    write_vector_matlab(ofile, "total_kinetic_energy", total_kinetic_energy);
    write_vector_matlab(ofile, "total_enstrophy", total_enstrophy);
    write_vector_matlab(ofile, "max_ftle", ftle_max);
    write_vector_matlab(ofile, "vorticity_l1", zeta_l1);
    write_vector_matlab(ofile, "vorticity_l2", zeta_l2);
    write_vector_matlab(ofile, "vorticity_linf", zeta_linf);
    write_vector_matlab(ofile, "velocity_l1", vel_l1);
    write_vector_matlab(ofile, "velocity_l2", vel_l2);
    write_vector_matlab(ofile, "velocity_linf", vel_linf);
    ofile.close();

  } // kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}

template <typename SeedType>
void exact_velocity(Kokkos::View<Real*[3]> passive_exact, Kokkos::View<Real*[3]> active_exact,
  Kokkos::View<Real*[3]> passive_error, Kokkos::View<Real*[3]> active_error,
  const RossbyWave54Velocity& rh54_velocity,
  const Incompressible2D<SeedType>& sphere) {

  // set time of evaluation to zero : stationary wave
  constexpr Real t_rh = 0; // TODO: fix for non-stationary waves

  const auto vx = sphere.mesh.vertices.phys_crds.view;
  const auto vvel = sphere.velocity_passive.view;
  Kokkos::parallel_for(sphere.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vx, i, Kokkos::ALL);

      Kokkos::Tuple<Real,3> vel = rh54_velocity(xi, t_rh);
      for (int j=0; j<3; ++j) {
        passive_exact(i,j) = vel[j];
        passive_error(i,j) = vvel(i,j) - vel[j];
      }
    });

  const auto fx = sphere.mesh.faces.phys_crds.view;
  const auto fvel = sphere.velocity_active.view;
  Kokkos::parallel_for(sphere.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(fx, i, Kokkos::ALL);

      Kokkos::Tuple<Real,3> vel = rh54_velocity(xi, t_rh);
      for (int j=0; j<3; ++j) {
        active_exact(i,j) = vel[j];
        active_error(i,j) = fvel(i,j) - vel[j];
      }
    });
}

template <typename SeedType>
void exact_vorticity(scalar_view_type zeta_passive,
  scalar_view_type zeta_active, const RossbyHaurwitz54& zeta,
  const Incompressible2D<SeedType>& sphere) {

  constexpr Real t0 = 0;
  const auto vx = sphere.mesh.vertices.phys_crds.view;
  Kokkos::parallel_for(sphere.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vx, i, Kokkos::ALL);

      zeta_passive(i) = zeta(xi);
    });
  const auto fx = sphere.mesh.faces.phys_crds.view;
  Kokkos::parallel_for(sphere.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(fx, i, Kokkos::ALL);
      zeta_active(i) = zeta(xi);
    });
}

void init_input(user::Input& input) {
      user::Option tfinal_option("tfinal", "-tf", "--time-final", "time final", 0.025);
      input.add_option(tfinal_option);

      user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of time steps", 1);
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

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("rh54"));
      input.add_option(output_file_root_option);

      user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", std::numeric_limits<int>::max());
      input.add_option(remesh_interval_option);

      user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
      input.add_option(remesh_strategy_option);

      user::Option remesh_interpolation_order("remesh_interpolation_order", "-ro", "--remesh-order", "polynomial order for gmls-based remesh interpolation", 4);
      input.add_option(remesh_interpolation_order);

      user::Option remesh_trigger_option("remesh_trigger", "-rt", "--remesh-trigger", "trigger for a remeshing : ftle or an interval", std::string("interval"), std::set<std::string>({"interval", "ftle"}));
      input.add_option(remesh_trigger_option);

      user::Option ftle_tolerance_option("ftle_tol", "-ftle", "--ftle-tol", "max value for ftle before remesh", 2.0);
      input.add_option(ftle_tolerance_option);
    }


