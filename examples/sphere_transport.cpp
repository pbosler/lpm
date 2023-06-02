#include "LpmConfig.h"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_2d_transport_rk4.hpp"
#include "lpm_2d_transport_rk4_impl.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_compadre.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "fortran/lpm_ssrfpack_interface.hpp"
#include "fortran/lpm_ssrfpack_interface_impl.hpp"
#include "util/lpm_progress_bar.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace Lpm;

/** @brief Returns the approximate Courant number using eqn. (24)
of Lauritzen et al., 2012,  A standard test case suite for two-dimensional
linear transport on the sphere, Geosci. Model Dev. 5:887-901.
*/
inline Real courant_number(const Real dt, const Real dlam) {
  constexpr Real umax = 3.26;
  return dt * umax / dlam;
}

/**
  The mesh type (e.g., quadrilateral or triangular faces) are set with this
  typedef. Changing the mesh seed will therefore require recompiling this
  program.
*/
// typedef CubedSphereSeed seed_type;
typedef IcosTriSphereSeed seed_type;
typedef LauritzenEtAlDeformationalFlow velocity_type;
typedef SphericalGaussianHills tracer_type;

/** @brief Collect all input values into a single struct; like a fortran
  namelist.

  Add ability to change input values via the command line, so that we don't have
  to recompile the program to change parameters.
*/
struct Input {
  Input(int argc, char* argv[]);
  Real dt;
  static constexpr Real dt_default = 0.05;
  Real tfinal;
  static constexpr Real tf_default = 5;
  Int nsteps;
  static constexpr Int nsteps_default = 100;
  std::string base_output_name;
  Int init_depth;
  static constexpr Int init_depth_default = 4;
  Int amr_limit;
  static constexpr Int amr_limit_default = 0;
  Int amr_max;
  static constexpr Int amr_max_default = 0;
  Int remesh_interval;
  static constexpr Int remesh_interval_default = 20;
  Int reset_lagrangian_interval;
  static constexpr Int reset_lagrangian_interval_default = LPM_NULL_IDX;
  Int output_interval;
  static constexpr Int output_interval_default = 1;
  std::string output_dir;
  Real tracer_mass_tol;
  Real tracer_var_tol;
  Real radius;
  Int gmls_order;
  static constexpr Int gmls_order_default = 5;

  bool help_and_exit;
  std::string info_string() const;
  std::string usage() const;
  std::string vtk_base_name;
};

int main(int argc, char* argv[]) {
  /**
    program initialize
  */
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
    /**
      program run
    */
    /**
      initialize problem
    */
    Logger<> logger("sphere transport amr", Log::level::info, comm);

    Input input(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    logger.info(input.info_string());
    const bool amr = input.amr_limit > 0;
    const bool write_output = input.output_interval > 0;

    tracer_type tracer;
    auto mesh_params = PolyMeshParameters<seed_type>(input.init_depth, input.radius, input.amr_limit);
    auto sphere = std::make_unique<TransportMesh2d<seed_type>>(mesh_params);
    sphere->template initialize_velocity<velocity_type>();
    sphere->initialize_tracer(tracer);

    Real tracer_mass_tol = 0;
    Real tracer_var_tol = 0;
    if (amr) {
      /**
        To start adaptive refinement, we convert relative tolerances from the
        input to absolute tolerances based on the initialized uniform mesh and
        functions defined on it.
      */
      Kokkos::View<bool*> flags("refinement_flags", mesh_params.nmaxfaces);

      auto face_area = sphere->faces.area;
      auto face_tracer = sphere->tracer_faces.at(tracer.name()).view;
      auto face_mask = sphere->faces.mask;
      ScalarIntegralFlag mass_flag(flags, face_tracer, face_area, face_mask, sphere->n_faces_host(), input.tracer_mass_tol);
      mass_flag.set_tol_from_relative_value();
      tracer_mass_tol = mass_flag.tol;
      const Real max_tracer_mass = mass_flag.tol / input.tracer_mass_tol;

      logger.info("max_tracer_mass per face = {}", max_tracer_mass);
      logger.info(
          "input relative tracer mass tol {} converts to absolute tracer mass "
          "tol {}",
          input.tracer_mass_tol, mass_flag.tol);


      auto vert_tracer = sphere->tracer_verts.at(tracer.name()).view;
      auto face_verts = sphere->faces.verts;

      ScalarVariationFlag var_flag(flags, face_tracer, vert_tracer, face_verts, face_mask, sphere->n_faces_host(), input.tracer_var_tol);
      var_flag.set_tol_from_relative_value();
      tracer_var_tol = var_flag.tol;
      const Real max_tracer_var = var_flag.tol / input.tracer_var_tol;
      logger.info("max_tracer_var per face = {}", max_tracer_var);

      logger.info(
          "input relative tracer var tol {} converts to absolute tol {}",
          input.tracer_var_tol, var_flag.tol);

      /**
        Adaptive refinement: for each iteration, all leaves in the mesh face
        tree are are checked and flagged if any amr criteria are met.
      */

      Index verts_start_idx = 0;
      Index faces_start_idx = 0;
      for (int i = 0; i < input.amr_max; ++i) {
        Index verts_end_idx = sphere->n_vertices_host();
        Index faces_end_idx = sphere->n_faces_host();

        /// refinement criterion 1: mass per face, designed to refine local
        /// maxima
        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
            mass_flag);
        Index mass_refinement_count;
        Kokkos::parallel_reduce(
            sphere->n_faces_host(),
            KOKKOS_LAMBDA(const Index i, Index& ct) { ct += Index(flags(i)); },
            mass_refinement_count);
        logger.info("amr iteration {}: initial mass_refinement_count = {}", i,
                    mass_refinement_count);

        /// refinement criterion 2: scalar variation, designed to refine steep
        /// gradients
        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
            var_flag);
        Index total_refinement_count;
        Kokkos::parallel_reduce(
            sphere->n_faces_host(),
            KOKKOS_LAMBDA(const Index i, Index& ct) { ct += Index(flags(i)); },
            total_refinement_count);

        logger.info("amr iteration {}: variation_refinement_count = {}", i,
                    total_refinement_count - mass_refinement_count);

        /// divide all flagged faces
        sphere->divide_flagged_faces(flags, logger);
        /// reset flags for next iteration
        Kokkos::deep_copy(flags, false);
        /// set scalar values on new faces
        verts_start_idx = verts_end_idx;
        faces_start_idx = faces_end_idx;
        sphere->set_tracer_from_lag_crds(tracer, verts_start_idx, faces_start_idx);
      }  // amr iterations

      /**
        Compute Courant number based on smallest panel size.
      */
      Real min_area;
      Kokkos::parallel_reduce(
          sphere->n_faces_host(),
          KOKKOS_LAMBDA(const Index i, Real& ar) {
            if (!face_mask(i)) {
              ar = (face_area(i) < ar ? face_area(i) : ar);
            }
          },
          Kokkos::Min<Real>(min_area));

      Real min_edge_length = sqrt(min_area);
      logger.info("Approximate minimum edge length = {}; Courant number = {}",
                  min_edge_length, courant_number(input.dt, min_edge_length));

    }  // if (amr)

    /**
      initialization complete; write initial conditions to output, if enabled.
    */
    logger.info(sphere->info_string());
    int frame_counter = 0;
#ifdef LPM_USE_VTK
    if (write_output) {
      VtkPolymeshInterface<seed_type> vtk = vtk_interface(*sphere);
      vtk.write(input.vtk_base_name + zero_fill_str(frame_counter++) +
                vtp_suffix());
    }
#endif

    /**
      problem run
    */
    auto solver = std::make_unique<Transport2dRK4<seed_type>>(input.dt, *sphere);
    Int remesh_ctr = 0;
    const bool do_remesh = (input.remesh_interval != LPM_NULL_IDX and input.remesh_interval > 0);
    const bool do_reset = (do_remesh and input.reset_lagrangian_interval != LPM_NULL_IDX and input.reset_lagrangian_interval > 0);
    gmls::Params gmls_params(input.gmls_order);
    const std::vector<Compadre::TargetOperation> gmls_ops({Compadre::VectorPointEvaluation});

    logger.debug("initiating time step loop.");
    ProgressBar progress("sphere_transport", input.nsteps);
    for (Int time_idx = 0; time_idx<input.nsteps; ++time_idx) {

      /**
        Remesh before timestep
      */
      if (do_remesh) {
        if ((time_idx + 1)%input.remesh_interval == 0) {
          ++remesh_ctr;
          /// build new (destination) mesh
          auto new_sphere = std::make_unique<TransportMesh2d<seed_type>>(mesh_params);
          new_sphere->t = sphere->t;
          new_sphere->t_idx = sphere->t_idx;
          new_sphere->allocate_scalar_tracer(tracer.name());

          /// gather data from current (source) mesh
          GatherMeshData<seed_type> gather_src(*sphere);
          gather_src.unpack_coordinates();
          logger.debug("created gather object");
          gather_src.init_scalar_fields(sphere->tracer_verts, sphere->tracer_faces);
          gather_src.gather_scalar_fields(sphere->tracer_verts, sphere->tracer_faces);
          logger.debug("gathered fields");
          gather_src.update_host();

          logger.debug("gathered src data");

          /// choose remeshing procedure
          if (remesh_ctr < input.reset_lagrangian_interval or !do_reset) {
            /// remesh using t=0
            logger.debug("initiating remesh {} to t=0", remesh_ctr);

            SSRFPackInterface ssrfpack(gather_src);
            ssrfpack.interpolate_lag_crds(*new_sphere);
            new_sphere->update_device();
            new_sphere->set_tracer_from_lag_crds(tracer);

            if (amr) {
              Kokkos::View<bool*> flags("refinement_flags", mesh_params.nmaxfaces);

              Index verts_start_idx = 0;
              Index faces_start_idx = 0;
              auto face_area = new_sphere->faces.area;
              auto face_tracer = new_sphere->tracer_faces.at(tracer.name()).view;
              auto face_mask = new_sphere->faces.mask;
              auto vert_tracer = new_sphere->tracer_verts.at(tracer.name()).view;
              auto face_verts = new_sphere->faces.verts;

              ScalarIntegralFlag mass_flag(flags, face_tracer, face_area, face_mask, new_sphere->n_faces_host(), tracer_mass_tol);

              ScalarVariationFlag var_flag(flags, face_tracer, vert_tracer, face_verts, face_mask, new_sphere->n_faces_host(), tracer_var_tol);

              for (int i=0; i<input.amr_max; ++i) {
                Index verts_end_idx = new_sphere->n_vertices_host();
                Index faces_end_idx = new_sphere->n_faces_host();

                Kokkos::parallel_for(
                  Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
                  mass_flag);

                Index mass_refinement_count;
                Kokkos::parallel_reduce(
                  Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
                  KOKKOS_LAMBDA (const Index i, Index& ct) {
                    ct += Index(flags(i));
                  }, mass_refinement_count);

                Kokkos::parallel_for(
                  Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
                  var_flag);

                Index total_refinement_count;
                Kokkos::parallel_reduce(
                  Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
                  KOKKOS_LAMBDA (const Index i, Index& ct) {
                    ct += Index(flags(i));
                  }, total_refinement_count);

                logger.info("remesh {}, amr iter. {} : mass refine count = {}, variation refine count = {}",
                  remesh_ctr, i, mass_refinement_count, total_refinement_count - mass_refinement_count);

                new_sphere->divide_flagged_faces(flags, logger);

                ssrfpack.interpolate_lag_crds(*new_sphere, verts_start_idx, faces_start_idx);
                new_sphere->update_device();
                new_sphere->set_tracer_from_lag_crds(tracer, verts_start_idx, faces_start_idx);

                verts_start_idx = verts_end_idx;
                faces_start_idx = faces_end_idx;
              }
            }
          }
          else if (do_reset and remesh_ctr == input.reset_lagrangian_interval) {
            /// remesh to t=0 reference, and create new reference to current time
            logger.error("reset_lagrangian_interval is not implemented yet.");
          }
          else if (do_reset and remesh_ctr>input.reset_lagrangian_interval and remesh_ctr%input.reset_lagrangian_interval == 0) {
            /// remesh to existing reference, then create a new reference to the current time
            logger.error("reset_lagrangian_interval is not implemented yet.");
          }
          else {
            /// remesh to existing reference
            logger.error("reset_lagrangian_interval is not implemented yet.");
          }

          /// set velocity on new mesh
          new_sphere->template set_velocity<velocity_type>(sphere->t);

          /// replace old mesh with new mesh
          sphere = std::move(new_sphere);

          /// replace old solver with new solver
          auto new_solver = std::make_unique<Transport2dRK4<seed_type>>(input.dt, *sphere);
          solver = std::move(new_solver);
        }

      } // end if do remesh

      /// time step
      solver->template advance_timestep<velocity_type>();
      progress.update();
      logger.debug("time step {} of {}; t = {}", sphere->t_idx, input.nsteps, sphere->t);

#ifdef LPM_USE_VTK
      /// write intermediate output
      if (write_output and (time_idx + 1)%input.output_interval == 0 and (time_idx+1) != input.nsteps) {
        VtkPolymeshInterface<seed_type> vtk = vtk_interface(*sphere);
        vtk.write(input.vtk_base_name + zero_fill_str(frame_counter++) + vtp_suffix());
      }
#endif
    } // time step loop

    /// compute tracer error
    Kokkos::View<Real*> tracer_error_faces("tracer_error", sphere->n_faces_host());
    Kokkos::View<Real*> tracer_error_verts("tracer_error", sphere->n_vertices_host());
    const auto vert_tracer = sphere->tracer_verts.at(tracer.name()).view;
    const auto face_tracer = sphere->tracer_faces.at(tracer.name()).view;
    const auto crd_verts = sphere->vertices.phys_crds.view;
    const auto crd_faces = sphere->faces.phys_crds.view;
    const auto face_mask = sphere->faces.mask;
    Kokkos::parallel_for("compute_tracer_error_verts", sphere->n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mcrd = Kokkos::subview(crd_verts, i, Kokkos::ALL);
        tracer_error_verts(i) = vert_tracer(i) - tracer(mcrd);
      });
    Kokkos::View<Real*> tracer_exact_faces("tracer_exact", sphere->n_faces_host());
    Kokkos::parallel_for("compute_tracer_exact_faces", sphere->n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        if (!face_mask(i)) {
          const auto mcrd = Kokkos::subview(crd_faces, i, Kokkos::ALL);
          tracer_exact_faces(i) = tracer(mcrd);
        }
      });
    const auto face_tracer_err_norms = ErrNorms(tracer_error_faces,
      Kokkos::subview(face_tracer, std::make_pair(0, sphere->n_faces_host())),
      tracer_exact_faces,
      Kokkos::subview(sphere->faces.area, std::make_pair(0, sphere->n_faces_host())),
      Kokkos::subview(sphere->faces.mask, std::make_pair(0, sphere->n_faces_host())));

    logger.info("Face tracer error : nsteps = {}, n_faces_final = {}, l1 = {}, l2 = {}, linf = {}",
      input.nsteps, sphere->faces.n_leaves_host(), face_tracer_err_norms.l1, face_tracer_err_norms.l2,
      face_tracer_err_norms.linf);


#ifdef LPM_USE_VTK
      /// write final output
      if (write_output) {
        VtkPolymeshInterface<seed_type> vtk = vtk_interface(*sphere);
        vtk.add_scalar_point_data(tracer_error_verts);
        vtk.add_scalar_cell_data(tracer_error_faces);
        vtk.write(input.vtk_base_name + zero_fill_str(frame_counter++) + vtp_suffix());
      }
#endif


  }  // Kokkos scope
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
}

Input::Input(int argc, char* argv[]) {
  dt = 0.05;
  tfinal = 5;
  nsteps = 400;
  base_output_name = "transport_";
  init_depth = 4;
  amr_limit = 0;
  amr_max = 1;
  tracer_mass_tol = 0.15;
  tracer_var_tol = 0.15;
  remesh_interval = 20;
  reset_lagrangian_interval = LPM_NULL_IDX;
  output_interval = 1;
  output_dir = "";
  help_and_exit = false;
  radius = 1;
  gmls_order = 5;
  bool use_dt = true;
  bool use_nsteps = false;
  for (int i = 1; i < argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-dt") {
      dt = std::stod(argv[++i]);
      use_dt = true;
      use_nsteps = false;
      LPM_REQUIRE(dt > 0);
    } else if (token == "-tf") {
      tfinal = std::stod(argv[++i]);
      LPM_REQUIRE(tfinal >= 0);
    } else if (token == "-o") {
      base_output_name = argv[++i];
    } else if (token == "-d") {
      init_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_depth >= 0);
    } else if (token == "-amr") {
      amr_limit = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_limit >= 0);
    } else if (token == "-amr_max") {
      amr_max = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_max >= 0);
    } else if (token == "-mass_tol") {
      tracer_mass_tol = std::stod(argv[++i]);
      LPM_REQUIRE(tracer_mass_tol > 0);
    } else if (token == "-var_tol") {
      tracer_var_tol = std::stod(argv[++i]);
      LPM_REQUIRE(tracer_var_tol > 0);
    } else if (token == "-rf") {
      remesh_interval = std::stoi(argv[++i]);
      LPM_REQUIRE(remesh_interval > 0 or remesh_interval == LPM_NULL_IDX);
    } else if (token == "-of") {
      output_interval = std::stoi(argv[++i]);
      LPM_REQUIRE(output_interval > 0 or output_interval == LPM_NULL_IDX);
    } else if (token == "-dir") {
      output_dir = argv[++i];
    } else if (token == "-h") {
      help_and_exit = true;
    } else if (token == "-nsteps") {
      nsteps = std::stoi(argv[++i]);
      use_nsteps = true;
      use_dt = false;
    } else if (token == "-reset") {
      reset_lagrangian_interval = std::stoi(argv[++i]);
      LPM_REQUIRE(reset_lagrangian_interval > 0 or reset_lagrangian_interval == LPM_NULL_IDX);
    } else if (token == "-gmls") {
      gmls_order = std::stoi(argv[++i]);
      LPM_REQUIRE(gmls_order >= 2);
    }
    else {
      throw std::invalid_argument(token);
    }
  }
  if (use_nsteps) {
    dt = tfinal / nsteps;
  }
  else if (use_dt) {
    nsteps = int(tfinal/dt);
  }
  vtk_base_name = output_dir + (output_dir.empty() ? "" : "/") +
                  base_output_name + seed_type::id_string() + "_d" +
                  std::to_string(init_depth) + "_";
  if (amr_limit > 0) {
    vtk_base_name += "amr" + std::to_string(amr_max)+"_";
  }
  const char* fmt = "dt%.3f";
  int sz = std::snprintf(nullptr, 0, fmt, dt);
  std::vector<char> buf(sz + 1);
  std::snprintf(&buf[0], buf.size(), fmt, dt);
  vtk_base_name += std::string(&buf[0], sz);
  vtk_base_name += "_rm" + std::to_string(remesh_interval) + "_";
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "Spherical transport with AMR:\n \tThis program demonstrates adaptive "
        "refinement in a spherical transport problem, solving test case 1 from "
        "Laurizten et al., 2012, A standard test case suite for "
        "two-dimensional linear transport on the sphere, Geosci. Model Dev. "
        "5.\n The test uses a reversible velocity field with period T=5, so "
        "that the exact solution of the tracer at t=T matches the initial "
        "condition at t=0.\n";
  auto tabstr = indent_string(1);
  ss << tabstr << "optional arguments:\n";
  ss << tabstr
     << "-dt [nonnegative real number] time step size (default: "
     << dt_default << "); "
     << "this will be overridden by -nsteps if both are present.\n";
  ss << tabstr
     << "-tf [nonnegative real number] final time for integration (default: "
     << tf_default << ")\n";
  ss << tabstr << "-nsteps [nonnegative integer] number of timesteps.\n";
  ss << tabstr
     << "-o [string] output filename root (default: "
        "\"sphere_transport_amr_example\"\n";
  ss << tabstr
     << "-d [nonnegative integer] initial depth of mesh quadtree (default: "
     << init_depth_default << ")\n";
  ss << tabstr
     << "-amr [nonnegative integer] number of uniform refinements beyond "
        "initial depth to allocate memory for; values > 0 will enable adaptive "
        "refinement (default: 0)\n";
  ss << tabstr
     << "-amr_max [nonnegative integer] maximum number of times a panel may be "
     << "divided (default: 1)\n.";
  ss << tabstr
     << "-mass_tol [positive real number] threshold for local tracer integral "
     << "refinement criterion; not used if amr = 0 (default: 0.1)\n";
  ss << tabstr
     << "-var_tol [positive real number] threshold for local tracer variation "
     <<  "refinement criterion; not used if amr = 0 (default: 0.15).\n";
  ss << tabstr
     << "-rf [positive integer or -1] frequency of remesh/remap "
        "interpolations; setting value to -1 will disable remeshing (default: "
     << remesh_interval_default << ")\n";
  ss << tabstr
     << "-of [positive integer or -1] frequency of vtk output; setting value "
        "to -1 will disable vtk output (default: "
     << output_interval_default << ")\n";
  ss << tabstr
     << "-reset [positive integer] number of remeshing steps to allow before creating a new reference (default: disabled)\n";
  ss << tabstr
     <<  "-gmls [positive integer >= 2] order of polynomial approximation for GMLS remeshing (default: 5)\n";
  ss << tabstr << "-h Print help message and exit.\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Spherical transport with AMR:\n";
  auto tabstr = indent_string(1);
  ss << tabstr << "Initializing sphere mesh seed: " << seed_type::id_string()
     << " to uniform depth " << init_depth << "; amr is "
     << (amr_limit > 0 ? "" : "not ") << "enabled.\n";
  ss << tabstr << "dt = " << dt << " tfinal = " << tfinal << " nsteps = " << nsteps << "\n";
  if (amr_limit > 0) {
    ss << tabstr << "amr mass tol = " << tracer_mass_tol
       << "; amr var tol = " << tracer_var_tol << "\n";
  }
  if (remesh_interval > 0) {
    ss << tabstr << "remesh frequency is " << remesh_interval << "\n";
    ss << tabstr << "reset_lagrangian_interval is " << reset_lagrangian_interval << "\n";
    ss << tabstr << "gmls order = " << gmls_order << "\n";
  } else {
    ss << tabstr << "remeshing is disabled.\n";
  }
  if (output_interval > 0) {
    ss << tabstr << "output frequency is " << output_interval << "\n";
    ss << tabstr << "output files will be named: " << vtk_base_name
       << "????.vtk\n";
  } else {
    ss << tabstr << "output is disabled.\n";
  }
  return ss.str();
}
