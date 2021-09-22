#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_constants.hpp"
#include "lpm_sphere_functions.hpp"
#include "lpm_bve_sphere.hpp"
#include "lpm_bve_sphere_impl.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_bve_rk4.hpp"
#include "lpm_bve_rk4_impl.hpp"
#include "lpm_error.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_filename.hpp"
#include "util/lpm_progress_bar.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include "Kokkos_Core.hpp"

#include <iomanip>
#include <sstream>

using namespace Lpm;

inline Real courant_number(const Real& dt, const Real& dlam) {return 2*constants::PI * dt / dlam;}

typedef CubedSphereSeed seed_type;
//typedef IcosTriSphereSeed seed_type;

struct Input {
  Input(int argc, char* argv[]);

  Real dt;
  Real tfinal;
  std::string case_name;
  Int init_depth;
  Int output_interval;

  std::string info_string() const;

  std::string usage() const;

  bool help_and_exit;

  std::string vtk_froot;
  std::string nc_froot;
};

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("bve_sphere_test", Log::level::info, comm);

  Input input(argc, argv);
  if (input.help_and_exit) {
    std::cout << input.usage();
    return 1;
  }
  logger.info(input.info_string());

  ko::initialize(argc, argv);
  {

    ko::Profiling::pushRegion("initialization");

    Timer init_timer("initialization");
    init_timer.start();
    Timer total_timer("total");
    total_timer.start();

    MeshSeed<seed_type> seed;

    /**
    Set memory allocations
    */
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, input.init_depth);

    logger.debug("max allocations: {}, {}, {}", nmaxverts, nmaxedges, nmaxfaces);

    /**
    Build the particle/panel mesh
    */
    const std::vector<std::string> tracer_names = {"u_dot_x",
      "vorticity_error"};

    auto sphere = std::shared_ptr<BVESphere<seed_type>>(new
      BVESphere<seed_type>(nmaxverts, nmaxedges, nmaxfaces, tracer_names));
    sphere->tree_init(input.init_depth, seed);
    sphere->update_device();

    logger.debug("sphere allocated.");

    sphere->set_omega(0);
    SolidBodyRotation relvort;
    sphere->init_vorticity(relvort);
    sphere->init_velocity();
    sphere->init_stream_fn();
    logger.info(sphere->info_string());
    const Real cr = courant_number(input.dt, sphere->appx_mesh_size());
    if (cr > 1.0 ) {
      logger.error("courant number {} exceeds 1", cr);
      LPM_REQUIRE(cr < 1);
    }
    else {
      logger.info("courant number: {}", cr);
    }


    /**
    Compute initial error
    */
    ko::View<Real*[3]> vert_velocity_error("vertex_velocity_error", sphere->n_vertices_host());
    ko::View<Real*[3]> face_velocity_error("face_velocity_error", sphere->n_faces_host());
    ko::View<Real*[3]> vert_position_error("vertex_position_error", sphere->n_vertices_host());
    ko::View<Real*[3]> face_position_error("face_position_error", sphere->n_faces_host());
    ko::View<Real*> vert_stream_fn_error("vertex_streamfn_error", sphere->n_vertices_host());
    ko::View<Real*> face_stream_fn_error("face_streamfn_error", sphere->n_faces_host());

    const auto vertx = sphere->vertices.phys_crds->crds;
    const auto facex = sphere->faces.phys_crds->crds;
    const auto vert_vel = sphere->velocity_verts;
    const auto face_vel = sphere->velocity_faces;
    const auto OMG = SolidBodyRotation::OMEGA;
    const auto vert_absvort = sphere->abs_vort_verts;
    const auto face_absvort = sphere->abs_vort_faces;
    const auto vert_relvort = sphere->rel_vort_verts;
    const auto face_relvort = sphere->rel_vort_faces;
    const auto vert_stream_fn = sphere->stream_fn_verts;
    const auto face_stream_fn = sphere->stream_fn_faces;
    const auto verta = sphere->vertices.lag_crds->crds;
    const auto facea = sphere->faces.lag_crds->crds;

    Kokkos::parallel_for(sphere->n_vertices_host(), KOKKOS_LAMBDA (const Index i) {
      const auto mxyz = Kokkos::subview(vertx, i, Kokkos::ALL);
      const auto muvw = Kokkos::subview(vert_vel, i, Kokkos::ALL);
      vert_velocity_error(i, 0) = -OMG * mxyz(1) - muvw(0);
      vert_velocity_error(i, 1) =  OMG * mxyz(0) - muvw(1);
      vert_velocity_error(i, 2) = -muvw(2);
      vert_stream_fn_error(i) = vert_stream_fn(i) - 2*constants::PI*mxyz(2);
    });

    Kokkos::parallel_for(sphere->n_faces_host(), KOKKOS_LAMBDA (const Index i) {
      const auto mxyz = Kokkos::subview(facex, i, Kokkos::ALL);
      const auto muvw = Kokkos::subview(face_vel, i, Kokkos::ALL);
      face_velocity_error(i, 0) = -OMG * mxyz(1) - muvw(0);
      face_velocity_error(i, 1) =  OMG * mxyz(0) - muvw(1);
      face_velocity_error(i, 2) = -muvw(2);
      face_stream_fn_error(i) = face_stream_fn(i) - 2*constants::PI*mxyz(2);
    });

    Kokkos::parallel_for(sphere->n_vertices_host(),
       SphereTangentFunctor(sphere->tracer_verts[0],
        sphere->vertices.phys_crds->crds,
        sphere->velocity_verts));

    Kokkos::parallel_for(sphere->n_faces_host(),
      SphereTangentFunctor(sphere->tracer_faces[0],
        sphere->faces.phys_crds->crds,
        sphere->velocity_faces));

    BaseFilename<seed_type> fname(input.case_name, input.init_depth, input.dt);


    int frame = 0;
    if (input.output_interval > 0) {
      /**
      Output initial data
      */
      const std::string vtkfname = fname(frame) + fname.vtp_suffix();
      logger.info("Initialization complete. Writing initial data to file {}.", vtkfname);
      VtkPolymeshInterface<seed_type> vtk = vtk_interface(sphere);
      vtk.add_scalar_point_data(vert_stream_fn_error, "stream_fn_error");
      vtk.add_scalar_cell_data(face_stream_fn_error, "stream_fn_error");
      vtk.add_vector_point_data(vert_velocity_error, "velocity_error");
      vtk.add_vector_cell_data(face_velocity_error, "velocity_error");
      vtk.add_vector_point_data(vert_position_error, "position_error");
      vtk.add_vector_cell_data(face_position_error, "position_error");
      vtk.write(vtkfname);
    }

    init_timer.stop();

    logger.info(init_timer.info_string());

    /**

        Time stepping

    */

    ko::TeamPolicy<> vertex_policy(sphere->n_vertices_host(), ko::AUTO());
    ko::TeamPolicy<> face_policy(sphere->n_faces_host(), ko::AUTO());

    const Real tfinal = input.tfinal;
    const Int ntimesteps = std::floor(tfinal/input.dt);
    const Real dt = tfinal/ntimesteps;
    BVERK4 solver(dt, sphere);

    ko::Profiling::pushRegion("main loop");
    ProgressBar progress("SolidBodyRotation test", ntimesteps);
    for (Int time_ind = 0; time_ind<ntimesteps; ++time_ind) {
      solver.advance_timestep(sphere);
       sphere->t = (time_ind+1)*dt;
       const Real t = sphere->t;
      ko::Profiling::pushRegion("post-timestep solve");

      ko::parallel_for("BVETest: vertex stream function", vertex_policy,
        BVEVertexStreamFn(sphere->stream_fn_verts, sphere->vertices.phys_crds->crds,
        sphere->faces.phys_crds->crds, face_relvort, sphere->faces.area, sphere->faces.mask, sphere->n_faces_host()));
      ko::parallel_for("BVETest: face stream function", face_policy,
        BVEFaceStreamFn(sphere->stream_fn_faces, sphere->faces.phys_crds->crds,
        face_relvort, sphere->faces.area, sphere->faces.mask,sphere->n_faces_host()));
      ko::parallel_for("BVETest: vertex velocity tangent", sphere->n_vertices_host(),
        SphereTangentFunctor(sphere->tracer_verts[0], sphere->vertices.phys_crds->crds, vert_vel));
      ko::parallel_for("BVETest: face velocity tangent", sphere->n_faces_host(),
        SphereTangentFunctor(sphere->tracer_faces[0], sphere->faces.phys_crds->crds, face_vel));

      ko::Profiling::popRegion();


      ko::Profiling::pushRegion("error computation");

      const auto vort_err_verts = sphere->tracer_verts[1];
      ko::parallel_for("vertex error", sphere->n_vertices_host(), KOKKOS_LAMBDA (const Index& i) {
        vort_err_verts(i) = vert_relvort(i) - vert_absvort(i);
        const auto myx = ko::subview(vertx, i, ko::ALL());
        vert_velocity_error(i,0) = vert_vel(i,0) - (- OMG * myx(1));
        vert_velocity_error(i,1) = vert_vel(i,1) - (  OMG * myx(0));
        vert_velocity_error(i,2) = vert_vel(i,2);
        const auto mya = ko::subview(verta, i, ko::ALL());
        const Real cosomgt = std::cos(OMG*t);
        const Real sinomgt = std::sin(OMG*t);
        Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
        for (Int j=0; j<3; ++j) {
          vert_position_error(i,j) = myx(j) - exactpos[j];
        }
        vert_stream_fn_error(i) = vert_stream_fn(i) - 2*constants::PI*myx(2);
      });


      const auto vort_err_faces = sphere->tracer_faces[1];
      ko::parallel_for("face error", sphere->n_faces_host(), KOKKOS_LAMBDA (const Index& i) {
        vort_err_faces(i) = face_relvort(i) - face_absvort(i);
        const auto myx = ko::subview(facex, i, ko::ALL());
        face_velocity_error(i,0) = face_vel(i,0) - (- OMG * myx(1));
        face_velocity_error(i,1) = face_vel(i,1) - (  OMG * myx(0));
        face_velocity_error(i,2) = face_vel(i,2);
        const auto mya = ko::subview(facea, i, ko::ALL());
        const Real cosomgt = std::cos(OMG*t);
        const Real sinomgt = std::sin(OMG*t);
        Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
        for (Int j=0; j<3; ++j) {
          face_position_error(i,j) = myx(j) - exactpos[j];
        }
        face_stream_fn_error(i) = face_stream_fn(i) - 2*constants::PI*myx(2);
      });

      ko::Profiling::popRegion();

      if (input.output_interval > 0) {
        if ( (time_ind+1)%input.output_interval == 0 || time_ind+1 == ntimesteps) {
          ko::Profiling::pushRegion("vtk output");

          const std::string vtkfname = fname(++frame) + fname.vtp_suffix();
          VtkPolymeshInterface<seed_type> vtk = vtk_interface(sphere);
          vtk.add_scalar_point_data(vert_stream_fn_error, "stream_fn_error");
          vtk.add_scalar_cell_data(face_stream_fn_error, "stream_fn_error");
          vtk.add_vector_point_data(vert_velocity_error, "velocity_error");
          vtk.add_vector_cell_data(face_velocity_error, "velocity_error");
          vtk.add_vector_point_data(vert_position_error, "position_error");
          vtk.add_vector_cell_data(face_position_error, "position_error");
          vtk.write(vtkfname);
          ko::Profiling::popRegion();
        }
      }

        progress.update();

    }

    Kokkos::View<Real*> fexactpsi("face_exact_stream_fn", sphere->n_faces_host());
    Kokkos::View<Real*[3]> fexactvel("face_exact_velocity", sphere->n_faces_host());
    Kokkos::parallel_for(sphere->n_faces_host(), KOKKOS_LAMBDA (const Index i) {
      const auto myx = Kokkos::subview(facex, i, Kokkos::ALL);
      fexactvel(i, 0) = - OMG * myx(1);
      fexactvel(i, 1) =   OMG * myx(0);
      fexactvel(i, 2) = 0;
      fexactpsi(i) = 2*constants::PI*myx(2);
    });

    ErrNorms<> facevort_err(sphere->tracer_faces[1], face_absvort, sphere->faces.area);
    ErrNorms<> facevel_err(face_velocity_error, face_vel, fexactvel, sphere->faces.area);
    ErrNorms<> facepos_err(face_position_error, facex, facea, sphere->faces.area);
    ErrNorms<> facepsi_err(face_stream_fn_error, fexactpsi, sphere->faces.area);

    logger.info("tfinal (stream fn): {}", facepsi_err.info_string());
    logger.info("tfinal (vorticity): {}", facevort_err.info_string());
    logger.info("tfinal (velocity):  {}", facevel_err.info_string());
    logger.info("tfinal (position):  {}", facepos_err.info_string());

    total_timer.stop();
    logger.info(total_timer.info_string());

  }
  ko::finalize();
  MPI_Finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  dt = 0.01;
  tfinal = 0.03;
  case_name = "bve_test";
  init_depth = 3;
  output_interval = 0;
  help_and_exit = false;
  for (Int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-d") {
      init_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_depth >= 0);
    }
    else if (token == "-o") {
      case_name = argv[++i];
    }
    else if (token == "-dt") {
      dt = std::stod(argv[++i]);
      LPM_REQUIRE(dt > 0);
    }
    else if (token == "-tf") {
      tfinal = std::stod(argv[++i]);
    }
    else if (token == "-f") {
      output_interval = std::stoi(argv[++i]);
    }
    else if (token == "-h") {
      help_and_exit = true;
    }
  }
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "BVE Test: This program initializes a uniform spherical mesh \n" <<
    "and solves the barotropic vortcity equation (BVE) on that mesh for a problem\n" <<
    "whose exact solution is known, then computes error.\n" <<
    "output is written the mesh to data files in 2 formats: \n\tVTK's .vtp format and the NetCDF4 .nc format.\n";
  ss << "\t" << "optional arguments:\n";
  ss << "\t   " << "-o [output_filename_root] (default: bve_test)\n";
  ss << "\t   " << "-d [nonnegative integer] (default: 3); defines the initial depth of the uniform mesh's face quadtree.\n";
  ss << "\t   " << "-dt [positive real number] (default: 0.01) time step size.\n";
  ss << "\t   " << "-tf [positive real number] (default: 1.0) final time of simulation.\n";
  ss << "\t   " << "-f  [positive integer] (default: 1) output i/o interval in units of time steps.\n";
  ss << "\t   " << "-h  Print help message and exit.\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Sphere mesh init:\n";
  ss << "\tInitializing from seed: " << seed_type::id_string() << "\n";
  ss << "\tTo uniform tree depth: " << init_depth << "\n";
  return ss.str();
}
