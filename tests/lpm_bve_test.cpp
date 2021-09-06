#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_constants.hpp"
#include "lpm_bve_sphere.hpp"
#include "lpm_bve_sphere_impl.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_vorticity_gallery.hpp"
// #include "lpm_bve_rk4.hpp"
#include "util/lpm_timer.hpp"
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
    return 1;
  }

  logger.info(input.usage());
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

    /** Build the particle/panel mesh
      */
    auto sphere = std::shared_ptr<BVESphere<seed_type>>(new
      BVESphere<seed_type>(nmaxverts, nmaxedges, nmaxfaces));
    sphere->tree_init(input.init_depth, seed);
    sphere->update_device();

    sphere->set_omega(0);
    SolidBodyRotation relvort;
    sphere->init_vorticity(relvort);

    logger.info(sphere->info_string());

//
//     const auto dotprod_ind = sphere->create_tracer("u dot x");
//     ko::parallel_for(sphere->nvertsHost(), SphereVelocityTangentTestFunctor(sphere->tracer_verts[0],
//       sphere->physVerts.crds, sphere->velocityVerts));
//     ko::parallel_for(sphere->nfacesHost(), SphereVelocityTangentTestFunctor(sphere->tracer_faces[0],
//       sphere->physFaces.crds, sphere->velocityFaces));
//     const auto vorticity_err_ind = sphere->create_tracer("abs(vorticity_error)");
//
//     logger.info("courant number: {}", courant_number(input.dt, sphere->appx_mesh_size());
//
//     ko::View<Real*[3]> vert_velocity_error("vertex_velocity_error", sphere->nvertsHost());
//     ko::View<Real*[3]> face_velocity_error("face_velocity_error", sphere->nfacesHost());
//     ko::View<Real*[3]> vert_position_error("vertex_position_error", sphere->nvertsHost());
//     ko::View<Real*[3]> face_position_error("face_position_error", sphere->nfacesHost());
//
//     const Real tfinal = input.tfinal;
//     const Int ntimesteps = std::floor(tfinal/input.dt);
//     const Real dt = tfinal/ntimesteps;
//     const Real Omega = 2*PI;
//     BVERK4 solver(dt, Omega);
//     solver.init(sphere->nvertsHost(), sphere->nfacesHost());
//     auto vertex_policy = ko::TeamPolicy<>(solver.nverts, ko::AUTO());
//     auto face_policy = ko::TeamPolicy<>(solver.nfaces, ko::AUTO());
//     const Real dlam = sphere->avg_mesh_size_radians();
//     const Real cr = courant_number(dt, dlam);
//     std::cout << "Solid body rotation test\n";
//     std::cout << sphere->infoString();
//     std::cout << "\tavg mesh size = " << RAD2DEG * dlam << " degrees\n";
//     std::cout << "\tdt = " << dt << "\n";
//     std::cout << "\tcr. = " << cr << "\n";
//     if (cr > 0.5) {
//       std::ostringstream ss;
//       ss << "** warning ** cr = " << cr << " ; usually cr <= 0.5 is required.\n";
//       std::cout << ss.str();
//     }
//     std::cout << "\ttfinal = " << tfinal << "\n";
//
//     Timer output_timer("output");
//     output_timer.start();
//     {
//       std::ostringstream ss;
//       Polymesh2dVtkInterface<seed_type> vtk(sphere);
//       sphere->addFieldsToVtk(vtk);
//       vtk.addVectorPointData(vert_velocity_error);
//       vtk.addVectorPointData(vert_position_error);
//       vtk.addVectorCellData(face_velocity_error);
//       vtk.addVectorCellData(face_position_error);
//
//       ss << "tmp/" << input.case_name << seed_type::faceStr() << input.init_depth << "_dt" << dt << "_" << "0000.vtp";
//       vtk.write(ss.str());
//     }
//     output_timer.stop();
//
//
//     Timer single_solve_timer("single solve");
//     single_solve_timer.start();
//     {
//
//       ko::Profiling::pushRegion("initial solve");
//
//       const auto facex = sphere->physFaces.crds;
//       ko::View<Real*[3]> fexactvel("exact_velocity", sphere->nfacesHost());
//       ko::parallel_for("initial face vel. err.", sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
//         const auto myx = ko::subview(facex,i,ko::ALL());
//         fexactvel(i,0) = -Omega*myx(1);
//         fexactvel(i,1) =  Omega*myx(0);
//         fexactvel(i,2) = 0.0;
//       });
//       ErrNorms<> facevel_err(face_velocity_error, sphere->velocityFaces, fexactvel, sphere->faces.area);
//       std::cout << facevel_err.infoString("velocity error at t=0");
//
//       ko::Profiling::popRegion();
//     }
//     single_solve_timer.stop();
//
//     const Real est_solve_time = 4*ntimesteps*single_solve_timer.elapsed();
//     const Real est_total_time = est_solve_time + ntimesteps/input.output_interval*output_timer.elapsed();
//     std::cout << "estimated run time = " << est_total_time << " seconds (" << est_total_time/60 << " minutes)\n";
//
//     ko::Profiling::popRegion();
//
//     Real t;
//
//     init_timer.stop();
//
//     std::cout << init_timer.infoString();
//
//     ko::Profiling::pushRegion("main loop");
//     ProgressBar progress("SolidBodyRotation test", ntimesteps);
//     for (Int time_ind = 0; time_ind<ntimesteps; ++time_ind) {
//       solver.advance_timestep(sphere->physVerts.crds, sphere->relVortVerts, sphere->velocityVerts,
//         sphere->physFaces.crds, sphere->relVortFaces, sphere->velocityFaces, sphere->faces.area, sphere->faces.mask);
//
//       sphere->t = (time_ind+1)*dt;
//       t = sphere->t;
//
//       ko::Profiling::pushRegion("post-timestep solve");
//
//       ko::parallel_for("BVETest: vertex stream function", vertex_policy,
//         BVEVertexStreamFn(sphere->streamFnVerts, sphere->physVerts.crds,
//         sphere->physFaces.crds, sphere->relVortFaces, sphere->faces.area, sphere->faces.mask, sphere->nfacesHost()));
//       ko::parallel_for("BVETest: face stream function", face_policy, BVEFaceStreamFn(sphere->streamFnFaces, sphere->physFaces.crds,
//         sphere->relVortFaces, sphere->faces.area, sphere->faces.mask,sphere->nfacesHost()));
//       ko::parallel_for("BVETest: vertex velocity tangent", sphere->nvertsHost(),
//         SphereVelocityTangentTestFunctor(sphere->tracer_verts[0], sphere->physVerts.crds, sphere->velocityVerts));
//       ko::parallel_for("BVETest: face velocity tangent", sphere->nfacesHost(),
//         SphereVelocityTangentTestFunctor(sphere->tracer_faces[0], sphere->physFaces.crds, sphere->velocityFaces));
//
//       ko::Profiling::popRegion();
//
//       progress.update();
//
//       ko::Profiling::pushRegion("error computation");
//       {
//
//         ko::Profiling::pushRegion("error at vertices");
//
//         const auto relvort = sphere->relVortVerts;
//         const auto absvort = sphere->absVortVerts;
//         auto vorterr = sphere->tracer_verts[vorticity_err_ind];
//         const auto appxvel = sphere->velocityVerts;
//         const auto appxpos = sphere->physVerts.crds;
//         const auto lagpos = sphere->lagVerts.crds;
//         ko::parallel_for("vertex vorticity error", sphere->nvertsHost(), KOKKOS_LAMBDA (const Index& i) {
//           vorterr(i) = relvort(i) - absvort(i);
//           const auto myx = ko::subview(appxpos, i, ko::ALL());
//           vert_velocity_error(i,0) = appxvel(i,0) - (- Omega * myx(1));
//           vert_velocity_error(i,1) = appxvel(i,1) - (  Omega * myx(0));
//           vert_velocity_error(i,2) = appxvel(i,2);
//           const auto mya = ko::subview(lagpos, i, ko::ALL());
//           const Real cosomgt = std::cos(Omega*t);
//           const Real sinomgt = std::sin(Omega*t);
//           Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
//           for (Int j=0; j<3; ++j) {
//             vert_position_error(i,j) = myx(j) - exactpos[j];
//           }
//         });
//
//         ko::Profiling::popRegion();
//       }
//       {
//         ko::Profiling::pushRegion("error at faces");
//
//         const auto relvort = sphere->relVortFaces;
//         const auto absvort = sphere->absVortFaces;
//         auto vorterr = sphere->tracer_faces[vorticity_err_ind];
//         const auto appxvel = sphere->velocityFaces;
//         const auto appxpos = sphere->physFaces.crds;
//         const auto lagpos = sphere->lagFaces.crds;
//         ko::parallel_for("face vorticity error", sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
//           vorterr(i) = relvort(i) - absvort(i);
//           const auto myx = ko::subview(appxpos, i, ko::ALL());
//           face_velocity_error(i,0) = appxvel(i,0) - (- Omega * myx(1));
//           face_velocity_error(i,1) = appxvel(i,1) - (  Omega * myx(0));
//           face_velocity_error(i,2) = appxvel(i,2);
//           const auto mya = ko::subview(lagpos, i, ko::ALL());
//           const Real cosomgt = std::cos(Omega*t);
//           const Real sinomgt = std::sin(Omega*t);
//           Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
//           for (Int j=0; j<3; ++j) {
//             face_position_error(i,j) = myx(j) - exactpos[j];
//           }
//         });
//
//         ko::Profiling::popRegion();
//       }
//       ko::Profiling::popRegion();
//       if ( (time_ind+1)%input.output_interval == 0 || time_ind+1 == ntimesteps) {
//
//         ko::Profiling::pushRegion("vtk output");
//
//         Polymesh2dVtkInterface<seed_type> vtk(sphere);
//         sphere->addFieldsToVtk(vtk);
//         vtk.addVectorPointData(vert_velocity_error);
//         vtk.addVectorPointData(vert_position_error);
//         vtk.addVectorCellData(face_velocity_error);
//         vtk.addVectorCellData(face_position_error);
//
//         std::ostringstream ss;
//         ss << "tmp/" << input.case_name << seed_type::faceStr() << input.init_depth << "_dt" << dt << "_";
//         ss << std::setfill('0') << std::setw(4) << time_ind+1 <<  ".vtp";
//         vtk.write(ss.str());
//
//         ko::Profiling::popRegion();
//       }
//     }
//     {
//       ko::Profiling::pushRegion("final error norms at faces");
//
//       const auto facex = sphere->physFaces.crds;
//       ko::View<Real*[3]> fexactvel("exact_velocity", sphere->nfacesHost());
//       ko::parallel_for("final face vel. err.",sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
//         const auto myx = ko::subview(facex,i,ko::ALL());
//         fexactvel(i,0) = -Omega*myx(1);
//         fexactvel(i,1) =  Omega*myx(0);
//         fexactvel(i,2) = 0.0;
//       });
//       ErrNorms<> facevort_err(sphere->tracer_faces[vorticity_err_ind], sphere->absVortFaces, sphere->faces.area);
//       ErrNorms<> facevel_err(face_velocity_error, sphere->velocityFaces, fexactvel, sphere->faces.area);
//       ErrNorms<> facepos_err(face_position_error, sphere->physFaces.crds, sphere->lagFaces.crds, sphere->faces.area);
//       std::cout << facevort_err.infoString("vorticity error at t=tfinal");
//       std::cout << facevel_err.infoString("velocity error at t=tfinal");
//       std::cout << facepos_err.infoString("position error at t=tfinal");
//
//       ko::Profiling::popRegion();
//     }
//     ko::Profiling::popRegion();
//     total_timer.stop();
//     std::cout << total_timer.infoString();
  }
  std::cout << "tests pass" << std::endl;
  ko::finalize();
MPI_Finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  dt = 0.01;
  tfinal = 1.0;
  case_name = "bve_test";
  init_depth = 3;
  output_interval = 1;
  help_and_exit = false;
  for (Int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-d") {
      init_depth = std::stoi(argv[++i]);
    }
    else if (token == "-o") {
      case_name = argv[++i];
    }
    else if (token == "-dt") {
      dt = std::stod(argv[++i]);
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
  ss << "\t   " << "-o [output_filename_root] (default: unif_)\n";
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
