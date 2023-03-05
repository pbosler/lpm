#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_2d_transport_rk4.hpp"
#include "lpm_2d_transport_rk4_impl.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdio>

using namespace Lpm;

/** @brief Returns the approximate Courant number using eqn. (24)
of Lauritzen et al., 2012,  A standard test case suite for two-dimensional linear transport on the sphere, Geosci. Model Dev. 5:887-901.
*/
inline Real courant_number(const Real dt, const Real dlam)  {
  constexpr Real umax = 3.26;
  return dt * umax / dlam;
}

typedef CubedSphereSeed seed_type;
// typedef IcosTriSphereSeed seed_type;
typedef LauritzenEtAlDeformationalFlow velocity_type;
typedef SphericalGaussianHills tracer_type;

struct Input {
  Input(int argc, char* argv[]);
  Real dt;
  Real tfinal;
  std::string base_output_name;
  Int init_depth;
  Int amr_limit;
  Int amr_max;
  Int remesh_interval;
  Int output_interval;
  std::string output_dir;
  Real tracer_mass_tol;
  Real tracer_var_tol;
  Real radius;

  bool help_and_exit;
  std::string info_string() const;
  std::string usage() const;
  std::string vtk_base_name;
};

int main(int argc, char* argv[]) {
  /*
    program setup
  */
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("sphere transport amr", Log::level::debug, comm);

  Input input(argc, argv);
  if (input.help_and_exit) {
    std::cout << input.usage();
    return 1;
  }
  logger.info(input.info_string());
  const bool amr = input.amr_limit > 0;
  const bool write_output = input.output_interval > 0;

  Kokkos::initialize(argc, argv);
{
  /*
    initialize problem
  */

  tracer_type tracer;
  auto mesh_params = std::make_shared<PolyMeshParameters<seed_type>>(
    input.init_depth, input.radius, input.amr_limit);
  auto sphere = std::make_shared<TransportMesh2d<seed_type>>(mesh_params);
  sphere->template initialize_velocity<velocity_type>();
  sphere->initialize_tracer(tracer);

  if (amr) {
    Real max_tracer_mass;
    auto face_area = sphere->faces.area;
    auto face_tracer = sphere->tracer_faces.at(tracer.name()).view;
    auto face_mask = sphere->faces.mask;
    Kokkos::parallel_reduce(sphere->n_faces_host(),
      KOKKOS_LAMBDA (const Index i, Real& m) {
        if (!face_mask(i)) {
          m = (face_tracer(i) * face_area(i) > m ? face_tracer(i) * face_area(i) : m);
        }
      }, Kokkos::Max<Real>(max_tracer_mass));

    logger.info("max_tracer_mass per face = {}", max_tracer_mass);
    const Real tracer_mass_tol = convert_to_absolute_tol(input.tracer_mass_tol, max_tracer_mass);
    logger.info("input relative tracer mass tol {} converts to absolute tracer mass tol {}",
      input.tracer_mass_tol, tracer_mass_tol);

    Real max_tracer_var;
    auto vert_tracer = sphere->tracer_verts.at(tracer.name()).view;
    auto face_verts = sphere->faces.verts;
    Kokkos::parallel_reduce(sphere->n_faces_host(),
      KOKKOS_LAMBDA (const Index i, Real& mvar) {
        if (!face_mask(i)) {
          Real minval = face_tracer(i);
          Real maxval = face_tracer(i);
          for (int j=0; j<seed_type::nfaceverts; ++j) {
            const auto vidx = face_verts(i,j);
            if (vert_tracer(vidx) < minval) minval = vert_tracer(vidx);
            if (vert_tracer(vidx) > maxval) maxval = vert_tracer(vidx);
          }
          auto var = maxval - minval;
          mvar = ( var > mvar ? var : mvar);
        }
      }, Kokkos::Max<Real>(max_tracer_var));
    logger.info("max_tracer_var per face = {}", max_tracer_var);
    const Real tracer_var_tol = convert_to_absolute_tol(input.tracer_var_tol, max_tracer_var);
    logger.info("input relative tracer var tol {} converts to absolute tol {}",
      input.tracer_var_tol, tracer_var_tol);

    Kokkos::View<bool*> flags("refinement_flags", sphere->n_faces_host());


    Index verts_start_idx = 0;
    Index faces_start_idx = 0;
    for (int i=0; i<input.amr_max; ++i) {

      Index verts_end_idx = sphere->n_vertices_host();
      Index faces_end_idx = sphere->n_faces_host();


      Kokkos::parallel_for(Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
      ScalarIntegralFlag(flags, sphere->tracer_faces.at(tracer.name()).view, sphere->faces.area, sphere->faces.mask, tracer_mass_tol));
    Index mass_refinement_count;
    Kokkos::parallel_reduce(sphere->n_faces_host(),
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += Index(flags(i));
      }, mass_refinement_count);
    logger.info("amr iteration {}: initial mass_refinement_count = {}", i, mass_refinement_count);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx),
      ScalarVariationFlag(flags, sphere->tracer_faces.at(tracer.name()).view, sphere->tracer_verts.at(tracer.name()).view, sphere->faces.verts, sphere->faces.mask, tracer_var_tol));
    Index total_refinement_count;
    Kokkos::parallel_reduce(sphere->n_faces_host(),
      KOKKOS_LAMBDA(const Index i, Index& ct) {
        ct += Index(flags(i));
      }, total_refinement_count);

    logger.info("amr iteration {}: variation_refinement_count = {}", i, total_refinement_count - mass_refinement_count);

      sphere->divide_flagged_faces(flags, logger);
      Kokkos::deep_copy(flags, false);
      verts_start_idx = verts_end_idx;
      faces_start_idx = faces_end_idx;
      sphere->initialize_tracer(tracer, verts_start_idx, faces_start_idx);

    }
  }

  logger.info(sphere->info_string());
  int frame_counter = 0;
  #ifdef LPM_USE_VTK
    if (write_output) {
      VtkPolymeshInterface<seed_type> vtk = vtk_interface(sphere);
      vtk.write(input.vtk_base_name + zero_fill_str(frame_counter++) + vtp_suffix());
    }
  #endif
}
  /*
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
}

Input::Input(int argc, char* argv[]) {
  dt = 0.03;
  tfinal = 5;
  base_output_name = "sphere_transport_amr_";
  init_depth = 4;
  amr_limit = 0;
  amr_max = 1;
  tracer_mass_tol = 0.1;
  tracer_var_tol = 0.1;
  remesh_interval = 20;
  output_interval = 1;
  output_dir = "";
  help_and_exit = false;
  radius = 1;
  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-dt") {
      dt = std::stod(argv[++i]);
      LPM_REQUIRE(dt > 0);
    }
    else if (token == "-tf") {
      tfinal = std::stod(argv[++i]);
      LPM_REQUIRE(tfinal >= 0);
    }
    else if (token == "-o") {
      base_output_name = argv[++i];
    }
    else if (token == "-d") {
      init_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_depth >= 0);
    }
    else if (token == "-amr") {
      amr_limit = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_limit >= 0);
    }
    else if (token == "-amr_max") {
      amr_max = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_max >= 0);
    }
    else if (token == "-mass_tol") {
      tracer_mass_tol = std::stod(argv[++i]);
      LPM_REQUIRE(tracer_mass_tol > 0);
    }
    else if (token == "-var_tol") {
      tracer_var_tol = std::stod(argv[++i]);
      LPM_REQUIRE(tracer_var_tol > 0);
    }
    else if (token == "-rf") {
      remesh_interval = std::stoi(argv[++i]);
      LPM_REQUIRE(remesh_interval > 0 or remesh_interval == LPM_NULL_IDX);
    }
    else if (token == "-of") {
      output_interval = std::stoi(argv[++i]);
      LPM_REQUIRE(output_interval > 0 or output_interval == LPM_NULL_IDX);
    }
    else if (token == "-dir") {
      output_dir = argv[++i];
    }
    else if (token == "-h") {
      help_and_exit = true;
    }
  }
  vtk_base_name = output_dir + (output_dir.empty() ? "" : "/") +  base_output_name + seed_type::id_string() + "_d" + std::to_string(init_depth) +"_";
  if (amr_limit >0) {
    vtk_base_name += "amr" + std::to_string(amr_limit);
  }
  const char* fmt = "dt%.2f";
  int sz = std::snprintf(nullptr, 0, fmt, dt);
  std::vector<char> buf(sz+1);
  std::snprintf(&buf[0], buf.size(), fmt, dt);
  vtk_base_name += std::string(&buf[0], sz);
  vtk_base_name += "_rm" + std::to_string(remesh_interval) + "_";
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "Spherical transport with AMR:\n \tThis program demonstrates adaptive refinement in a spherical transport problem, solving test case 1 from Laurizten et al., 2012, A standard test case suite for two-dimensional linear transport on the sphere, Geosci. Model Dev. 5.\n The test uses a reversible velocity field with period T=5, so that the exact solution of the tracer at t=T matches the initial condition at t=0.\n";
  auto tabstr = indent_string(1);
  ss << tabstr << "optional arguments:\n";
  ss << tabstr << "-dt [nonnegative real number] time step size (default: 0.03)\n";
  ss << tabstr << "-tf [nonnegative real number] final time for integration (default: 5)\n";
  ss << tabstr << "-o [string] output filename root (default: \"sphere_transport_amr_example\"\n";
  ss << tabstr << "-d [nonnegative integer] initial depth of mesh quadtree (default: 4)\n";
  ss << tabstr << "-amr [nonnegative integer] number of uniform refinements beyond initial depth to allocate memory for; values > 0 will enable adaptive refinement (default: 0)\n";
  ss << tabstr << "-amr_max [nonnegative integer] maximum number of times a panel may be divided (default: 1)\n.";
  ss << tabstr << "-mass_tol [positive real number] threshold for local tracer integral refinement criterion; not used if amr = 0 (default: 0.1)\n";
  ss << tabstr << "-var_tol [positive real number] threshold for local tracer variation refinement criterion; not used if amr = 0 (default: 0.15).\n";
  ss << tabstr << "-rf [positive integer or -1] frequency of remesh/remap interpolations; setting value to -1 will disable remeshing (default: 20)\n";
  ss << tabstr << "-of [positive integer or -1] frequency of vtk output; setting value to -1 will disable vtk output (default: 1)\n";
  ss << tabstr << "-h Print help message and exit.\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Spherical transport with AMR:\n";
  auto tabstr = indent_string(1);
  ss << tabstr << "Initializing sphere mesh seed: " << seed_type::id_string() << " to uniform depth " << init_depth << "; amr is " << (amr_limit > 0 ? "" : "not ") << "enabled.\n";
  if (amr_limit > 0) {
    ss << tabstr << "amr mass tol = " << tracer_mass_tol << "; amr var tol = " <<  tracer_var_tol << "\n";
  }
  if (remesh_interval > 0) {
    ss << tabstr << "remesh frequency is " << remesh_interval << "\n";
  }
  else {
    ss << tabstr << "remeshing is disabled.\n";
  }
  if (output_interval > 0) {
    ss << tabstr << "output frequency is " << output_interval << "\n";
    ss << tabstr << "output files will be named: " << vtk_base_name << "????.vtk\n";
  }
  else {
    ss << tabstr << "output is disabled.\n";
  }
  return ss.str();
}
