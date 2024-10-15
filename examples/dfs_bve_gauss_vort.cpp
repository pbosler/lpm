#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <cstdio>
#include <iomanip>
#include <iostream>

using namespace Lpm;

/** Test input.  Sets default parameter values, reads optional replacement
  values from the command line.
*/
struct Input {
  Input(int argc, char* argv[]);
  Real dt;  // time step
  Real tfinal; // final computation time
  std::string case_name; // test case name, will be the beginning of all output filenames.
  Int init_mesh_depth; // initial depth of Lagrangian particle/panel mesh quadtree
  Int nlon; // number of longitude points in the Fourier grid
  Int output_interval; // number of time steps between each output file

  std::string vtk_froot; // base filename for vtk output

  // write Input state to a string
  std::string info_string() const;
  // get a usage string for this program
  std::string usage() const;
  // if true, stop output a help message and stop running
  bool help_and_exit;

};

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


/** Computes vorticity error at each particle and panel.
*/
template <typename SeedType>
void compute_vorticity_error(scalar_view_type vert_err, scalar_view_type face_err,
  const DFS::DFSBVE<SeedType>& sph) {
  Kokkos::parallel_for("relative vorticity error (vertices)",
    sph.mesh.n_vertices_host(),
    RelVortError(vert_err,
      sph.abs_vort_passive.view, sph.rel_vort_passive.view,
      sph.mesh.vertices.phys_crds.view, sph.coriolis.Omega));
  Kokkos::parallel_for("relative vorticity error (faces)",
    sph.mesh.n_faces_host(),
    RelVortError(face_err,
      sph.abs_vort_active.view, sph.rel_vort_active.view,
      sph.mesh.faces.phys_crds.view, sph.coriolis.Omega));
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("dfs_bve", Log::level::info, comm);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
    /**
      program run
    */
    /**
      initialize problem
    */
    Input input(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    logger.info(input.info_string());
    const bool write_output = input.output_interval > 0;
    const Int nlon = input.nlon;
    const Int ntracers = 0;
    // problem types: vorticity definition
    GaussianVortexSphere vorticity_fn;

    // LPM particle/panel initialization
    typedef CubedSphereSeed seed_type;
    const Int mesh_depth = input.init_mesh_depth;
    PolyMeshParameters<seed_type> mesh_params(mesh_depth);
    const Int gmls_order = 4; // TODO: this can be an input parameter
    gmls::Params gmls_params(gmls_order);
    // DFS initialization
    DFS::DFSBVE<seed_type> sphere(mesh_params, nlon, gmls_params);
    sphere.init_vorticity(vorticity_fn);
    Real total_vorticity;
    Kokkos::parallel_reduce("set Gauss constant", sphere.mesh.n_faces_host(),
      TotalVorticity(sphere.rel_vort_active.view, sphere.mesh.faces.area),
      total_vorticity);
    vorticity_fn.set_gauss_const(total_vorticity);
    logger.info("total_vorticity = {}; gauss_const = {}", total_vorticity, vorticity_fn.gauss_const);
    sphere.init_vorticity(vorticity_fn);
      logger.info("rel vort active info: {}", sphere.rel_vort_active.info_string());
    sphere.init_velocity_from_vorticity();
    logger.info(sphere.info_string());

    ScalarField<VertexField> vert_rel_vort_error("relative_vorticity_error", sphere.mesh.n_vertices_host());
    ScalarField<FaceField> face_rel_vort_error("relative_vorticity_error", sphere.mesh.n_faces_host());
    compute_vorticity_error(vert_rel_vort_error.view, face_rel_vort_error.view, sphere);

    // Solver initialization
    const Real dt = input.dt;
    const Real tfinal = input.tfinal;
    const int nsteps = int(tfinal/dt);
    DFS::DFSRK2<seed_type> rk2_solver(dt, sphere);
   // DFS::DFSRK3<seed_type> rk3_solver(dt, sphere);
    //DFS::DFSRK2<seed_type> rk2_solver(dt, sphere);
    DFS::DFSRK4<seed_type> rk4_solver(dt, sphere);
    int output_ctr = 0;
    const std::string fname_root = "dfs_bve_gauss_vort_" + seed_type::id_string() + "_nlon" +
    std::to_string(nlon) + "_dt" + std::to_string(dt);
    input.vtk_froot = fname_root;
    // output initial data
    #ifdef LPM_USE_VTK
    {
      const std::string mesh_vtk_file = input.vtk_froot + "_" + zero_fill_str(output_ctr) + ".vtp";
      const std::string grid_vtk_file = input.vtk_froot + "_" + zero_fill_str(output_ctr) + ".vts";
      auto vtk_mesh = vtk_mesh_interface(sphere);
      auto vtk_grid = vtk_grid_interface(sphere);
      vtk_mesh.add_scalar_point_data(vert_rel_vort_error.view);
      vtk_mesh.add_scalar_cell_data(face_rel_vort_error.view);
      vtk_mesh.write(mesh_vtk_file);
      vtk_grid.write(grid_vtk_file);
      ++output_ctr;
    }
    #endif

    // timestepping loop
    for (int time_idx=0; time_idx<nsteps; ++time_idx) {
      sphere.advance_timestep(rk4_solver);
      compute_vorticity_error(vert_rel_vort_error.view, face_rel_vort_error.view, sphere);
      #ifdef LPM_USE_VTK
      {
        const std::string mesh_vtk_file = input.vtk_froot + "_" + zero_fill_str(output_ctr) + ".vtp";
        const std::string grid_vtk_file = input.vtk_froot + "_" + zero_fill_str(output_ctr) + ".vts";
        auto vtk_mesh = vtk_mesh_interface(sphere);
        auto vtk_grid = vtk_grid_interface(sphere);
        vtk_mesh.add_scalar_point_data(vert_rel_vort_error.view);
        vtk_mesh.add_scalar_cell_data(face_rel_vort_error.view);
        vtk_mesh.write(mesh_vtk_file);
        vtk_grid.write(grid_vtk_file);
        ++output_ctr;
      }
      #endif
      logger.info("t = {}", time_idx*dt);
    }

    const auto abs_vort_faces = sphere.abs_vort_active.view;
    const auto OMG = sphere.coriolis.Omega;
    const auto xyz_faces = sphere.mesh.faces.phys_crds.view;
    Kokkos::View<Real*> rel_vort_faces_exact("rel_vort_faces_exact", sphere.mesh.n_faces_host());
    Kokkos::parallel_for("compute exact rel_vort on LPM faces",
      sphere.mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        rel_vort_faces_exact(i) = abs_vort_faces(i) - 2*OMG*xyz_faces(i,2);
      });

    ErrNorms rel_vort_err(face_rel_vort_error.view, rel_vort_faces_exact, sphere.mesh.faces.area);
    logger.info("At t = {}, active panel error is:\n\t{}", sphere.t, rel_vort_err.info_string());

  }  // Kokkos scope
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
}

Input::Input(int argc, char* argv[]) {
  dt = 0.01;
  tfinal = 0.1;
  init_mesh_depth = 4;
  nlon = 40;
  output_interval = 0;
  help_and_exit = false;
  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-d" or token == "--depth") {
      init_mesh_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_mesh_depth>=0);
    }
    else if (token == "-nl" or token == "--nlon") {
      nlon = std::stoi(argv[++i]);
      LPM_REQUIRE((nlon > 0) and (nlon%2 == 0));
    }
    else if (token == "-o" or token == "--output-file-root") {
      case_name = argv[++i];
      vtk_froot = case_name;
    }
    else if (token == "-dt" or token == "--timestep") {
      dt = std::stod(argv[++i]);
      LPM_REQUIRE(dt > 0);
    }
    else if (token == "-tf" or token == "--tfinal") {
      tfinal = std::stod(argv[++i]);
      LPM_REQUIRE(tfinal >= 0);
    }
    else if (token == "-f" or token == "--output--frequency") {
      output_interval = std::stoi(argv[++i]);
    }
    else if (token == "-h" or token == "--help") {
      help_and_exit = true;
    }
  }
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "DFS BVE Test: This program solves the barotropic vorticity equation\n"
     << "on a rotating sphere using a set of Lagrangian particles and panels\n"
     << "for vorticity advection and a double Fourier series on a mesh to solve\n"
     << "for velocity.\n";
  ss << "\t" << "optional arguments:\n";
  ss << "\t\t" << "-d [--depth] initial depth of the Lagrangian panel quadtree\n"
     << "\t\t" << "-nl [--nlon] number of longitude points in DFS grid\n"
     << "\t\t" << "-o [--output-file-root] filename root for all output files\n"
     << "\t\t" << "-dt [--timestep] time step size\n"
     << "\t\t" << "-tf [--tfinal] final time of computation\n"
     << "\t\t" << "-f [--output-frequency] number of time steps between each output step\n"
     << "\t\t" << "-h [--help] display this message and exit\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Lpm DFS/BVE test input info:\n";
  ss << "\t" << "initial particle/panel quadtree depth : " << init_mesh_depth << "\n"
     << "\t" << "nlon : " << nlon << "\n"
     << "\t" << "filename root: " << case_name << "\n"
     << "\t\t" << "particle/panel data files will have a .vtp suffix;\n"
     << "\t\t" << "DFS grid data files will have a .vts suffix\n"
     << "\t" << "time step: " << dt << "\n"
     << "\t" << "tfinal : " << tfinal << "\n";
  return ss.str();
}
