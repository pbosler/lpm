#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#ifdef LPM_USE_NETCDF

#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_compadre.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif

#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "netcdf/lpm_netcdf_reader.hpp"
#include "netcdf/lpm_netcdf_reader_impl.hpp"

#include <sstream>
#include <string>
#include <memory>
#include <mpi.h>

using namespace Lpm;

struct Input {
  static constexpr int init_depth_default = 2;
  int init_depth;
  static constexpr int amr_refinement_limit_default = 0;
  int amr_refinement_limit;
  static constexpr int amr_refinement_buffer_default = 0;
  int amr_refinement_buffer;
  int source_timestep;
  Real scalar_max_tol;
  Real scalar_integral_tol;
  Real scalar_var_tol;
  std::string src_filename;
  std::string ofroot;
  std::string vtk_fname;
  std::string nc_fname;
  std::string field_name;

  Input(int argc, char* argv[]);

  std::string info_string() const;

  std::string usage() const;

  bool help_and_exit;
};

/**
  Choose a MeshSeed
*/
typedef CubedSphereSeed seed_type;
// typedef IcosTriSphereSeed seed_type;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  LPM_REQUIRE_MSG(comm.size() == 1, "Distributed data not implemented.");
  ko::initialize(argc, argv);
  {
    Logger<> logger("sphere_mesh_from_data", Log::level::info, comm);

    Input input(argc, argv);
    if (input.help_and_exit or input.src_filename.empty()) {
      return 1;
    }
    logger.info(input.info_string());

    /**
        Load data from source file
    */
    UnstructuredNcReader<SphereGeometry> reader(input.src_filename);
    logger.info(reader.info_string());

    Coords<SphereGeometry> src_coords = reader.create_coords();
    logger.info(src_coords.info_string());
    const int time_idx = 0;
    const auto src_data = reader.create_scalar_field(input.field_name, time_idx);


    /**
        Setup interpolation from source data
    */


    /**
        Initialize target mesh
    */
    MeshSeed<seed_type> seed;

    /**
    Set memory allocations
    */
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, input.init_depth + input.amr_refinement_buffer);

    /** Build the particle/panel mesh
    */
    auto sphere = std::make_unique<PolyMesh2d<seed_type>>(nmaxverts, nmaxedges, nmaxfaces);
    sphere->tree_init(input.init_depth, seed);
    sphere->update_device();

    logger.info(sphere->info_string());

#ifdef LPM_USE_VTK
    /** Output mesh to a vtk file */
    VtkPolymeshInterface<seed_type> vtk(*sphere);
    vtk.write(input.vtk_fname);
#endif
#ifdef LPM_USE_NETCDF
// TODO
//       * Output mesh to a netCDF file */
//       NcWriter<SphereGeometry> nc(input.nc_fname);
//       nc.define_polymesh(*sphere);
//       logger.debug(nc.info_string());
#endif
  }

  ko::finalize();
  MPI_Finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  init_depth = init_depth_default;
  amr_refinement_buffer = amr_refinement_buffer_default;
  amr_refinement_limit = amr_refinement_limit_default;
  scalar_max_tol = 0;
  scalar_integral_tol = 0;
  scalar_var_tol = 0;
  ofroot = "sph_from_data_";
  help_and_exit = false;
  if (argc < 2) {
    src_filename = "";
  }
  else {
    src_filename = argv[1];
  }
  for (int i=2; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-o") {
      vtk_fname = argv[++i];
    }
    else if (token == "-d" or token == "--init-depth") {
      init_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_depth >= 0);
    }
    else if (token == "-h") {
      std::cout << usage() << "\n";
      help_and_exit = true;
    }
    else if (token == "-i") {
      src_filename = argv[++i];
    }
    else if (token == "-amr") {
      amr_refinement_buffer = std::stoi(argv[++i]);
      amr_refinement_limit = amr_refinement_buffer;
      LPM_REQUIRE(amr_refinement_buffer >= 0);
    }
    else if (token == "--amr-limit") {
      amr_refinement_limit = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_refinement_limit >= 0);
    }
    else if (token == "--amr-buffer") {
      amr_refinement_buffer = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_refinement_buffer >= 0);
    }
    else if (token == "--max-tol") {
      scalar_max_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_max_tol > 0);
    }
    else if (token == "--integral-tol") {
      scalar_integral_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_integral_tol > 0);
    }
    else if (token == "--var-tol") {
      scalar_var_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_var_tol > 0);
    }
    else if (token == "-f" or token == "--field-name") {
      field_name = argv[++i];
    }
    else if (token == "-t") {
      source_timestep = std::stoi(argv[++i]);
    }
  }
  bool amr_valid = ( (amr_refinement_buffer == 0 and amr_refinement_limit == 0)
                  or (amr_refinement_buffer > 0 and amr_refinement_limit > 0) );
  LPM_REQUIRE_MSG(amr_valid, "valid input for amr requires both refinement buffer and refinement limit to be zero (no amr) or to be positive");
  ofroot += (std::is_same<CubedSphereSeed, seed_type>::value ?
    "cubed_sphere" : "icos_tri_sphere") + std::to_string(init_depth)
      + (amr_refinement_buffer > 0 ? "_amr" + std::to_string(amr_refinement_buffer) : "_unif");

  vtk_fname = ofroot + ".vtp";
  nc_fname = ofroot + ".nc";
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "Sphere Mesh from data: This program initializes a uniform spherical mesh \n" <<
    "and writes the mesh to data files in 2 formats: \n\tVTK's .vtp format and the NetCDF4 .nc format.\n";
  ss << "\t" << "first argument, or '-i' option: source data file.\n";
  ss << "\t" << "optional arguments:\n";
  ss << "\t   " << "-o [output_filename_root] \n";
  ss << "\t   " << "-i [filename] source data file path\n";
  ss << "\t   " << "-d [nonnegative integer] ; defines the initial depth of the uniform mesh's face quadtree.\n";
  ss << "\t   " << "-amr [nonnegative integer]; sets both amr buffer and amr limit\n";
  ss << "\t   " << "--amr-limit [nonnegative integer]; sets the maximum number of times a panel may be refined relative to its immediate neighbors; --amr-buffer must be positive whenever this option is positive";
  ss << "\t   " << "--amr-buffer [nonnegative integer]; sets maximum memory limit for the particle/panel mesh to be a uniform mesh of depth init-depth + buffer.  --amr-limit must be positive whenever --amr-buffer is positive\n";
  ss << "\t   " << "--max-tol sets maximum value threshold for scalar data amr\n";
  ss << "\t   " << "--integral-tol sets maximum integral threshold for scalar data amr\n";
  ss << "\t   " << "--var-tol sets maximum variation threshold for scalar data amr\n";
  ss << "\t   " << "-h print help message and exit\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Sphere mesh from data:\n";
  ss << "\tsource data file: " << src_filename << "\n";
  ss << "\toutput vtk file: " << vtk_fname << "\n";
  ss << "\toutput netcdf file: " << nc_fname << "\n";
  ss << "\tInitializing from seed: " << seed_type::id_string() << "\n";
  ss << "\tTo depth: " << init_depth << "\n";
  if (amr_refinement_buffer > 0) {
    ss << "\tAMR:\n";
    ss << "\t\t(buffer, limit) = (" << amr_refinement_buffer << ", " << amr_refinement_limit << ")\n";
    ss << "\t\tmax_tol = " << scalar_max_tol << "\n";
    ss << "\t\tintegral_tol = " << scalar_integral_tol << "\n";
    ss << "\t\tvar_tol = " << scalar_var_tol << "\n";
  }

  return ss.str();
}

#endif // LPM_USE_NETCDF
