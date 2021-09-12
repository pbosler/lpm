#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include <sstream>
#include <string>
#include <memory>
#include <mpi.h>

using namespace Lpm;

struct Input {
  int init_depth;
  std::string ofroot;
  std::string vtk_fname;
  std::string nc_fname;

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

  Logger<> logger("sphere_mesh_init", Log::level::info, comm);

  Input input(argc, argv);
  if (input.help_and_exit) {
    return 1;
  }

  logger.info(input.usage());
  logger.info(input.info_string());

  ko::initialize(argc, argv);
  {
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
      auto sphere = std::shared_ptr<PolyMesh2d<seed_type>>(new
        PolyMesh2d<seed_type>(nmaxverts, nmaxedges, nmaxfaces));
      sphere->tree_init(input.init_depth, seed);
      sphere->update_device();

      logger.info(sphere->info_string());

      /** Output mesh to a vtk file */
      VtkPolymeshInterface<seed_type> vtk(sphere);
      vtk.write(input.vtk_fname);

      /** Output mesh to a netCDF file */
      NcWriter<SphereGeometry> nc(input.nc_fname);
      nc.define_polymesh(*sphere);
      logger.debug(nc.info_string());
  }

  ko::finalize();
  MPI_Finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  init_depth = 2;
  ofroot = "unif_";
  help_and_exit = false;

  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-o") {
      vtk_fname = argv[++i];
    }
    else if (token == "-n") {
      init_depth = std::stoi(argv[++i]);
    }
    else if (token == "-h") {
      std::cout << usage() << "\n";
      help_and_exit = true;
    }
  }

  ofroot += (std::is_same<CubedSphereSeed, seed_type>::value ?
    "cubed_sphere" : "icos_tri_sphere") + std::to_string(init_depth);

  vtk_fname = ofroot + ".vtp";
  nc_fname = ofroot + ".nc";
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "Sphere Mesh Init: This program initializes a uniform spherical mesh \n" <<
    "and writes the mesh to data files in 2 formats: \n\tVTK's .vtp format and the NetCDF4 .nc format.\n";
  ss << "\t" << "optional arguments:\n";
  ss << "\t   " << "-o [output_filename_root] (default: unif_)\n";
  ss << "\t   " << "-n [nonnegative integer] (default: 2); defines the initial depth of the uniform mesh's face quadtree.\n";
  return ss.str();
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Sphere mesh init:\n";
  ss << "\tInitializing from seed: " << seed_type::id_string() << "\n";
  ss << "\tTo uniform tree depth: " << init_depth << "\n";
  ss << "\tvtk data saved in file: " << vtk_fname << "\n";
  ss << "\tnetCDF data saved in file: " << nc_fname << "\n";
  return ss.str();
}

