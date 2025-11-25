#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif
#ifdef LPM_USE_NETCDF
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#endif
#include <sstream>
#include <string>
#include <memory>
#include <mpi.h>

using namespace Lpm;

void input_init(user::Input& input);

/**
  Choose a MeshSeed
*/
// typedef CubedSphereSeed SeedType;
typedef IcosTriSphereSeed SeedType;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Logger<> logger("sphere_mesh_init", Log::level::info, comm);
  ko::initialize(argc, argv);
  {
      user::Input input("swe_mesh_init");
      input_init(input);
      input.parse_args(argc, argv);
      if (input.help_and_exit) {
        logger.info(input.usage());
        Kokkos::finalize();
        MPI_Finalize();
        return 1;
      }

      MeshSeed<SeedType> seed;

      /**
      Set memory allocations
      */
      const int init_depth = input.get_option("tree_depth").get_int();
      Index nmaxverts;
      Index nmaxedges;
      Index nmaxfaces;
      seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, init_depth);

      /** Build the particle/panel mesh
      */
      auto sphere = std::make_unique<PolyMesh2d<SeedType>>(nmaxverts, nmaxedges, nmaxfaces);
      sphere->tree_init(init_depth, seed);
      sphere->update_device();

      logger.info(sphere->info_string());

      const std::string output_root = input.get_option("output_file_root").get_str() + SeedType::id_string() +
        std::to_string(init_depth);
#ifdef LPM_USE_VTK
      /** Output mesh to a vtk file */
      VtkPolymeshInterface<SeedType> vtk(*sphere);
      vtk.write(output_root + vtp_suffix());
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

void input_init(user::Input& input) {
  user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree depth", 4);
  input.add_option(tree_depth_option);

  user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("sphere_mesh"));
  input.add_option(output_file_root_option);
}
