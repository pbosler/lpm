#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2d.hpp"

using namespace Lpm;

struct Input {
  int init_depth;
  std::string vtk_fname;

  Input(int argc, char* argv[]);

};

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    Input input(argc, argv);
    LPM_THROW_IF( input.init_depth < 0 || input.init_depth > 9,
       "invalid initial  mesh tree depth.");

    /**
      Choose a MeshSeed
    */
    typedef CubedSphereSeed seed_type;
    //typedef IcosTriSphereSeed seed_type;
    MeshSeed<seed_type> seed;

    /**
    Set memory allocations
    */
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, input.init_depth);

    /** Build the particle/panel mesh
    */
    PolyMesh2d<seed_type> sphere(nmaxverts, nmaxedges, nmaxfaces);
    sphere.treeInit(input.init_depth, seed);
    sphere.updateDevice();

    std::cout << sphere.infoString("sphere_mesh_example",0,false);

    /** Output mesh to a vtk file */
    sphere.outputVtk(input.vtk_fname);
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  init_depth = 2;
  vtk_fname = "sphere_mesh_example.vtk";

  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-o") {
      vtk_fname = argv[++i];
    }
    else if (token == "-n") {
      init_depth = std::stoi(argv[++i]);
    }
  }
}
