#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmShallowWater.hpp"
#include "LpmShallowWater_Impl.hpp"
#include "LpmSWEGallery.hpp"
// #include "LpmSWEKernels.hpp"
#include "LpmVorticityGallery.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace Lpm;

struct Input {
  Input(int argc, char* argv[]);

  Real dt;
  Real tfinal;
  std::string case_name;
  Int max_depth;
  Int output_interval;
  Real mesh_radius;

};

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{

  Input input(argc, argv);
  typedef QuadRectSeed seed_type;
//   typedef TriHexSeed seed_type;
//   typedef UnitDiskSeed seed_type;

  const Index tree_depth = input.max_depth;
  Index nmaxverts, nmaxedges, nmaxfaces;
  MeshSeed<seed_type> seed(input.mesh_radius);
  seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);

  auto plane = std::shared_ptr<ShallowWater<seed_type>>(new ShallowWater<seed_type>(
    nmaxverts, nmaxedges, nmaxfaces));
  plane->treeInit(tree_depth,seed);

  typedef Thacker81CurvedOscillation problem_type;
  plane->init_problem<problem_type>();

  const Real dt = input.dt;

  {
   Polymesh2dVtkInterface<seed_type> vtk(plane, plane->surfaceHeightVerts);
   plane->addFieldsToVtk(vtk);

   std::ostringstream ss;
   ss << "tmp/" << input.case_name << seed_type::faceStr() << tree_depth;
   ss << "_dt" << dt << "_0000.vtp";
   vtk.write(ss.str());
  }
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  dt = 0.1;
  tfinal = 1.0;
  case_name = "plane_swe_test";
  max_depth = 3;
  output_interval = 1;
  mesh_radius = 1.0;
  for (Int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-d") {
      max_depth = std::stoi(argv[++i]);
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
    else if (token == "-r") {
      mesh_radius = std::stod(argv[++i]);
    }
  }
}
