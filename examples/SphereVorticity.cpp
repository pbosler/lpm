#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSpherePoisson.hpp"
#include <iostream>
#include <sstream>

using namespace Lpm;

struct Input {
  int init_depth;
  std::string vtk_froot;
  int nthreads_per_team;

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
    SpherePoisson<seed_type> sphere(nmaxverts, nmaxedges, nmaxfaces);
    sphere.treeInit(input.init_depth, seed);
    sphere.updateDevice();

    /** initialize the Poisson problem */
    sphere.init();
    std::cout << sphere.infoString("sphere_poisson_solver",0,false);
    {
      const auto solve_start_time = tic();
      sphere.solve(input.nthreads_per_team);
      const auto elapsed_time = toc(solve_start_time);
      std::cout << "solve time = " << elapsed_time << " seconds.\n";
    }
    sphere.updateHost();

    /** Output mesh to a vtk file */
    std::ostringstream ss;
    ss << input.vtk_froot << input.init_depth << ".vtk";
    sphere.outputVtk(ss.str());
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  init_depth = 2;
  vtk_froot = "sphere_poisson_";
  nthreads_per_team = 0;
  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-o") {
      vtk_froot = argv[++i];
    }
    else if (token == "-d") {
      init_depth = std::stoi(argv[++i]);
    }
    else if (token == "-n") {
      nthreads_per_team = std::stoi(argv[++i]);
    }
  }
}
