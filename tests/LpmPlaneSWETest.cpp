#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmShallowWater.hpp"
#include "LpmShallowWater_Impl.hpp"
#include "LpmSWEGallery.hpp"
#include "LpmSWEKernels.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmTimer.hpp"
#include "LpmSWERK4.hpp"
#include "LpmSWERK4_Impl.hpp"
#include "LpmPSE.hpp"
#include "lpm_progress_bar.hpp"

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

  Timer total_timer("total");
  total_timer.start();
  Timer init_timer("initialize");
  init_timer.start();

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

//   typedef Thacker81CurvedOscillation problem_type;
  typedef SimpleGravityWave problem_type;
  plane->init_problem<problem_type>();

  const Real dt = input.dt;
  const Real g = problem_type::g;
  const Real f0 = problem_type::f0;
  const Real beta = problem_type::beta;
  const Real eps = pse_eps(plane->appx_mesh_size());
  SWERK4<seed_type,problem_type> solver(plane, dt, eps);
//   std::cout << solver.infoString();

  Timer single_output_timer("output");
  single_output_timer.start();
  {
   Polymesh2dVtkInterface<seed_type> vtk(plane, plane->surfaceHeightVerts);
   plane->addFieldsToVtk(vtk);

   std::ostringstream ss;
   ss << "tmp/" << input.case_name << seed_type::faceStr() << tree_depth;
   ss << "_dt" << dt << "_0000.vtp";
   vtk.write(ss.str());
  }
  single_output_timer.stop();
  std::cout << "total mass = " << plane->total_mass() << '\n';
  init_timer.stop();
  std::cout << init_timer.infoString();

  std::cout << plane->infoString("plane_mesh_init", 0, (tree_depth < 3));

  Real t;
  const Real tfinal = input.tfinal;
  const Int ntimesteps = std::floor(tfinal/dt);
  ProgressBar progress("SWE_PlaneTest", ntimesteps);
  for (Int time_ind = 0; time_ind < ntimesteps; ++time_ind) {
    solver.advance_timestep();

    progress.update();

//     std::cout << "mass_integral = " << plane->total_mass_integral() << std::endl;

    if ((time_ind+1)%input.output_interval == 0 || time_ind + 1 == ntimesteps) {
       Polymesh2dVtkInterface<seed_type> vtk(plane, plane->surfaceHeightVerts);
       plane->addFieldsToVtk(vtk);

       std::ostringstream ss;
       ss << "tmp/" << input.case_name << seed_type::faceStr() << tree_depth;
       ss << "_dt" << dt << "_" << std::setfill('0') << std::setw(4) << time_ind + 1 << ".vtp";
       vtk.write(ss.str());
    }

  }
  std::cout << plane->infoString("plane_mesh_final", 0, (tree_depth < 3));

  total_timer.stop();
  std::cout << total_timer.infoString();
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  dt = 0.01;
  tfinal = 3*dt;
  case_name = "plane_swe_test";
  max_depth = 3;
  output_interval = 1;
  mesh_radius = 6.0;
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
