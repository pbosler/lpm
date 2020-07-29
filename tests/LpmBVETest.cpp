#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBVESphere.hpp"
#include "LpmBVEKernels.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmRK4.hpp"
#include "LpmRK4_Impl.hpp"
#include "LpmMeshSeed.hpp"
#include "Kokkos_Core.hpp"
#include <iomanip>
#include <sstream>

using namespace Lpm;

int main (int argc, char* argv[]) {
ko::initialize(argc, argv);
{

  typedef CubedSphereSeed seed_type;
  //typedef IcosTriSphereSeed seed_type;


  const Index tree_depth = 3;
  Index nmaxverts, nmaxedges, nmaxfaces;
  MeshSeed<seed_type> seed;
  seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);

  BVESphere<seed_type> sphere(nmaxverts, nmaxedges, nmaxfaces, 0);
  sphere.treeInit(tree_depth, seed);

  const auto relvort = std::shared_ptr<VorticityInitialCondition>(new SolidBodyRotation());
  sphere.set_omega(0);
  sphere.init_vorticity(relvort);
  const auto dotprod_ind = sphere.create_tracer("u dot x");

  const auto vvel = sphere.velocityVerts;
  const auto fvel = sphere.velocityFaces;
  const auto vx = sphere.physVerts.crds;
  const auto fx = sphere.physFaces.crds;
  const auto dpv = sphere.tracer_verts[dotprod_ind];
  const auto dpf = sphere.tracer_faces[dotprod_ind];
  ko::parallel_for(sphere.nvertsHost(), KOKKOS_LAMBDA (const Index& i) {
    const auto mx = ko::subview(vx,i,ko::ALL());
    const auto mv = ko::subview(vvel, i, ko::ALL());
    dpv(i) = SphereGeometry::dot(mx,mv);
  });
  ko::parallel_for(sphere.nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
    const auto mx = ko::subview(fx, i, ko::ALL());
    const auto mv = ko::subview(fvel,i,ko::ALL());
    dpf(i) = SphereGeometry::dot(mx,mv);
  });
  sphere.outputVtk("tmp/bve_test0000.vtk");

  const Real tfinal = 0.003;
  const Real dt = 0.001;
  const Int ntimesteps = std::floor(tfinal/dt);
  const Real Omega = 2*PI;
  BVERK4 solver(dt, Omega);
  solver.init(sphere.nvertsHost(), sphere.nfacesHost());

  std::stringstream ss;

  for (Int time_ind = 0; time_ind<ntimesteps; ++time_ind) {
    solver.advance_timestep(sphere.physVerts.crds, sphere.relVortVerts, sphere.velocityVerts,
      sphere.physFaces.crds, sphere.relVortFaces, sphere.velocityFaces, sphere.faces.area, sphere.faces.mask);

    sphere.t = (time_ind+1)*dt;
    ss << "tmp/bve_test"  << std::setfill('0') << std::setw(4) << time_ind+1 <<  ".vtk";
    sphere.outputVtk(ss.str());
    ss.str("");
  }
}
std::cout << "tests pass" << std::endl;
ko::finalize();
return 0;
}
