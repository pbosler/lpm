#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBVESphere.hpp"
#include "LpmBVEKernels.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmRK4.hpp"
#include "LpmRK4_Impl.hpp"
#include "LpmSphereTestKernels.hpp"
#include "LpmMeshSeed.hpp"
#include "Kokkos_Core.hpp"
#include "KokkosBlas.hpp"
#include <iomanip>
#include <sstream>

using namespace Lpm;

inline Real courant_number(const Real& dt, const Real& dlam) {return 2*PI * dt / dlam;}

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
  ko::parallel_for(sphere.nvertsHost(), SphereVelocityTangentTestFunctor(sphere.tracer_verts[0],
    sphere.physVerts.crds, sphere.velocityVerts));
  ko::parallel_for(sphere.nfacesHost(), SphereVelocityTangentTestFunctor(sphere.tracer_faces[0],
    sphere.physFaces.crds, sphere.velocityFaces));
  sphere.outputVtk("tmp/bve_test0000.vtk");

  ko::View<Real*[3]> vert_velocity_error("vertex_velocity_error", sphere.nvertsHost());
  ko::View<Real*[3]> face_velocity_error("face_velocity_error", sphere.nfacesHost());
  ko::View<Real*[3]> vert_position_error("vertex_position_error", sphere.nvertsHost());
  ko::View<Real*[3]> face_position_error("face_position_error", sphere.nfacesHost());



  const Real tfinal = 1.0;
  const Real dt = 0.01;
  const Int ntimesteps = std::floor(tfinal/dt);
  const Real Omega = 2*PI;
  BVERK4 solver(dt, Omega);
  solver.init(sphere.nvertsHost(), sphere.nfacesHost());
  auto vertex_policy = ko::TeamPolicy<>(solver.nverts, ko::AUTO());
  auto face_policy = ko::TeamPolicy<>(solver.nfaces, ko::AUTO());
  const Real dlam = sphere.avg_mesh_size_radians();
  std::cout << "Solid body rotation test\n";
  std::cout << "\tavg mesh size = " << RAD2DEG * dlam << " degrees\n";
  std::cout << "\tdt = " << dt << "\n";
  std::cout << "\tcr. = " << courant_number(dt, dlam) << "\n";

  std::stringstream ss;

  for (Int time_ind = 0; time_ind<ntimesteps; ++time_ind) {
    solver.advance_timestep(sphere.physVerts.crds, sphere.relVortVerts, sphere.velocityVerts,
      sphere.physFaces.crds, sphere.relVortFaces, sphere.velocityFaces, sphere.faces.area, sphere.faces.mask);
    ko::parallel_for(vertex_policy, BVEVertexStreamFn(sphere.streamFnVerts, sphere.physVerts.crds,
      sphere.physFaces.crds, sphere.relVortFaces, sphere.faces.area, sphere.faces.mask, sphere.nfacesHost()));
    ko::parallel_for(face_policy, BVEFaceStreamFn(sphere.streamFnFaces, sphere.physFaces.crds,
      sphere.relVortFaces, sphere.faces.area, sphere.faces.mask,sphere.nfacesHost()));

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
