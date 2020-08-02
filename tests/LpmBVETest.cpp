#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBVESphere.hpp"
#include "LpmBVEKernels.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmRK4.hpp"
#include "LpmRK4_Impl.hpp"
#include "LpmSphereTestKernels.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2dVtkInterface.hpp"
#include "LpmPolyMesh2dVtkInterface_Impl.hpp"
#include "LpmErrorNorms.hpp"

#include "Kokkos_Core.hpp"
#include "KokkosBlas.hpp"
#include <iomanip>
#include <sstream>

using namespace Lpm;

inline Real courant_number(const Real& dt, const Real& dlam) {return 2*PI * dt / dlam;}

struct Input {
  Input(int argc, char* argv[]);

  Real dt;
  Real tfinal;
  std::string case_name;
  Int max_depth;
  Int output_interval;

};

int main (int argc, char* argv[]) {
ko::initialize(argc, argv);
{
  Input input(argc, argv);
  typedef CubedSphereSeed seed_type;
  //typedef IcosTriSphereSeed seed_type;


  const Index tree_depth = input.max_depth;
  Index nmaxverts, nmaxedges, nmaxfaces;
  MeshSeed<seed_type> seed;
  seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);

  auto sphere = std::shared_ptr<BVESphere<seed_type>>(new BVESphere<seed_type>(nmaxverts, nmaxedges, nmaxfaces, 0));
  sphere->treeInit(tree_depth, seed);

  const auto relvort = std::shared_ptr<VorticityInitialCondition>(new SolidBodyRotation());
  sphere->set_omega(0);
  sphere->init_vorticity(relvort);
  const auto dotprod_ind = sphere->create_tracer("u dot x");
  ko::parallel_for(sphere->nvertsHost(), SphereVelocityTangentTestFunctor(sphere->tracer_verts[0],
    sphere->physVerts.crds, sphere->velocityVerts));
  ko::parallel_for(sphere->nfacesHost(), SphereVelocityTangentTestFunctor(sphere->tracer_faces[0],
    sphere->physFaces.crds, sphere->velocityFaces));
  const auto vorticity_err_ind = sphere->create_tracer("abs(vorticity_error)");

  ko::View<Real*[3]> vert_velocity_error("vertex_velocity_error", sphere->nvertsHost());
  ko::View<Real*[3]> face_velocity_error("face_velocity_error", sphere->nfacesHost());
  ko::View<Real*[3]> vert_position_error("vertex_position_error", sphere->nvertsHost());
  ko::View<Real*[3]> face_position_error("face_position_error", sphere->nfacesHost());

  const Real tfinal = input.tfinal;
  const Int ntimesteps = std::floor(tfinal/input.dt);
  const Real dt = tfinal/ntimesteps;
  const Real Omega = 2*PI;
  BVERK4 solver(dt, Omega);
  solver.init(sphere->nvertsHost(), sphere->nfacesHost());
  auto vertex_policy = ko::TeamPolicy<>(solver.nverts, ko::AUTO());
  auto face_policy = ko::TeamPolicy<>(solver.nfaces, ko::AUTO());
  const Real dlam = sphere->avg_mesh_size_radians();
  const Real cr = courant_number(dt, dlam);
  std::cout << "Solid body rotation test\n";
  std::cout << sphere->infoString();
  std::cout << "\tavg mesh size = " << RAD2DEG * dlam << " degrees\n";
  std::cout << "\tdt = " << dt << "\n";
  std::cout << "\tcr. = " << cr << "\n";
  if (cr > 0.5) {
    std::ostringstream ss;
    ss << "** warning ** cr = " << cr << " ; usually cr <= 0.5 is required.\n";
    std::cout << ss.str();
  }
  std::cout << "\ttfinal = " << tfinal << "\n";

  {
    std::ostringstream ss;
    Polymesh2dVtkInterface<seed_type> vtk(sphere);
    sphere->addFieldsToVtk(vtk);
    vtk.addVectorPointData(vert_velocity_error);
    vtk.addVectorPointData(vert_position_error);
    vtk.addVectorCellData(face_velocity_error);
    vtk.addVectorCellData(face_position_error);

    ss << "tmp/" << input.case_name << seed_type::faceStr() << input.max_depth << "_dt" << dt << "_" << "0000.vtp";
    vtk.write(ss.str());
  }
  {
    const auto facex = sphere->physFaces.crds;
    ko::View<Real*[3]> fexactvel("exact_velocity", sphere->nfacesHost());
    ko::parallel_for(sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
      const auto myx = ko::subview(facex,i,ko::ALL());
      fexactvel(i,0) = -Omega*myx(1);
      fexactvel(i,1) =  Omega*myx(0);
      fexactvel(i,2) = 0.0;
    });
    ErrNorms<> facevel_err(face_velocity_error, sphere->velocityFaces, fexactvel, sphere->faces.area);
    std::cout << facevel_err.infoString("velocity error at t=0");
  }


  Real t;
  ProgressBar progress("SolidBodyRotation test", ntimesteps);
  for (Int time_ind = 0; time_ind<ntimesteps; ++time_ind) {
    solver.advance_timestep(sphere->physVerts.crds, sphere->relVortVerts, sphere->velocityVerts,
      sphere->physFaces.crds, sphere->relVortFaces, sphere->velocityFaces, sphere->faces.area, sphere->faces.mask);

    sphere->t = (time_ind+1)*dt;
    t = sphere->t;

    ko::parallel_for("BVETest: vertex stream function", vertex_policy,
      BVEVertexStreamFn(sphere->streamFnVerts, sphere->physVerts.crds,
      sphere->physFaces.crds, sphere->relVortFaces, sphere->faces.area, sphere->faces.mask, sphere->nfacesHost()));
    ko::parallel_for("BVETest: face stream function", face_policy, BVEFaceStreamFn(sphere->streamFnFaces, sphere->physFaces.crds,
      sphere->relVortFaces, sphere->faces.area, sphere->faces.mask,sphere->nfacesHost()));
    ko::parallel_for("BVETest: vertex velocity tangent", sphere->nvertsHost(),
      SphereVelocityTangentTestFunctor(sphere->tracer_verts[0], sphere->physVerts.crds, sphere->velocityVerts));
    ko::parallel_for("BVETest: face velocity tangent", sphere->nfacesHost(),
      SphereVelocityTangentTestFunctor(sphere->tracer_faces[0], sphere->physFaces.crds, sphere->velocityFaces));

    progress.update();

    {
      const auto relvort = sphere->relVortVerts;
      const auto absvort = sphere->absVortVerts;
      auto vorterr = sphere->tracer_verts[vorticity_err_ind];
      const auto appxvel = sphere->velocityVerts;
      const auto appxpos = sphere->physVerts.crds;
      const auto lagpos = sphere->lagVerts.crds;
      ko::parallel_for("vertex vorticity error", sphere->nvertsHost(), KOKKOS_LAMBDA (const Index& i) {
        vorterr(i) = relvort(i) - absvort(i);
        const auto myx = ko::subview(appxpos, i, ko::ALL());
        vert_velocity_error(i,0) = appxvel(i,0) - (- Omega * myx(1));
        vert_velocity_error(i,1) = appxvel(i,1) - (  Omega * myx(0));
        vert_velocity_error(i,2) = appxvel(i,2);
        const auto mya = ko::subview(lagpos, i, ko::ALL());
        const Real cosomgt = std::cos(Omega*t);
        const Real sinomgt = std::sin(Omega*t);
        Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
        for (Int j=0; j<3; ++j) {
          vert_position_error(i,j) = myx(j) - exactpos[j];
        }
      });
    }
    {
      const auto relvort = sphere->relVortFaces;
      const auto absvort = sphere->absVortFaces;
      auto vorterr = sphere->tracer_faces[vorticity_err_ind];
      const auto appxvel = sphere->velocityFaces;
      const auto appxpos = sphere->physFaces.crds;
      const auto lagpos = sphere->lagFaces.crds;
      ko::parallel_for("face vorticity error", sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
        vorterr(i) = relvort(i) - absvort(i);
        const auto myx = ko::subview(appxpos, i, ko::ALL());
        face_velocity_error(i,0) = appxvel(i,0) - (- Omega * myx(1));
        face_velocity_error(i,1) = appxvel(i,1) - (  Omega * myx(0));
        face_velocity_error(i,2) = appxvel(i,2);
        const auto mya = ko::subview(lagpos, i, ko::ALL());
        const Real cosomgt = std::cos(Omega*t);
        const Real sinomgt = std::sin(Omega*t);
        Real exactpos[3] = {mya(0)*cosomgt - mya(1)*sinomgt, mya(1)*cosomgt + mya(0)*sinomgt, mya(2)};
        for (Int j=0; j<3; ++j) {
          face_position_error(i,j) = myx(j) - exactpos[j];
        }
      });
    }
    if ( (time_ind+1)%input.output_interval == 0 || time_ind+1 == ntimesteps) {

      Polymesh2dVtkInterface<seed_type> vtk(sphere);
      sphere->addFieldsToVtk(vtk);
      vtk.addVectorPointData(vert_velocity_error);
      vtk.addVectorPointData(vert_position_error);
      vtk.addVectorCellData(face_velocity_error);
      vtk.addVectorCellData(face_position_error);

      std::ostringstream ss;
      ss << "tmp/" << input.case_name << seed_type::faceStr() << input.max_depth << "_dt" << dt << "_";
      ss << std::setfill('0') << std::setw(4) << time_ind+1 <<  ".vtp";
      vtk.write(ss.str());
    }
  }
  {
    const auto facex = sphere->physFaces.crds;
    ko::View<Real*[3]> fexactvel("exact_velocity", sphere->nfacesHost());
    ko::parallel_for(sphere->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
      const auto myx = ko::subview(facex,i,ko::ALL());
      fexactvel(i,0) = -Omega*myx(1);
      fexactvel(i,1) =  Omega*myx(0);
      fexactvel(i,2) = 0.0;
    });
    ErrNorms<> facevort_err(sphere->tracer_faces[vorticity_err_ind], sphere->absVortFaces, sphere->faces.area);
    ErrNorms<> facevel_err(face_velocity_error, sphere->velocityFaces, fexactvel, sphere->faces.area);
    ErrNorms<> facepos_err(face_position_error, sphere->physFaces.crds, sphere->lagFaces.crds, sphere->faces.area);
    std::cout << facevort_err.infoString("vorticity error at t=tfinal");
    std::cout << facevel_err.infoString("velocity error at t=tfinal");
    std::cout << facepos_err.infoString("position error at t=tfinal");
  }
}
std::cout << "tests pass" << std::endl;
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  dt = 0.01;
  tfinal = 1.0;
  case_name = "bve_test";
  max_depth = 3;
  output_interval = 1;
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
  }
}
