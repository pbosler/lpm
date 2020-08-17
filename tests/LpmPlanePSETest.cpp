#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmErrorNorms.hpp"
#include "LpmPSE.hpp"
#include "LpmPolyMesh2dVtkInterface.hpp"
#include "LpmPolyMesh2dVtkInterface_Impl.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace Lpm;

template <typename CV> KOKKOS_INLINE_FUNCTION
Real sfn(const CV& xy) {
  const Real r = PlaneGeometry::mag(xy);
  return r * std::exp(-r)*std::sin(r);
}

template <typename CV> KOKKOS_INLINE_FUNCTION
Real lap_sfn(const CV& xy) {
  const Real r = PlaneGeometry::mag(xy);
  const Real rsq = square(r);
  return safe_divide(r)*(std::exp(-r)*((-2*rsq + 3*r)*std::cos(r) + (1-3*r)*std::sin(r)));
}

struct Input {
  Input(int argc, char* argv[]);

  Int max_depth;
  std::string case_name;
};

struct Output {
  std::vector<Index> nsrc;
  std::vector<Real> dx;
  std::vector<Real> eps;
  std::vector<Real> lap_linf_verts;
  std::vector<Real> lap_l2_faces;
  std::vector<Real> lap_linf_faces;
  std::vector<Real> lap_linf_rate_verts;
  std::vector<Real> lap_linf_rate_faces;
  std::vector<Real> lap_l2_rate_faces;;

  Output(const int ntrials) : nsrc(ntrials), dx(ntrials), eps(ntrials),
    lap_linf_verts(ntrials), lap_l2_faces(ntrials), lap_linf_faces(ntrials),
    lap_linf_rate_verts(ntrials), lap_l2_rate_faces(ntrials), lap_linf_rate_faces(ntrials) {}

  std::string infoString() const;

  void compute_rates();
};

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
  const Real mesh_radius = 8;
  Input input(argc, argv);
  const Int max_tree_depth = input.max_depth;

//   typedef TriHexSeed seed_type;
  typedef QuadRectSeed seed_type;
  MeshSeed<seed_type> seed(mesh_radius);
  Index nv, ne, nf;

  const Short ntrials = max_tree_depth - 2 +1;
  Output output(ntrials);

  for (Short trial_ind = 0; trial_ind < ntrials; ++trial_ind) {

    const Short tree_depth = trial_ind + 2;
    seed.setMaxAllocations(nv, ne, nf, tree_depth);

    auto plane = std::shared_ptr<PolyMesh2d<seed_type>>(new PolyMesh2d<seed_type>(nv, ne, nf));
    plane->treeInit(tree_depth, seed);
    std::cout << plane->infoString("plane pse");

    const Real dx =  plane->appx_mesh_size();
    const Real eps = pse_eps(dx);
    output.nsrc[trial_ind] = plane->faces.nLeavesHost();
    output.dx[trial_ind] = dx;
    output.eps[trial_ind] = eps;


    std::cout << '\t' << "appx. dx = " << dx << '\n';
    std::cout << '\t' << "pse_eps = " << eps << '\n';

    scalar_view_type vert_data("vert_data", plane->nvertsHost());
    scalar_view_type vert_lap_exact("vert_lap_exact", plane->nvertsHost());
    scalar_view_type vert_lap_pse("vert_lap_pse", plane->nvertsHost());
    auto vx = plane->physVerts.crds;
    ko::parallel_for(plane->nvertsHost(), KOKKOS_LAMBDA (const Index& i) {
      const auto mxy = ko::subview(vx, i, ko::ALL);
      vert_data(i) = sfn(mxy);
      vert_lap_exact(i) = lap_sfn(mxy);
    });

    scalar_view_type face_data("face_data", plane->nfacesHost());
    scalar_view_type face_lap_exact("face_lap_exact", plane->nfacesHost());
    scalar_view_type face_lap_pse("face_lap_pse", plane->nfacesHost());
    auto fx = plane->physFaces.crds;
    ko::parallel_for(plane->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
      const auto mxy = ko::subview(fx, i, ko::ALL);
      face_data(i) = sfn(mxy);
      face_lap_exact(i) = lap_sfn(mxy);
    });

    ko::TeamPolicy<> vertex_policy(plane->nvertsHost(), ko::AUTO());
    ko::parallel_for(vertex_policy, PlanePSELaplacian(vert_lap_pse, vx, vert_data,
      fx, face_data, plane->faces.area, eps, plane->nfacesHost()));

    ko::TeamPolicy<> face_policy(plane->nfacesHost(), ko::AUTO());
    ko::parallel_for(face_policy, PlanePSELaplacian(face_lap_pse, fx, face_data,
      fx, face_data, plane->faces.area, eps, plane->nfacesHost()));

    scalar_view_type vert_err("vert_err", plane->nvertsHost());
    ko::parallel_for(plane->nvertsHost(), KOKKOS_LAMBDA (const Index& i) {
      vert_err(i) = vert_lap_pse(i) - vert_lap_exact(i);
    });
    scalar_view_type face_err("face_err", plane->nfacesHost());
    const auto fm = plane->faces.mask;
    ko::parallel_for(plane->nfacesHost(), KOKKOS_LAMBDA (const Index& i) {
      if (fm(i)) {
        face_err(i) = 0;
      }
      else {
        face_err(i) = face_lap_pse(i) - face_lap_exact(i);
      }
    });
    ErrNorms<> enorm_faces(face_err, face_lap_exact, plane->faces.area);
    Real li=0;
    ko::parallel_reduce(plane->nvertsHost(), KOKKOS_LAMBDA (const Index& i, Real& me) {
      if (vert_err(i) > me) me = vert_err(i);
    }, ko::Max<Real>(li));

    output.lap_linf_verts[trial_ind] = li/4;
    output.lap_linf_faces[trial_ind] = enorm_faces.linf;
    output.lap_l2_faces[trial_ind] = enorm_faces.l2;

    std::ostringstream ss;
    Polymesh2dVtkInterface<seed_type> vtk(plane);
    vtk.addScalarPointData(vert_data);
    vtk.addScalarPointData(vert_lap_pse);
    vtk.addScalarPointData(vert_lap_exact);
    vtk.addScalarPointData(vert_err);
    vtk.addScalarCellData(face_data);
    vtk.addScalarCellData(face_lap_pse);
    vtk.addScalarCellData(face_lap_exact);
    vtk.addScalarCellData(face_err);
    ss << "tmp/" << input.case_name << "_" << seed_type::faceStr() << tree_depth << ".vtp";
    vtk.write(ss.str());
  }
  output.compute_rates();
  std::cout << output.infoString();
}
ko::finalize();
return 0;
}

void Output::compute_rates() {
  for (int i=1; i<nsrc.size(); ++i) {
    const Real run = std::log(dx[i]) - std::log(dx[i-1]);
    lap_linf_rate_verts[i] = (std::log(lap_linf_verts[i]) - std::log(lap_linf_verts[i-1]))/run;
    lap_linf_rate_faces[i] = (std::log(lap_linf_faces[i]) - std::log(lap_linf_faces[i-1]))/run;
    lap_l2_rate_faces[i] = (std::log(lap_l2_faces[i]) - std::log(lap_l2_faces[i-1]))/run;
  }
}

std::string Output::infoString() const {
  const Int fw = 12;
  std::ostringstream ss;
  ss << "PSE Planar Laplacian error data:\n";
  ss << std::setw(fw) << "nsrc" << std::setw(fw) << "dx" << std::setw(fw) << "eps";
  ss << std::setw(fw) << "linf(verts)" << std::setw(fw) << "rate" << std::setw(fw)
     << "l2(faces)" << std::setw(fw) << "rate" << std::setw(fw) << "linf(faces)"
     << std::setw(fw) << "rate\n";
  for (int i=0; i<nsrc.size(); ++i) {
    ss << std::setw(fw) << nsrc[i] << std::setw(fw) << dx[i] << std::setw(fw) << eps[i];
    ss << std::setw(fw) << lap_linf_verts[i] << std::setw(fw) << lap_linf_rate_verts[i];
    ss << std::setw(fw) << lap_l2_faces[i] << std::setw(fw) << lap_l2_rate_faces[i];
    ss << std::setw(fw) << lap_linf_faces[i] << std::setw(fw) << lap_linf_rate_faces[i];
    ss << '\n';
  }

  return ss.str();
}

Input::Input(int argc, char* argv[]) {
  max_depth = 3;
  case_name = "pse_test";
  for (Int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-d") {
      max_depth = std::stoi(argv[++i]);
    }
    else if (token == "-o") {
      case_name = argv[++i];
    }
  }
}
