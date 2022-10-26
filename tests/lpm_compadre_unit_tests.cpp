#include <sstream>
#include <iomanip>
#include <fstream>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_lat_lon_pts.hpp"
#include "lpm_field.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_functions.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"

#include "Compadre_Evaluator.hpp"

#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include "catch.hpp"
#include <memory>
#include <sstream>

using namespace Lpm;

struct Input {
  Int tree_depth;
  Int gmls_order;
  Int gmls_eps;
  Int nlon;
  std::string m_filename;

  Input(const std::map<std::string,std::string>& params);

  std::string info_string() const;
};

TEST_CASE("compadre_unit_tests", "") {
  Comm comm;

  Logger<> logger("compadre_unit_tests", Log::level::debug, comm);

  auto& ts = TestSession::get();
  Input input(ts.params);

  logger.info(input.info_string());

  SECTION("plane geometry") {
  }

  SECTION("sphere geometry") {
    PolyMeshParameters<IcosTriSphereSeed> ic_params(input.tree_depth);

    auto trisphere = std::shared_ptr<PolyMesh2d<IcosTriSphereSeed>>(new
      PolyMesh2d<IcosTriSphereSeed>(ic_params));

    logger.info(trisphere->info_string());

    RossbyHaurwitz54 rh54;

    ScalarField<VertexField> rh54_verts("rh54", trisphere->n_vertices_host());
    const auto vcrds = trisphere->vertices.phys_crds->crds;
    Kokkos::parallel_for("init rh54 verts", trisphere->n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        rh54_verts.view(i) = rh54(vcrds(i,0), vcrds(i,1), vcrds(i,2));
      });
    ScalarField<FaceField> rh54_faces("rh54", trisphere->n_faces_host());
    const auto fcrds = trisphere->faces.phys_crds->crds;
    Kokkos::parallel_for("init rh54 faces", trisphere->n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        rh54_faces.view(i) = rh54(fcrds(i,0), fcrds(i,1), fcrds(i,2));
      });

    // target points will be a uniform lat-lon grid
    const Real nlat = input.nlon/2 + 1;
    const Real nlon = input.nlon;
    LatLonPts ll(nlat, nlon);

    Kokkos::View<Real*> ll_rh54("rh54_exact", ll.n());
    Kokkos::View<Real*> ll_lap_rh54("lap_rh54_exact", ll.n());
    Kokkos::parallel_for(ll.n(), KOKKOS_LAMBDA (const Index i) {
      const Real zeta = rh54(ll.pts(i,0), ll.pts(i,1), ll.pts(i,2));
      ll_rh54(i) = zeta;
      ll_lap_rh54(i) = -30*zeta;
    });

    gmls::Params gmls_params(input.gmls_order);
    gmls::Neighborhoods face_neighbors(trisphere->faces.phys_crds->get_host_crd_view(),
      ll.h_pts, gmls_params);

    logger.info(face_neighbors.info_string());

    const std::vector<Compadre::TargetOperation> gmls_ops =
      {Compadre::ScalarPointEvaluation,
       Compadre::LaplacianOfScalarPointEvaluation};

    auto scalar_gmls = sphere_scalar_gmls(trisphere->faces.phys_crds->crds,
      ll.pts, face_neighbors, gmls_params, gmls_ops);

    Compadre::Evaluator gmls_eval_scalar(&scalar_gmls);

    auto rh54_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(
        rh54_faces.view,
        Compadre::ScalarPointEvaluation,
        Compadre::PointSample);
    auto lap_rh54_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(
        rh54_faces.view,
        Compadre::LaplacianOfScalarPointEvaluation,
        Compadre::PointSample);

    Kokkos::View<Real*> rh54_error("rh54_error", ll.n());
    ErrNorms rh54_err_norms(rh54_error, rh54_gmls, ll_rh54, ll.wts);
    logger.info("interpolation error: {}", rh54_err_norms.info_string());

    Kokkos::View<Real*> lap_rh54_error("lap_rh54_error", ll.n());
    ErrNorms lap_rh54_err_norms(lap_rh54_error, lap_rh54_gmls, ll_lap_rh54, ll.wts);
    logger.info("laplacian error: {}", lap_rh54_err_norms.info_string());
  }
}

Input::Input(const std::map<std::string, std::string>& params) {
  tree_depth = 4;
  gmls_order = 3;
  gmls_eps = 2;
  nlon = 60;
  m_filename = "compadre_tests.m";
  for (const auto& p : params) {
    if (p.first == "tree_depth") {
      tree_depth = std::stoi(p.second);
    }
    else if (p.first == "gmls_order") {
      gmls_order = std::stod(p.second);
    }
    else if (p.first == "gmls_eps") {
      gmls_eps = std::stod(p.second);
    }
    else if (p.first == "nlon") {
      nlon = std::stoi(p.second);
    }
    else if (p.first == "m_filename") {
      m_filename = p.second;
    }
  }
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Compadre tests input:\n";
  const auto tab = "\t";
  ss << tab << "tree_depth = " << tree_depth << "\n"
     << tab << "gmls_order = " << gmls_order << "\n"
     << tab << "gmls_eps   = " << gmls_eps << "\n"
     << tab << "nlon       = " << nlon << "\n"
     << tab << "m_filename = " << m_filename << "\n";
  return ss.str();
}
