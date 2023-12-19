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
#include "lpm_tracer_gallery.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "util/lpm_matlab_io.hpp"

#include <Compadre_Evaluator.hpp>

#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <fstream>
#include <sstream>

using namespace Lpm;

struct Input {
  Int tree_depth;
  Int gmls_order;
  Int gmls_eps;
  Int nunif;
  std::string m_filename;

  Input(const std::map<std::string,std::string>& params);

  std::string info_string() const;
};

struct PlanarGrid {
  Kokkos::View<Real*[2]> pts;
  Kokkos::View<Real*> wts;
  typename Kokkos::View<Real*[2]>::HostMirror h_pts;
  typename Kokkos::View<Real*>::HostMirror h_wts;
  Real xmin;
  Real xmax;
  Real ymin;
  Real ymax;

  PlanarGrid(const int n, const Real xmi, const Real xma) :
    pts("pts", n*n),
    wts("wts", n*n),
    xmin(xmi),
    xmax(xma),
    ymin(xmi),
    ymax(xma) {

    h_pts = Kokkos::create_mirror_view(pts);
    h_wts = Kokkos::create_mirror_view(wts);

    const Real dx = (xmax - xmin)/(n-1);
    const Real dy = (ymax - ymin)/(n-1);
    for (int i=0; i<n*n; ++i) {
      const Int ii = i/n;
      const Int jj = i%n;
      h_pts(i, 0) = xmin + ii*dx;
      h_pts(i, 1) = ymin + jj*dy;
      h_wts(i) = dx*dy;
    }
    Kokkos::deep_copy(pts, h_pts);
    Kokkos::deep_copy(wts, h_wts);
  }
};

TEST_CASE("compadre_unit_tests", "") {
  Comm comm;

  Logger<> logger("compadre_unit_tests", Log::level::debug, comm);

  auto& ts = TestSession::get();
  Input input(ts.params);

  logger.info(input.info_string());

  SECTION("plane geometry") {

    const Real radius = 6;

    PolyMeshParameters<QuadRectSeed> qr_params(input.tree_depth, radius);
    auto quad_plane = PolyMesh2d<QuadRectSeed>(qr_params);

    logger.info(quad_plane.info_string());

    PlanarGaussian gaussian;

    ScalarField<VertexField> gaussian_verts("gaussian",
      quad_plane.n_vertices_host());
    ScalarField<FaceField> gaussian_faces("gaussian",
      quad_plane.n_faces_host());

    auto vcrds = quad_plane.vertices.phys_crds.view;
    auto fcrds = quad_plane.faces.phys_crds.view;
    Kokkos::parallel_for(quad_plane.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        gaussian_verts.view(i) = gaussian(Kokkos::subview(vcrds, i, Kokkos::ALL));
      });
    Kokkos::parallel_for(quad_plane.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        gaussian_faces.view(i) = gaussian(Kokkos::subview(fcrds, i, Kokkos::ALL));
      });

    auto src_crds = quad_plane.get_leaf_face_crds();
    auto h_src_crds = Kokkos::create_mirror_view(src_crds);
    auto src_data = quad_plane.faces.leaf_field_vals(gaussian_faces);
    Kokkos::deep_copy(h_src_crds, src_crds);

    PlanarGrid grid(input.nunif, -radius, radius);
    gmls::Params gmls_params(input.gmls_order, PlaneGeometry::ndim);
    gmls_params.topo_dim = 2;
    gmls_params.ambient_dim = 2;
    gmls::Neighborhoods face_neighbors(h_src_crds, grid.h_pts, gmls_params);
    gmls::Neighborhoods mesh_neighbors(h_src_crds, quad_plane.vertices.phys_crds.get_host_crd_view(), gmls_params);

    logger.info(face_neighbors.info_string());
    logger.info(mesh_neighbors.info_string());

    const std::vector<Compadre::TargetOperation> gmls_ops =
      {Compadre::ScalarPointEvaluation,
       Compadre::LaplacianOfScalarPointEvaluation};

    auto scalar_gmls = gmls::plane_scalar_gmls(src_crds,
      grid.pts, face_neighbors, gmls_params, gmls_ops);

    auto mesh_gmls = gmls::plane_scalar_gmls(src_crds, vcrds, mesh_neighbors, gmls_params, gmls_ops);

    Compadre::Evaluator gmls_eval_scalar(&scalar_gmls);
    Compadre::Evaluator gmls_mesh_eval(&mesh_gmls);

    auto gaussian_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        src_data,
        Compadre::ScalarPointEvaluation,
        Compadre::PointSample);
    auto lap_gaussian_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        src_data,
        Compadre::LaplacianOfScalarPointEvaluation,
        Compadre::PointSample);

    auto gaussian_gmls_verts = gmls_mesh_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(src_data, Compadre::ScalarPointEvaluation, Compadre::PointSample);

    Kokkos::View<Real*> grid_gaussian("gaussian_exact", grid.pts.extent(0));
    Kokkos::View<Real*> grid_lap_gaussian("laplacian_gauss_exact", grid.pts.extent(0));
    Kokkos::parallel_for(grid.pts.extent(0),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(grid.pts, i, Kokkos::ALL);
        grid_gaussian(i) = gaussian(xy);
        grid_lap_gaussian(i) = gaussian.laplacian(xy);
      });

    Kokkos::View<Real*> gauss_err("gaussian_error", grid.pts.extent(0));
    Kokkos::View<Real*> gauss_lap_err("laplacian_gauss_err", grid.pts.extent(0));

    ErrNorms gauss_err_norms(gauss_err, gaussian_gmls, grid_gaussian, grid.wts);
    ErrNorms lap_gauss_err_norms(gauss_lap_err, lap_gaussian_gmls, grid_lap_gaussian, grid.wts);

    logger.info("interpolation error: {}", gauss_err_norms.info_string());
    logger.info("laplacian error: {}", lap_gauss_err_norms.info_string());

#ifdef LPM_USE_VTK
  VtkPolymeshInterface<QuadRectSeed> vtk(quad_plane);
  vtk.add_scalar_point_data(gaussian_verts.view);
  vtk.add_scalar_point_data(gaussian_gmls_verts);
  vtk.add_scalar_cell_data(gaussian_faces.view);
  vtk.write("compadre_plane_test.vtp");
#endif

    auto h_gauss_gmls = Kokkos::create_mirror_view(gaussian_gmls);
    Kokkos::deep_copy(h_gauss_gmls, gaussian_gmls);
    auto h_gauss_lap_gmls = Kokkos::create_mirror_view(lap_gaussian_gmls);
    Kokkos::deep_copy(h_gauss_lap_gmls, lap_gaussian_gmls);

    auto h_grid_gaussian = Kokkos::create_mirror_view(grid_gaussian);
    auto h_grid_lap_gaussian = Kokkos::create_mirror_view(grid_lap_gaussian);
    Kokkos::deep_copy(h_grid_gaussian, grid_gaussian);
    Kokkos::deep_copy(h_grid_lap_gaussian, grid_lap_gaussian);

    std::ofstream mfile(input.m_filename);
    write_array_matlab(mfile, "gridxy", grid.h_pts);
    write_vector_matlab(mfile, "gaussian_exact", h_grid_gaussian);
    write_vector_matlab(mfile, "gaussian_gmls", h_gauss_gmls);
    write_vector_matlab(mfile, "lap_gaussian_exact", h_grid_lap_gaussian);
    mfile.close();
  }

  SECTION("sphere geometry") {
    PolyMeshParameters<IcosTriSphereSeed> ic_params(input.tree_depth);

    auto trisphere = PolyMesh2d<IcosTriSphereSeed>(ic_params);

    logger.info(trisphere.info_string());

    RossbyHaurwitz54 rh54;

    ScalarField<VertexField> rh54_verts("rh54", trisphere.n_vertices_host());
    const auto vcrds = trisphere.vertices.phys_crds.view;
    Kokkos::parallel_for("init rh54 verts", trisphere.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        rh54_verts.view(i) = rh54(vcrds(i,0), vcrds(i,1), vcrds(i,2));
      });
    ScalarField<FaceField> rh54_faces("rh54", trisphere.n_faces_host());
    const auto fcrds = trisphere.faces.phys_crds.view;
    Kokkos::parallel_for("init rh54 faces", trisphere.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        rh54_faces.view(i) = rh54(fcrds(i,0), fcrds(i,1), fcrds(i,2));
      });

    auto src_crds = trisphere.get_leaf_face_crds();
    auto h_src_crds = Kokkos::create_mirror_view(src_crds);
    Kokkos::deep_copy(h_src_crds, src_crds);

    // target points will be a uniform lat-lon grid
    const int nlat = input.nunif/2 + 1;
    const int nlon = input.nunif;
    LatLonPts ll(nlat, nlon);

    Kokkos::View<Real*> ll_rh54("rh54_exact", ll.n());
    Kokkos::View<Real*> ll_lap_rh54("lap_rh54_exact", ll.n());
    Kokkos::parallel_for(ll.n(), KOKKOS_LAMBDA (const Index i) {
      const Real zeta = rh54(ll.pts(i,0), ll.pts(i,1), ll.pts(i,2));
      ll_rh54(i) = zeta;
      ll_lap_rh54(i) = -30*zeta;
    });

#ifdef LPM_USE_VTK
  VtkPolymeshInterface<IcosTriSphereSeed> vtk(trisphere);
  vtk.add_scalar_point_data(rh54_verts.view);
  vtk.add_scalar_cell_data(rh54_faces.view);
  vtk.write("compadre_sphere_test.vtp");
#endif

    gmls::Params gmls_params(input.gmls_order);
    gmls::Neighborhoods face_neighbors(h_src_crds, ll.h_pts, gmls_params);

    logger.info(face_neighbors.info_string());

    const std::vector<Compadre::TargetOperation> gmls_ops =
      {Compadre::ScalarPointEvaluation,
       Compadre::LaplacianOfScalarPointEvaluation};

    auto scalar_gmls = gmls::sphere_scalar_gmls(src_crds,
      ll.pts, face_neighbors, gmls_params, gmls_ops);

    Compadre::Evaluator gmls_eval_scalar(&scalar_gmls);

    auto src_data = trisphere.faces.leaf_field_vals(rh54_faces);
    auto rh54_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        src_data,
        Compadre::ScalarPointEvaluation,
        Compadre::PointSample);
    auto lap_rh54_gmls =
      gmls_eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        src_data,
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
  nunif = 60;
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
    else if (p.first == "n") {
      nunif = std::stoi(p.second);
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
     << tab << "nunif      = " << nunif << "\n"
     << tab << "m_filename = " << m_filename << "\n";
  return ss.str();
}
