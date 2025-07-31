#include <sstream>
#include <iomanip>
#include <fstream>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_rbf_knn.hpp"
#include "lpm_rbf.hpp"
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
  Int rbf_order;
  Int rbf_eps;
  Int nunif;
  Int N;
  std::string m_filename;

  Input(const std::map<std::string,std::string>& params);

  std::string info_string() const;
};


TEST_CASE("rbf_unit_tests", "") {
  Comm comm;

  Logger<> logger("compadre_unit_tests", Log::level::debug, comm);

  auto& ts = TestSession::get();
  Input input(ts.params);

  logger.info(input.info_string());

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
  vtk.write("rbf_sphere_test.vtp");
#endif


    logger.info(face_neighbors.info_string());

    // Target operations interpoaltion and laplacian evaluation
    /**insert function call**/
//    std::cout << face_neighbors.neighbor_lists(0,0) << std::endl;
//    std::cout << face_neighbors. << std::endl;
    
    //input variables and neighbor search
    rbf_neighbor_search<Input> rbf_nn(input);
    Kokkos::deep_copy(rbf_nn.source_data_sites,src_crds);

    std::cout << "KNN search start" << std::endl;
    rbf_nn.knn_search();
    std::cout << "KNN search complete" << std::endl;
//    std::cout <<rbf_nn.num_nbrs_list(2) <<std::endl;
  
    //instantiate rbf matrix class and copy neighbor lists
    rbf_team_matrices<Input> rbf_mat(input);
    ko::deep_copy(rbf_mat.dpts,src_crds);
//    ko::deep_copy(rbf_mat.number_neighbors_list,rbf_nn.num_nbrs_list);
//    ko::deep_copy(rbf_mat.cr_neighbor_lists,rbf_nn.nbr_lists);
    rbf_mat.number_neighbors_list = rbf_nn.num_nbrs_list;
    rbf_mat.cr_neighbor_lists = rbf_nn.nbr_lists;
//    
//    //Construct u for f = Au
    sphharm54(src_crds,rbf_mat.u,input.N);

    //generate rbf fd weights for lap in tangent plane (laplace beltrami)    
    auto start = high_resolution_clock::now(); 
    std::cout << "Begin Weight Generation " << std::endl;
    rbf_mat.tpm_lap_wgts();
    std::cout << "Weight Generation Completion" << std::endl;

    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    std::cout << "Time to calculate weights: "
          << duration.count() * 0.000001 << " seconds" << std::endl; 


//    rbf_team_matrices rbf_mat(input);

    //evaluation of weights to target data
    /**insert function call**/

    //Set result vectors
//    auto src_data = trisphere.faces.leaf_field_vals(rh54_faces);
////    auto rh54_rbf = rbf_team_matrices.Iku
////    auto lap_rh54_rbf =
//
//    // Error evaluation
//    Kokkos::View<Real*> rh54_error("rh54_error", ll.n());
//    ErrNorms rh54_err_norms(rh54_error, rh54_rbf, ll_rh54, ll.wts);
//    logger.info("interpolation error: {}", rh54_err_norms.info_string());
//
//    Kokkos::View<Real*> lap_rh54_error("lap_rh54_error", ll.n());
//    ErrNorms lap_rh54_err_norms(lap_rh54_error, lap_rh54_rbf, ll_lap_rh54, ll.wts);
//    logger.info("laplacian error: {}", lap_rh54_err_norms.info_string());
  }
}

Input::Input(const std::map<std::string, std::string>& params) {
  tree_depth = 4;
  rbf_order = 3;
  rbf_eps = 2;
  nunif = 60;
  m_filename = "rbf_tests.m";
  for (const auto& p : params) {
    if (p.first == "tree_depth") {
      tree_depth = std::stoi(p.second);
    }
    else if (p.first == "rbf_order") {
      rbf_order = std::stod(p.second);
    }
    else if (p.first == "rbf_eps") {
      rbf_eps = std::stod(p.second);
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
  ss << "Rbf tests input:\n";
  const auto tab = "\t";
  ss << tab << "tree_depth = " << tree_depth << "\n"
     << tab << "rbf_order = " << rbf_order << "\n"
     << tab << "rbf_eps   = " << rbf_eps << "\n"
     << tab << "nunif      = " << nunif << "\n"
     << tab << "m_filename = " << m_filename << "\n";
  return ss.str();
}
