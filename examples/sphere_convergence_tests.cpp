#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_lat_lon_pts.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "util/lpm_matlab_io.hpp"
#include "lpm_constants.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_pse.hpp"
#include "lpm_compadre.hpp"
#include "fortran/lpm_ssrfpack_interface.hpp"
#include "fortran/lpm_ssrfpack_interface_impl.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

using namespace Lpm;

struct Input {
  int start_depth;
  int end_depth;
  std::string ofname_base;

  Input(const int argc, char* argv[]);

  std::string info_string() const;
};

template <typename SeedType>
struct SphereConvergenceTest {
  using OtherSeedType = typename std::conditional<
    std::is_same<SeedType, CubedSphereSeed>::value,
    IcosTriSphereSeed, CubedSphereSeed>::type;

  using VelocityType = LauritzenEtAlDeformationalFlow;

  using pse_type = pse::BivariateOrder8<typename SeedType::geo>;

  SphereConvergenceTest(const Input& ipt) : input(ipt),
    start_depth(ipt.start_depth), end_depth(ipt.end_depth) {}

  const Input& input;
  std::string tgt_vtk_fname;

  int start_depth;
  int end_depth;
  std::string vtk_base_fname;
  std::string txt_fname;

  std::vector<Real> dxs;

  std::vector<Real> rh54_interp_l1_pse;
  std::vector<Real> rh54_interp_l2_pse;
  std::vector<Real> rh54_interp_linf_pse;

  std::vector<Real> rh54_lap_l1_pse;
  std::vector<Real> rh54_lap_l2_pse;
  std::vector<Real> rh54_lap_linf_pse;

  std::vector<Real> sxyz_interp_l1_pse;
  std::vector<Real> sxyz_interp_l2_pse;
  std::vector<Real> sxyz_interp_linf_pse;

  std::vector<Real> sxyz_lap_l1_pse;
  std::vector<Real> sxyz_lap_l2_pse;
  std::vector<Real> sxyz_lap_linf_pse;

  std::vector<Real> rh54_interp_l1_gmls;
  std::vector<Real> rh54_interp_l2_gmls;
  std::vector<Real> rh54_interp_linf_gmls;

  std::vector<Real> rh54_lap_l1_gmls;
  std::vector<Real> rh54_lap_l2_gmls;
  std::vector<Real> rh54_lap_linf_gmls;

  std::vector<Real> sxyz_interp_l1_gmls;
  std::vector<Real> sxyz_interp_l2_gmls;
  std::vector<Real> sxyz_interp_linf_gmls;

  std::vector<Real> sxyz_lap_l1_gmls;
  std::vector<Real> sxyz_lap_l2_gmls;
  std::vector<Real> sxyz_lap_linf_gmls;

  std::vector<Real> rh54_interp_l1_herm;
  std::vector<Real> rh54_interp_l2_herm;
  std::vector<Real> rh54_interp_linf_herm;

  std::vector<Real> sxyz_interp_l1_herm;
  std::vector<Real> sxyz_interp_l2_herm;
  std::vector<Real> sxyz_interp_linf_herm;

  std::vector<Real> velocity_l1_pse;
  std::vector<Real> velocity_l2_pse;
  std::vector<Real> velocity_linf_pse;

  std::vector<Real> velocity_l1_gmls;
  std::vector<Real> velocity_l2_gmls;
  std::vector<Real> velocity_linf_gmls;

  std::vector<Real> velocity_l1_herm;
  std::vector<Real> velocity_l2_herm;
  std::vector<Real> velocity_linf_herm;

  void run() {
    Comm comm;

    Logger<> logger("sphere_conv", Log::level::debug, comm);
    logger.debug("test run called.");
    Timer run_timer("SphereConvergenceTest");
    run_timer.start();

    /**
      setup target mesh
    */
    constexpr int output_depth = 5;
    PolyMeshParameters<OtherSeedType> other_params(output_depth);
    auto tgt = TransportMesh2d<OtherSeedType>(other_params);

    constexpr Real rh_u0 = 0;
    constexpr Real rh_amp = 1/30;
    RossbyHaurwitz54 rh54(rh_u0, rh_amp);
    SphereXYZTrigTracer sxyz;

    tgt.template initialize_velocity<VelocityType>();
    tgt.allocate_scalar_tracer("rh54_interp_pse");
    tgt.allocate_scalar_tracer("rh54_lap_pse");
    tgt.allocate_scalar_tracer("sxyz_interp_pse");
    tgt.allocate_scalar_tracer("sxyz_lap_pse");
    tgt.allocate_scalar_tracer("rh54_interp_gmls");
    tgt.allocate_scalar_tracer("rh54_lap_gmls");
    tgt.allocate_scalar_tracer("sxyz_interp_gmls");
    tgt.allocate_scalar_tracer("sxyz_lap_gmls");
    tgt.allocate_scalar_tracer("rh54_interp_herm");
    tgt.allocate_scalar_tracer("sxyz_interp_herm");
    tgt.allocate_scalar_tracer("rh54_interp_pse_error");
    tgt.allocate_scalar_tracer("rh54_lap_pse_error");
    tgt.allocate_scalar_tracer("rh54_interp_gmls_error");
    tgt.allocate_scalar_tracer("rh54_lap_gmls_error");
    tgt.allocate_scalar_tracer("rh54_interp_herm_error");
    tgt.allocate_scalar_tracer("sxyz_interp_pse_error");
    tgt.allocate_scalar_tracer("sxyz_lap_pse_error");
    tgt.allocate_scalar_tracer("sxyz_interp_gmls_error");
    tgt.allocate_scalar_tracer("sxyz_lap_gmls_error");
    tgt.allocate_scalar_tracer("sxyz_interp_herm_error");


    tgt.initialize_tracer(rh54, "rh54_exact");
    tgt.initialize_tracer(sxyz, "sxyz_exact");

    tgt.allocate_scalar_tracer("rh54_lap_exact");
    tgt.allocate_scalar_tracer("sxyz_lap_exact");
    const auto vcrds = tgt.mesh.vertices.phys_crds.view;
    const auto fcrds = tgt.mesh.faces.phys_crds.view;
    auto rh54_lap_exact_verts = tgt.tracer_verts.at("rh54_lap_exact").view;
    auto rh54_lap_exact_faces = tgt.tracer_faces.at("rh54_lap_exact").view;
    auto sxyz_lap_exact_verts = tgt.tracer_verts.at("sxyz_lap_exact").view;
    auto sxyz_lap_exact_faces = tgt.tracer_faces.at("sxyz_lap_exact").view;
    Kokkos::parallel_for("init vert laplacian vals", tgt.mesh.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mxyz = Kokkos::subview(vcrds, i, Kokkos::ALL);
        rh54_lap_exact_verts(i) = rh54.laplacian(mxyz);
        sxyz_lap_exact_verts(i) = sxyz.laplacian(mxyz);
      });
    Kokkos::parallel_for("init face laplacian vals", tgt.mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto mxyz = Kokkos::subview(fcrds, i, Kokkos::ALL);
        rh54_lap_exact_faces(i) = rh54.laplacian(mxyz);
        sxyz_lap_exact_faces(i) = sxyz.laplacian(mxyz);
      });

    VectorField<SphereGeometry, VertexField> vpse_verts("velocity_pse", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vpse_faces("velocity_pse", tgt.mesh.n_faces_host());
    VectorField<SphereGeometry, VertexField> vgmls_verts("velocity_gmls", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vgmls_faces("velocity_gmls", tgt.mesh.n_faces_host());
    VectorField<SphereGeometry, VertexField> vherm_verts("velocity_herm", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vherm_faces("velocity_herm", tgt.mesh.n_faces_host());

    VectorField<SphereGeometry, VertexField> vpse_verts_err("velocity_pse_err", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vpse_faces_err("velocity_pse_err", tgt.mesh.n_faces_host());
    VectorField<SphereGeometry, VertexField> vgmls_verts_err("velocity_gmls_err", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vgmls_faces_err("velocity_gmls_err", tgt.mesh.n_faces_host());
    VectorField<SphereGeometry, VertexField> vherm_verts_err("velocity_herm_err", tgt.mesh.n_vertices_host());
    VectorField<SphereGeometry, FaceField>   vherm_faces_err("velocity_herm_err", tgt.mesh.n_faces_host());

    const auto tgt_vert_xyz = tgt.mesh.vertices.phys_crds.view;
    const auto tgt_face_xyz = tgt.mesh.faces.phys_crds.view;
    const auto tgt_face_mask = tgt.mesh.faces.mask;
    auto verts_pse_interp_rh54 = tgt.tracer_verts.at("rh54_interp_pse").view;
    auto faces_pse_interp_rh54 = tgt.tracer_faces.at("rh54_interp_pse").view;
    auto verts_pse_interp_sxyz = tgt.tracer_verts.at("sxyz_interp_pse").view;
    auto faces_pse_interp_sxyz = tgt.tracer_faces.at("sxyz_interp_pse").view;
    auto verts_pse_lap_rh54 = tgt.tracer_verts.at("rh54_lap_pse").view;
    auto faces_pse_lap_rh54 = tgt.tracer_faces.at("rh54_lap_pse").view;
    auto verts_pse_lap_sxyz = tgt.tracer_verts.at("sxyz_lap_pse").view;
    auto faces_pse_lap_sxyz = tgt.tracer_faces.at("sxyz_lap_pse").view;
    Kokkos::TeamPolicy<> pse_vertex_policy(tgt.mesh.n_vertices_host(), Kokkos::AUTO());
    Kokkos::TeamPolicy<> pse_face_policy(tgt.mesh.n_faces_host(), Kokkos::AUTO());

    logger.debug("tgt mesh initialized.");

    for (int d=start_depth; d<=end_depth; ++d) {
      tgt_vtk_fname = input.ofname_base + "_tgt_" + OtherSeedType::id_string() + std::to_string(output_depth) + "_src_" + SeedType::id_string() +
        std::to_string(d) + vtp_suffix();

      PolyMeshParameters<SeedType> src_params(d);
      auto src = TransportMesh2d<SeedType>(src_params);
      src.template initialize_velocity<VelocityType>();

      dxs.push_back(src.mesh.appx_mesh_size());

      const auto src_vert_xyz = src.mesh.vertices.phys_crds.view;
      const auto src_face_xyz = src.mesh.faces.phys_crds.view;
      const auto src_mask = src.mesh.faces.mask;
      const auto src_area = src.mesh.faces.area;
      const auto src_rh54 = src.tracer_faces.at("rh54_exact").view;

//       Kokkos::parallel_for(pse_vertex_policy,
//         pse::ScalarInterpolation<pse_type>(verts_pse_interp_rh54, tgt_vert_xyz,
//           src_face_xyz, src_rh54, src_area, src_mask,


#ifdef LPM_USE_VTK
      VtkPolymeshInterface<OtherSeedType> tgt_vtk(tgt.mesh);
      tgt_vtk.add_vector_point_data(vpse_verts.view);
      tgt_vtk.add_vector_cell_data(vpse_faces.view);
      tgt_vtk.add_vector_point_data(vgmls_verts.view);
      tgt_vtk.add_vector_cell_data(vgmls_faces.view);
      tgt_vtk.add_vector_point_data(vherm_verts.view);
      tgt_vtk.add_vector_cell_data(vherm_faces.view);
      tgt_vtk.add_vector_point_data(vpse_verts_err.view);
      tgt_vtk.add_vector_cell_data(vpse_faces_err.view);
      tgt_vtk.add_vector_point_data(vgmls_verts_err.view);
      tgt_vtk.add_vector_cell_data(vgmls_faces_err.view);
      tgt_vtk.add_vector_point_data(vherm_verts_err.view);
      tgt_vtk.add_vector_cell_data(vherm_faces_err.view);
      tgt_vtk.write(tgt_vtk_fname);
#endif
  }

  }
};

int main(int argc, char* argv[]) {
/**
    program initialize
  */
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);

  Kokkos::initialize(argc, argv);
  {  // Kokkos scope
  /**
    initialize problem
  */
  Logger<> logger("sphere conv tests", Log::level::debug, comm);
  Input input(argc, argv);
  logger.info(input.info_string());

  SphereConvergenceTest<CubedSphereSeed> conv_test(input);

  conv_test.run();

  } // Kokkos
  /**
    program finalize
  */
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}


Input::Input(const int argc, char* argv[]) {
  start_depth = 2;
  end_depth = 3;
  ofname_base = "sphere_conv";
  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-s") {
      start_depth = std::stoi(argv[++i]);
    }
    else if (token == "-e") {
      end_depth = std::stoi(argv[++i]);
    }
    else if (token == "-o") {
      ofname_base = argv[++i];
    }
  }
  LPM_REQUIRE(start_depth >= 0);
  LPM_REQUIRE(end_depth >= start_depth);
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "start depth = " << start_depth << " end_depth = " << end_depth << " filename_root: " << ofname_base;
  return ss.str();
}
