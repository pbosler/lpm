#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_pse.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "util/lpm_matlab_io.hpp"
#include "lpm_constants.hpp"
#include "lpm_velocity_gallery.hpp"
#include "fortran/lpm_f_interp.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace Lpm;

struct PlanarGrid {
  Kokkos::View<Real*[2]> pts;
  Kokkos::View<Real*> wts;
  typename Kokkos::View<Real*[2]>::HostMirror h_pts;
  typename Kokkos::View<Real*>::HostMirror h_wts;
  Real xmin;
  Real xmax;
  Real ymin;
  Real ymax;

  PlanarGrid(const int n, const Real maxr) :
    pts("pts", n*n),
    wts("wts", n*n),
    xmin(-maxr),
    xmax(maxr),
    ymin(-maxr),
    ymax(maxr) {

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

  inline int size() const {return pts.extent(0);}

  inline int n() const {return int(sqrt(pts.extent(0)));}

  inline int nx() const {return n();}

  inline int ny() const {return n();}
};


template <typename VelocityType, typename SeedType>
struct BivarConvergenceTest {
  int start_depth;
  int end_depth;
  Real radius;
  std::vector<Real> dxs;
  std::vector<Real> grid_interp_l1;
  std::vector<Real> grid_interp_l2;
  std::vector<Real> grid_interp_linf;
  std::vector<Real> face_interp_l1;
  std::vector<Real> face_interp_l2;
  std::vector<Real> face_interp_linf;

  BivarConvergenceTest(const int sd, const int ed, const Real r) :
    start_depth(sd),
    end_depth(ed),
    radius(r) {}

  void run() {
    Comm comm;

    PlanarGaussian tracer;

    Logger<> logger("planar_bivar_conv", Log::level::debug, comm);
    logger.debug("test run called.");

    const int n_unif = 60;
    const Real radius = 6;
    PlanarGrid grid(n_unif, radius);
    scalar_view_type grid_tracer("tracer", grid.size());
    scalar_view_type grid_tracer_interp("tracer_interp", grid.size());
    scalar_view_type grid_error("error", grid.size());

    Kokkos::parallel_for(grid.size(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(grid.pts, i, Kokkos::ALL);
        grid_tracer(i) = tracer(xy);
      });

    for (int i=0; i<(end_depth - start_depth) + 1; ++i) {
      const int depth = start_depth + i;
      std::ostringstream ss;
      ss << "bivar_planar_conv_" << SeedType::id_string() << depth;
      const auto test_name = ss.str();
      ss.str("");

      Timer timer(test_name);
      timer.start();

      logger.info("starting test: {} {}", test_name, depth);

      PolyMeshParameters<SeedType> params(depth, radius);
      const auto pm = std::shared_ptr<TransportMesh2d<SeedType>>(new
         TransportMesh2d<SeedType>(params));
      pm->template initialize_velocity<VelocityType>();
      pm->initialize_tracer(tracer);
      pm->initialize_scalar_tracer("tracer_interp");
      pm->initialize_scalar_tracer("tracer_error");
      dxs.push_back(pm->appx_mesh_size());

      GatherSourceData<SeedType> gathered_data(pm);
      gathered_data.init_scalar_fields(pm->tracer_verts, pm->tracer_faces);
      gathered_data.gather_scalar_fields(pm->tracer_verts, pm->tracer_faces);

      logger.debug("gathered bivar input data.");

      auto xo = Kokkos::subview(grid.pts, Kokkos::ALL, 0);
      auto yo = Kokkos::subview(grid.pts, Kokkos::ALL, 1);
      auto h_gt = Kokkos::create_mirror_view(grid_tracer_interp);
      BivarInterface bivar(grid.size(), gathered_data);
      Kokkos::deep_copy(bivar.x_out, xo);
      Kokkos::deep_copy(bivar.y_out, yo);
      bivar.z_out = h_gt;
      bivar.interpolate_scalar();
      Kokkos::deep_copy(grid_tracer_interp, h_gt);

      ErrNorms grid_interp_err(grid_error, grid_tracer_interp, grid_tracer,
        grid.wts);
      logger.info("grid interp error : {}", grid_interp_err.info_string());
      grid_interp_l1.push_back(grid_interp_err.l1);
      grid_interp_l2.push_back(grid_interp_err.l2);
      grid_interp_linf.push_back(grid_interp_err.linf);

      scalar_view_type mesh_interp("tracer_interp", gathered_data.n());
      xo = Kokkos::subview(gathered_data.src_crds, Kokkos::ALL, 0);
      yo = Kokkos::subview(gathered_data.src_crds, Kokkos::ALL, 1);
      auto h_zo = Kokkos::create_mirror_view(mesh_interp);
      auto h_xo = Kokkos::create_mirror_view(xo);
      auto h_yo = Kokkos::create_mirror_view(yo);
      bivar.reset_output_pts(h_xo, h_yo, h_zo);
      bivar.interpolate_scalar();
      Kokkos::deep_copy(mesh_interp, h_zo);

      std::shared_ptr<PolyMesh2d<SeedType>> base_ptr(pm);
      ScatterViewToMesh scatter_data(gathered_data.scalar_fields.at("tracer_interp"),
        pm->tracer_verts.at("tracer_interp").view,
        pm->tracer_faces.at("tracer_interp").view,
        base_ptr);
      scatter_data.scatter();

      compute_error(pm->tracer_verts.at("tracer_error").view,
        pm->tracer_verts.at("tracer_interp").view,
        pm->tracer_verts.at("tracer").view);

      ErrNorms face_interp_err(pm->tracer_faces.at("tracer_error").view,
        pm->tracer_faces.at("tracer_interp").view,
        pm->tracer_faces.at("tracer").view,
        pm->faces.area);

      logger.info("face interp error : {}", face_interp_err.info_string());
      face_interp_l1.push_back(face_interp_err.l1);
      face_interp_l2.push_back(face_interp_err.l2);
      face_interp_linf.push_back(face_interp_err.linf);


#ifdef LPM_USE_VTK
      VtkPolymeshInterface<SeedType> vtk(pm);
      vtk.write(test_name + vtp_suffix());
#endif

      auto h_grid_tracer = Kokkos::create_mirror_view(grid_tracer);
      auto h_grid_tracer_interp = Kokkos::create_mirror_view(grid_tracer_interp);
      auto h_grid_error = Kokkos::create_mirror_view(grid_error);
      Kokkos::deep_copy(h_grid_tracer, grid_tracer);
      Kokkos::deep_copy(h_grid_tracer_interp, grid_tracer_interp);
      Kokkos::deep_copy(h_grid_error, grid_error);
      std::ofstream mfile(test_name + ".m");
      write_array_matlab(mfile, "gridxy", grid.h_pts);
      write_vector_matlab(mfile, "gridwts", grid.h_wts);
      write_vector_matlab(mfile, "tracer", h_grid_tracer);
      write_vector_matlab(mfile, "tracer_interp", h_grid_tracer_interp);
      write_vector_matlab(mfile, "error", h_grid_error);
      mfile.close();


    }
    const auto grid_interp_l1_rate = convergence_rates(dxs, grid_interp_l1);
    const auto grid_interp_l2_rate = convergence_rates(dxs, grid_interp_l2);
    const auto grid_interp_linf_rate = convergence_rates(dxs, grid_interp_linf);
    logger.info(convergence_table("dx", dxs, "grid_interp_l1", grid_interp_l1, grid_interp_l1_rate));
    logger.info(convergence_table("dx", dxs, "grid_interp_l2", grid_interp_l2, grid_interp_l2_rate));
    logger.info(convergence_table("dx", dxs, "grid_interp_linf", grid_interp_linf, grid_interp_linf_rate));
    const auto face_interp_l1_rate = convergence_rates(dxs, face_interp_l1);
    const auto face_interp_l2_rate = convergence_rates(dxs, face_interp_l2);
    const auto face_interp_linf_rate = convergence_rates(dxs, face_interp_linf);
    logger.info(convergence_table("dx", dxs, "face_interp_l1", face_interp_l1, face_interp_l1_rate));
    logger.info(convergence_table("dx", dxs, "face_interp_l2", face_interp_l2, face_interp_l2_rate));
    logger.info(convergence_table("dx", dxs, "face_interp_linf", face_interp_linf, face_interp_linf_rate));
  }
};

TEST_CASE("planar_bivar", "") {
  const int start_depth = 2;
  int end_depth = 3;
  const Real radius = 6;

  SECTION("tri_hex_seed") {
    typedef TriHexSeed seed_type;
    typedef PlanarConstantEastward velocity_type;
    BivarConvergenceTest<velocity_type,seed_type>
      bivar_test(start_depth, end_depth, radius);
    bivar_test.run();

  }
  SECTION("quad_rect_seed") {
    typedef QuadRectSeed seed_type;
    typedef PlanarConstantEastward velocity_type;
    BivarConvergenceTest<velocity_type,seed_type>
      bivar_test(start_depth, end_depth, radius);
    bivar_test.run();
  }
}
