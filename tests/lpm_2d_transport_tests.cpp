#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "lpm_2d_transport_mesh.hpp"
#include "lpm_2d_transport_mesh_impl.hpp"
#include "lpm_2d_transport_rk4.hpp"
#include "lpm_2d_transport_rk4_impl.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <map>
#include <memory>
#include <sstream>
#include <iomanip>

using namespace Lpm;

template <typename VelocityType, typename SeedType> struct TimeConvergenceTest {
  int tree_lev;
  int amr_limit;
  Real radius;
  Real tfinal;
  std::vector<Int> nsteps;
  std::vector<Real> dts;
  std::vector<Real> l1;
  std::vector<Real> l2;
  std::vector<Real> linf;
  std::vector<Real> l1rate;
  std::vector<Real> l2rate;
  std::vector<Real> linfrate;
  std::unique_ptr<TransportMesh2d<SeedType>> tm;

  typedef typename std::conditional<std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    PlanarHump, SphericalGaussianHills>::type TracerType1;
  typedef typename std::conditional<std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    PlanarCone, SphericalCosineBells>::type TracerType2;
  typedef typename std::conditional<std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    PlanarSlottedDisk, SphericalCosineBells>::type TracerType3;

  TimeConvergenceTest(const int tl, const int amr, const Real r, const Real tf,
    const std::vector<Int>& ns) :
    tree_lev(tl),
    amr_limit(amr),
    radius(r),
    tfinal(tf),
    nsteps(ns) {
      for (const auto& n : nsteps) {
        dts.push_back(tfinal/n);
      }
    }

    void run() {
      std::string test_name = "transport2d_dtconv_" + SeedType::id_string();

      Comm comm;
      Logger<> logger(test_name, Log::level::debug, comm);

      TracerType1 t1;
      TracerType2 t2;
      TracerType3 t3;

      for (int n_idx=0; n_idx<nsteps.size(); ++n_idx) {

        const auto ns = nsteps[n_idx];
        const auto dt = dts[n_idx];

        PolyMeshParameters<SeedType> params(tree_lev, radius, amr_limit);
        tm = std::make_unique<TransportMesh2d<SeedType>>(params);

        tm->initialize_tracer(t1);
        tm->initialize_tracer(t2);
        tm->initialize_tracer(t3);

        tm->template initialize_velocity<VelocityType>();

        logger.info(tm->info_string());

        #ifdef LPM_USE_VTK
        int frame_counter = 0;
        std::stringstream ss;
        ss << "transport_2d_" << SeedType::id_string() << tree_lev << "_dt_conv_" << float_str(dt);
        const auto base_filename = ss.str();
        ss.str("");
        {
          VtkPolymeshInterface<SeedType> vtk = vtk_interface(*tm);
          vtk.write(base_filename + zero_fill_str(frame_counter++) + vtp_suffix());
          ss.str("");
        }
        #endif

        Transport2dRK4<SeedType> solver(dt, *tm);
        for (int time_idx = 0; time_idx<ns; ++time_idx) {
          solver.template advance_timestep<VelocityType>();
          #ifdef LPM_USE_VTK
            if (time_idx == ns/2) {
              VtkPolymeshInterface<SeedType> vtk = vtk_interface(*tm);
              vtk.write(base_filename + zero_fill_str(frame_counter++) + vtp_suffix());
              ss.str("");
            }
          #endif
        }

        #ifdef LPM_USE_VTK
        {
          VtkPolymeshInterface<SeedType> vtk = vtk_interface(*tm);
          vtk.write(base_filename + zero_fill_str(frame_counter++) + vtp_suffix());
          ss.str("");
        }
        #endif

        Kokkos::View<Real**> face_position_error("face_position_error",
          tm->mesh.n_faces_host(), SeedType::geo::ndim);
        auto final_pos = tm->mesh.faces.phys_crds.view;
        auto init_pos = tm->mesh.faces.lag_crds.view;

        auto face_err_norms = ErrNorms(face_position_error, final_pos, init_pos, tm->mesh.faces.area);
        logger.info("ns = {}, l1 = {}, l2 = {}, linf = {}", ns, face_err_norms.l1, face_err_norms.l2, face_err_norms.linf);

        l1.push_back(face_err_norms.l1);
        l2.push_back(face_err_norms.l2);
        linf.push_back(face_err_norms.linf);
      }
     const auto l1rate = convergence_rates(dts, l1);
     const auto l2rate = convergence_rates(dts, l2);
     const auto linfrate = convergence_rates(dts, linf);

     logger.info(convergence_table("dt", dts, "l1", l1, l1rate));
     logger.info(convergence_table("dt", dts, "l2", l2, l2rate));
     logger.info(convergence_table("dt", dts, "linf", linf, linfrate));

     REQUIRE(FloatingPoint<Real>::equiv(l1rate.back(), 4, 0.05));
     REQUIRE(FloatingPoint<Real>::equiv(l2rate.back(), 4, 0.05));
     REQUIRE(FloatingPoint<Real>::equiv(linfrate.back(), 4, 0.05));
    }
};


TEST_CASE("planar meshes", "") {

  const int tree_lev = 5;
  const int amr_limit = 0;
  const Real radius = 4;
  const Real tfinal = 5;
  const std::vector<Int> nsteps = {10, 20, 40, 80, 100};

  typedef PlanarRigidRotation velocity_field;
//   typedef PlanarDeformationalFlow velocity_field;

  SECTION("triangular panels") {
    typedef TriHexSeed seed_type;

    TimeConvergenceTest<velocity_field, seed_type>
      test_case(tree_lev, amr_limit, radius, tfinal, nsteps);
    test_case.run();
  }

  SECTION("quadrilateral panels") {
    typedef QuadRectSeed seed_type;

    TimeConvergenceTest<velocity_field, seed_type>
      test_case(tree_lev, amr_limit, radius, tfinal, nsteps);
    test_case.run();
  }
}

TEST_CASE("spherical meshes", "") {

  const int tree_lev = 3;
  const int amr_limit = 0;
  const Real radius = 1;
  const Real tfinal = 5;
  const std::vector<Int> nsteps = {10, 20, 40, 80, 100};

  typedef LauritzenEtAlDeformationalFlow velocity_field;

  SECTION("triangular panels") {
    typedef IcosTriSphereSeed seed_type;

    TimeConvergenceTest<velocity_field, seed_type>
      test_case(tree_lev, amr_limit, radius, tfinal, nsteps);
    test_case.run();
  }

  SECTION("quadrilateral panels") {
    typedef CubedSphereSeed seed_type;

    TimeConvergenceTest<velocity_field, seed_type>
      test_case(tree_lev, amr_limit, radius, tfinal, nsteps);
    test_case.run();


  }
}
