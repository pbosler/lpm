#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
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
  std::shared_ptr<TransportMesh2d<SeedType>> tm;

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
      PolyMeshParameters<SeedType> params(tree_lev, radius, amr_limit);
      tm = std::shared_ptr<TransportMesh2d<SeedType>>(new TransportMesh2d<SeedType>(params));
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

        tm->initialize_tracer(t1);
        tm->initialize_tracer(t2);
        tm->initialize_tracer(t3);

        tm->template initialize_velocity<VelocityType>();

        logger.info(tm->info_string());

        #ifdef LPM_USE_VTK
        {
            VtkPolymeshInterface<SeedType> vtk = vtk_interface(tm);
            vtk.write(test_name + "0.vtp");
        }
        #endif

        Transport2dRK4<SeedType> solver(dt, tm);
        for (int time_idx = 0; time_idx<ns; ++time_idx) {
          solver.template advance_timestep<VelocityType>();
          #ifdef LPM_USE_VTK
            if (time_idx == ns/2) {
              VtkPolymeshInterface<SeedType> vtk = vtk_interface(tm);
              vtk.write(test_name + "1.vtp");
            }
          #endif
        }

        #ifdef LPM_USE_VTK
        {
            VtkPolymeshInterface<SeedType> vtk = vtk_interface(tm);
            vtk.write(test_name + "2.vtp");
        }
        #endif

        Kokkos::View<Real**> face_position_error("face_position_error",
          tm->n_faces_host(), SeedType::geo::ndim);
        auto final_pos = tm->faces.phys_crds->crds;
        auto init_pos = tm->faces.lag_crds->crds;

        auto face_err_norms = ErrNorms<>(face_position_error, final_pos, init_pos, tm->faces.area);
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


  Comm comm;
  Logger<> logger("lpm_2d_transport::spherical_mesh_test", Log::level::debug, comm);

  const int tree_lev = 3;
  const int amr_limit = 0;
  const Real radius = 1;
  const Real tfinal = 5;
  const std::vector<Int> nsteps = {10, 20, 40, 80, 100};
  std::vector<Real> dts;
  for (int i=0; i<nsteps.size(); ++i) {
    dts.push_back(tfinal/nsteps[i]);
  }
//   typedef SphericalRigidRotation velocity_field;
  typedef LauritzenEtAlDeformationalFlow velocity_field;

  SECTION("triangular panels") {
    typedef IcosTriSphereSeed seed_type;

    std::vector<Real> l1;
    std::vector<Real> l2;
    std::vector<Real> linf;
    std::vector<Real> l1conv;
    std::vector<Real> l2conv;
    std::vector<Real> linfconv;

    for (const auto& ns : nsteps) {
      const Real dt = tfinal/ns;

      PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
      auto tm = std::shared_ptr<TransportMesh2d<seed_type>>(new TransportMesh2d<seed_type>(params));

      SphericalSlottedCylinders sc;
      SphericalCosineBells cb;
      SphericalGaussianHills gh;

      tm->initialize_tracer(gh);
      tm->initialize_tracer(cb);
      tm->initialize_tracer(sc);

      Kokkos::parallel_for(tm->vertices.nh(),
        VelocityKernel<velocity_field>(tm->velocity_verts.view, tm->vertices.phys_crds->crds, 0));
      Kokkos::parallel_for(tm->faces.nh(),
        VelocityKernel<velocity_field>(tm->velocity_faces.view, tm->faces.phys_crds->crds, 0));

      logger.info(tm->info_string());

      #ifdef LPM_USE_VTK
      {
          VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
          vtk.write("transport2d_icostrisphere_test0.vtp");
      }
      #endif

      Transport2dRK4<seed_type> solver(dt, tm);
      const Int nsteps = tfinal/dt;
      for (int time_idx = 0; time_idx<nsteps; ++time_idx) {
        solver.advance_timestep<velocity_field>();
        #ifdef LPM_USE_VTK
          if (time_idx == nsteps/2) {
            VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
            vtk.write("transport2d_icostrisphere_test1.vtp");
          }
        #endif
      }
      #ifdef LPM_USE_VTK
      {
          VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
          vtk.write("transport2d_icostrisphere_test2.vtp");
      }
      #endif

      Kokkos::View<Real*[3]> vert_position_error("vertex_position_error", tm->n_vertices_host());
      Kokkos::View<Real*[3]> face_position_error("face_position_error", tm->n_faces_host());

      const auto pxf = tm->faces.phys_crds->crds;
      const auto lxf = tm->faces.lag_crds->crds;

      auto face_err_norms = ErrNorms<>(face_position_error, pxf, lxf, tm->faces.area);
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

  SECTION("quadrilateral panels") {
    typedef CubedSphereSeed seed_type;

    std::vector<Real> l1;
    std::vector<Real> l2;
    std::vector<Real> linf;
    std::vector<Real> l1conv;
    std::vector<Real> l2conv;
    std::vector<Real> linfconv;

    for (const auto& ns : nsteps) {
      const Real dt = tfinal/ns;

      PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
      auto tm = std::shared_ptr<TransportMesh2d<seed_type>>(new TransportMesh2d<seed_type>(params));

      SphericalSlottedCylinders sc;
      SphericalCosineBells cb;
      SphericalGaussianHills gh;

      tm->initialize_tracer(gh);
      tm->initialize_tracer(cb);
      tm->initialize_tracer(sc);

      Kokkos::parallel_for(tm->vertices.nh(),
        VelocityKernel<velocity_field>(tm->velocity_verts.view, tm->vertices.phys_crds->crds, 0));
      Kokkos::parallel_for(tm->faces.nh(),
        VelocityKernel<velocity_field>(tm->velocity_faces.view, tm->faces.phys_crds->crds, 0));

      logger.info(tm->info_string());

      #ifdef LPM_USE_VTK
      {
          VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
          vtk.write("transport2d_icostrisphere_test0.vtp");
      }
      #endif

      Transport2dRK4<seed_type> solver(dt, tm);
      const Int nsteps = tfinal/dt;
      for (int time_idx = 0; time_idx<nsteps; ++time_idx) {
        solver.advance_timestep<velocity_field>();
        #ifdef LPM_USE_VTK
          if (time_idx == nsteps/2) {
            VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
            vtk.write("transport2d_icostrisphere_test1.vtp");
          }
        #endif
      }
      #ifdef LPM_USE_VTK
      {
          VtkPolymeshInterface<seed_type> vtk = vtk_interface(tm);
          vtk.write("transport2d_icostrisphere_test2.vtp");
      }
      #endif

      Kokkos::View<Real*[3]> vert_position_error("vertex_position_error", tm->n_vertices_host());
      Kokkos::View<Real*[3]> face_position_error("face_position_error", tm->n_faces_host());

      const auto pxf = tm->faces.phys_crds->crds;
      const auto lxf = tm->faces.lag_crds->crds;

      auto face_err_norms = ErrNorms<>(face_position_error, pxf, lxf, tm->faces.area);
      logger.debug("ns = {}, l1 = {}, l2 = {}, linf = {}", ns, face_err_norms.l1, face_err_norms.l2, face_err_norms.linf);

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
}
