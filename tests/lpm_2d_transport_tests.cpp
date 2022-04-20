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

TEST_CASE("spherical meshes", "") {
  Comm comm;
  Logger<> logger("lpm_tracer_init::spherical_mesh_test", Log::level::debug, comm);

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
