#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_field.hpp"
#include "lpm_tracer_gallery.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_floating_point.hpp"
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <memory>
#include <sstream>

using namespace Lpm;

TEST_CASE("planar particles", "") {
  Comm comm;
  Logger<> logger("lpm_tracer_init::planar_particles_test", Log::level::info, comm);
}

TEST_CASE("planar mesh", "") {
  Comm comm;
  Logger<> logger("lpm_tracer_init::planar_mesh_test", Log::level::info, comm);

  const int tree_lev = 5;
  const int amr_limit = 2;
  const Real radius = 2;

  PlanarHump ph;
  PlanarSlottedDisk pd;
  PlanarCone pc;

  SECTION("triangular panels") {
    typedef TriHexSeed seed_type;
    PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
    auto pm = PolyMesh2d<seed_type>(params);
    logger.info(pm.info_string());

    std::map<std::string, ScalarField<VertexField>> tracer_verts; /// passive tracers at passive particles
    std::map<std::string, ScalarField<FaceField>> tracer_faces; /// passive tracers at active particles

    tracer_verts.emplace(ph.name(), ScalarField<VertexField>(ph.name(), params.nmaxverts));
    tracer_verts.emplace(pd.name(), ScalarField<VertexField>(pd.name(), params.nmaxverts));
    tracer_verts.emplace(pc.name(), ScalarField<VertexField>(pc.name(), params.nmaxverts));
    tracer_verts.emplace("sum_all", ScalarField<VertexField>("sum_all", params.nmaxverts));

    tracer_faces.emplace(ph.name(), ScalarField<FaceField>(ph.name(), params.nmaxfaces));
    tracer_faces.emplace(pd.name(), ScalarField<FaceField>(pd.name(), params.nmaxfaces));
    tracer_faces.emplace(pc.name(), ScalarField<FaceField>(pc.name(), params.nmaxfaces));
    tracer_faces.emplace("sum_all", ScalarField<FaceField>("sum_all", params.nmaxfaces));

    auto phv = tracer_verts.at(ph.name()).view;
    auto pdv = tracer_verts.at(pd.name()).view;
    auto pcv = tracer_verts.at(pc.name()).view;
    auto sav = tracer_verts.at("sum_all").view;
    auto xyv = pm.vertices.phys_crds.view;
    Kokkos::parallel_for(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(xyv, i, Kokkos::ALL);
        phv(i) = ph(xy);
        pdv(i) = pd(xy);
        pcv(i) = pc(xy);
        sav(i) = phv(i) + pdv(i) + pcv(i);
    });

    auto phf = tracer_faces.at(ph.name()).view;
    auto pdf = tracer_faces.at(pd.name()).view;
    auto pcf = tracer_faces.at(pc.name()).view;
    auto saf = tracer_faces.at("sum_all").view;
    auto xyf = pm.faces.phys_crds.view;
    Kokkos::parallel_for(pm.faces.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(xyf, i, Kokkos::ALL);
        phf(i) = ph(xy);
        pdf(i) = pd(xy);
        pcf(i) = pc(xy);
        saf(i) = phf(i) + pdf(i) + pcf(i);
    });

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<seed_type> vtk(pm);
    vtk.add_scalar_point_data(phv, ph.name());
    vtk.add_scalar_point_data(pdv, pd.name());
    vtk.add_scalar_point_data(pcv, pc.name());
    vtk.add_scalar_point_data(sav, "sum_all");

    vtk.add_scalar_cell_data(phf, ph.name());
    vtk.add_scalar_cell_data(pdf, pd.name());
    vtk.add_scalar_cell_data(pcf, pc.name());
    vtk.add_scalar_cell_data(saf, "sum_all");

    vtk.write("tracer_init_test_trihex.vtp");
#endif

    typename Kokkos::MinMax<Real>::value_type sc_minmax_verts;
    Kokkos::parallel_reduce(pm.vertices.nh(),
      KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
        if (pdv(i) < mm.min_val) mm.min_val = pdv(i);
        if (pdv(i) > mm.max_val) mm.max_val = pdv(i);
      }, Kokkos::MinMax<Real>(sc_minmax_verts));

    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.min_val, 0));
    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.max_val, 1));

  }

  SECTION("quadrilateral panels")
  {
    typedef QuadRectSeed seed_type;
    PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
    auto pm = PolyMesh2d<seed_type>(params);
    logger.info(pm.info_string());

    std::map<std::string, ScalarField<VertexField>> tracer_verts; /// passive tracers at passive particles
    std::map<std::string, ScalarField<FaceField>> tracer_faces; /// passive tracers at active particles

    tracer_verts.emplace(ph.name(), ScalarField<VertexField>(ph.name(), params.nmaxverts));
    tracer_verts.emplace(pd.name(), ScalarField<VertexField>(pd.name(), params.nmaxverts));
    tracer_verts.emplace(pc.name(), ScalarField<VertexField>(pc.name(), params.nmaxverts));
    tracer_verts.emplace("sum_all", ScalarField<VertexField>("sum_all", params.nmaxverts));

    tracer_faces.emplace(ph.name(), ScalarField<FaceField>(ph.name(), params.nmaxfaces));
    tracer_faces.emplace(pd.name(), ScalarField<FaceField>(pd.name(), params.nmaxfaces));
    tracer_faces.emplace(pc.name(), ScalarField<FaceField>(pc.name(), params.nmaxfaces));
    tracer_faces.emplace("sum_all", ScalarField<FaceField>("sum_all", params.nmaxfaces));

    auto phv = tracer_verts.at(ph.name()).view;
    auto pdv = tracer_verts.at(pd.name()).view;
    auto pcv = tracer_verts.at(pc.name()).view;
    auto sav = tracer_verts.at("sum_all").view;
    auto xyv = pm.vertices.phys_crds.view;
    Kokkos::parallel_for(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(xyv, i, Kokkos::ALL);
        phv(i) = ph(xy);
        pdv(i) = pd(xy);
        pcv(i) = pc(xy);
        sav(i) = phv(i) + pdv(i) + pcv(i);
    });

    auto phf = tracer_faces.at(ph.name()).view;
    auto pdf = tracer_faces.at(pd.name()).view;
    auto pcf = tracer_faces.at(pc.name()).view;
    auto saf = tracer_faces.at("sum_all").view;
    auto xyf = pm.faces.phys_crds.view;
    Kokkos::parallel_for(pm.faces.nh(), KOKKOS_LAMBDA (const Index i) {
        const auto xy = Kokkos::subview(xyf, i, Kokkos::ALL);
        phf(i) = ph(xy);
        pdf(i) = pd(xy);
        pcf(i) = pc(xy);
        saf(i) = phf(i) + pdf(i) + pcf(i);
    });

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<seed_type> vtk(pm);
    vtk.add_scalar_point_data(phv, ph.name());
    vtk.add_scalar_point_data(pdv, pd.name());
    vtk.add_scalar_point_data(pcv, pc.name());
    vtk.add_scalar_point_data(sav, "sum_all");

    vtk.add_scalar_cell_data(phf, ph.name());
    vtk.add_scalar_cell_data(pdf, pd.name());
    vtk.add_scalar_cell_data(pcf, pc.name());
    vtk.add_scalar_cell_data(saf, "sum_all");

    vtk.write("tracer_init_test_quadrect.vtp");
#endif

    typename Kokkos::MinMax<Real>::value_type sc_minmax_verts;
    Kokkos::parallel_reduce(pm.vertices.nh(),
      KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
        if (pdv(i) < mm.min_val) mm.min_val = pdv(i);
        if (pdv(i) > mm.max_val) mm.max_val = pdv(i);
      }, Kokkos::MinMax<Real>(sc_minmax_verts));

    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.min_val, 0));
    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.max_val, 1));
  }
}

TEST_CASE("spherical particles", "") {
  Comm comm;
  Logger<> logger("lpm_tracer_init::spherical_particles_test", Log::level::info, comm);

}

TEST_CASE("spherical mesh", "") {
  Comm comm;
  Logger<> logger("lpm_tracer_init::spherical_mesh_test", Log::level::debug, comm);

  const int tree_lev = 5;
  const int amr_limit = 2;
  const Real radius = 1;

  SphericalSlottedCylinders sc;
  SphericalCosineBells cb;
  SphericalGaussianHills gh;
  MovingVorticesTracer mv;
  LatitudeTracer lt;

  SECTION("triangular panels") {
    typedef IcosTriSphereSeed seed_type;
    PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
    auto pm = PolyMesh2d<seed_type>(params);
    logger.info(pm.info_string());

    std::map<std::string, ScalarField<VertexField>> tracer_verts; /// passive tracers at passive particles
    std::map<std::string, ScalarField<FaceField>> tracer_faces; /// passive tracers at active particles

    tracer_verts.emplace(gh.name(), ScalarField<VertexField>(gh.name(), params.nmaxverts));
    tracer_verts.emplace(cb.name(), ScalarField<VertexField>(cb.name(), params.nmaxverts));
    tracer_verts.emplace(sc.name(), ScalarField<VertexField>(sc.name(), params.nmaxverts));
    tracer_verts.emplace(mv.name(), ScalarField<VertexField>(mv.name(), params.nmaxverts));
    tracer_verts.emplace(lt.name(), ScalarField<VertexField>(lt.name(), params.nmaxverts));

    tracer_faces.emplace(gh.name(), ScalarField<FaceField>(gh.name(), params.nmaxfaces));
    tracer_faces.emplace(cb.name(), ScalarField<FaceField>(cb.name(), params.nmaxfaces));
    tracer_faces.emplace(sc.name(), ScalarField<FaceField>(sc.name(), params.nmaxfaces));
    tracer_faces.emplace(mv.name(), ScalarField<FaceField>(mv.name(), params.nmaxfaces));
    tracer_faces.emplace(lt.name(), ScalarField<FaceField>(lt.name(), params.nmaxfaces));

    const Int ntracers = tracer_verts.size();

    auto ghviewv = tracer_verts.at(gh.name()).view;
    auto cbviewv = tracer_verts.at(cb.name()).view;
    auto scviewv = tracer_verts.at(sc.name()).view;
    auto mvviewv = tracer_verts.at(mv.name()).view;
    auto ltviewv = tracer_verts.at(lt.name()).view;

    const auto pcrdsv = pm.vertices.phys_crds.view;

    Kokkos::parallel_for(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i) {
      const auto xyz = Kokkos::subview(pcrdsv, i, Kokkos::ALL);
      ghviewv(i) = gh(xyz);
      cbviewv(i) = cb(xyz);
      scviewv(i) = sc(xyz);
      mvviewv(i) = mv(xyz);
      ltviewv(i) = lt(xyz);
    });

    tracer_verts.at(sc.name()).update_host();
    const auto schview = tracer_verts.at(sc.name()).hview;
    for (Index i=0; i<pm.vertices.nh(); ++i) {
        if ( !(FloatingPoint<Real>::equiv(schview(i), 0.1) or FloatingPoint<Real>::equiv(schview(i), 1))) {
        logger.error("unexpected value at vertex {}: sc = {}", i, schview(i));
      }
    }


    auto ghviewf = tracer_faces.at(gh.name()).view;
    auto cbviewf = tracer_faces.at(cb.name()).view;
    auto scviewf = tracer_faces.at(sc.name()).view;
    auto mvviewf = tracer_faces.at(mv.name()).view;
    auto ltviewf = tracer_faces.at(lt.name()).view;

    const auto pcrdsf = pm.faces.phys_crds.view;
    Kokkos::parallel_for(pm.faces.nh(), KOKKOS_LAMBDA (const Index i) {
      const auto xyz = Kokkos::subview(pcrdsf, i, Kokkos::ALL);
      ghviewf(i) = gh(xyz);
      cbviewf(i) = cb(xyz);
      scviewf(i) = sc(xyz);
      mvviewf(i) = mv(xyz);
      ltviewf(i) = lt(xyz);
    });

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<seed_type> vtk(pm);
    vtk.add_scalar_point_data(ghviewv, gh.name());
    vtk.add_scalar_point_data(cbviewv, cb.name());
    vtk.add_scalar_point_data(scviewv, sc.name());
    vtk.add_scalar_point_data(mvviewv, mv.name());
    vtk.add_scalar_point_data(ltviewv, lt.name());

    vtk.add_scalar_cell_data(ghviewf, gh.name());
    vtk.add_scalar_cell_data(cbviewf, cb.name());
    vtk.add_scalar_cell_data(scviewf, sc.name());
    vtk.add_scalar_cell_data(mvviewf, mv.name());
    vtk.add_scalar_cell_data(ltviewf, lt.name());

    vtk.write("tracer_init_test_icos_tri_sphere.vtp");
#endif
    const auto sc_verts = tracer_verts.at("SphericalSlottedCylinders").view;
    typename Kokkos::MinMax<Real>::value_type sc_minmax_verts;
    Kokkos::parallel_reduce(pm.vertices.nh(),
      KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
        if (sc_verts(i) > mm.max_val) mm.max_val = sc_verts(i);
        if (sc_verts(i) < mm.min_val) mm.min_val = sc_verts(i);
      }, Kokkos::MinMax<Real>(sc_minmax_verts));
    logger.debug("sc_minmax = ({}, {})", sc_minmax_verts.min_val, sc_minmax_verts.max_val);

    Real sc_vert_min;
    Real sc_vert_max;
    Kokkos::parallel_reduce(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i, Real& m) {
      if (sc_verts(i) < m) m = sc_verts(i);
    }, Kokkos::Min<Real>(sc_vert_min));
    Kokkos::parallel_reduce(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i, Real& m) {
      if (sc_verts(i) > m) m = sc_verts(i);
    }, Kokkos::Max<Real>(sc_vert_max));
    logger.debug("(sc_min, sc_max) = ({}, {})", sc_vert_min, sc_vert_max);

    REQUIRE(FloatingPoint<Real>::equiv(sc_vert_min, 0.1));
    REQUIRE(FloatingPoint<Real>::equiv(sc_vert_max, 1));
    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.min_val, 0.1));
    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.max_val, 1.0));

  }

  SECTION("quadrilateral panels") {
    typedef CubedSphereSeed seed_type;
    PolyMeshParameters<seed_type> params(tree_lev, radius, amr_limit);
    auto pm = PolyMesh2d<seed_type>(params);
    std::map<std::string, ScalarField<VertexField>> tracer_verts; /// passive tracers at passive particles
    std::map<std::string, ScalarField<FaceField>> tracer_faces; /// passive tracers at active particles

    logger.info(pm.info_string());

    tracer_verts.emplace(gh.name(), ScalarField<VertexField>(gh.name(), params.nmaxverts));
    tracer_verts.emplace(cb.name(), ScalarField<VertexField>(cb.name(), params.nmaxverts));
    tracer_verts.emplace(sc.name(), ScalarField<VertexField>(sc.name(), params.nmaxverts));
    tracer_verts.emplace(mv.name(), ScalarField<VertexField>(mv.name(), params.nmaxverts));
    tracer_verts.emplace(lt.name(), ScalarField<VertexField>(lt.name(), params.nmaxverts));

    tracer_faces.emplace(gh.name(), ScalarField<FaceField>(gh.name(), params.nmaxfaces));
    tracer_faces.emplace(cb.name(), ScalarField<FaceField>(cb.name(), params.nmaxfaces));
    tracer_faces.emplace(sc.name(), ScalarField<FaceField>(sc.name(), params.nmaxfaces));
    tracer_faces.emplace(mv.name(), ScalarField<FaceField>(mv.name(), params.nmaxfaces));
    tracer_faces.emplace(lt.name(), ScalarField<FaceField>(lt.name(), params.nmaxfaces));

    const Int ntracers = tracer_verts.size();

    auto ghviewv = tracer_verts.at(gh.name()).view;
    auto cbviewv = tracer_verts.at(cb.name()).view;
    auto scviewv = tracer_verts.at(sc.name()).view;
    auto mvviewv = tracer_verts.at(mv.name()).view;
    auto ltviewv = tracer_verts.at(lt.name()).view;

    const auto pcrdsv = pm.vertices.phys_crds.view;

    Kokkos::parallel_for(pm.vertices.nh(), KOKKOS_LAMBDA (const Index i) {
      const auto xyz = Kokkos::subview(pcrdsv, i, Kokkos::ALL);
      ghviewv(i) = gh(xyz);
      cbviewv(i) = cb(xyz);
      scviewv(i) = sc(xyz);
      mvviewv(i) = mv(xyz);
      ltviewv(i) = lt(xyz);
    });

    auto ghviewf = tracer_faces.at(gh.name()).view;
    auto cbviewf = tracer_faces.at(cb.name()).view;
    auto scviewf = tracer_faces.at(sc.name()).view;
    auto mvviewf = tracer_faces.at(mv.name()).view;
    auto ltviewf = tracer_faces.at(lt.name()).view;

    const auto pcrdsf = pm.faces.phys_crds.view;
    Kokkos::parallel_for(pm.faces.nh(), KOKKOS_LAMBDA (const Index i) {
      const auto xyz = Kokkos::subview(pcrdsf, i, Kokkos::ALL);
      ghviewf(i) = gh(xyz);
      cbviewf(i) = cb(xyz);
      scviewf(i) = sc(xyz);
      mvviewf(i) = mv(xyz);
      ltviewf(i) = lt(xyz);
    });

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<seed_type> vtk(pm);
    vtk.add_scalar_point_data(ghviewv, gh.name());
    vtk.add_scalar_point_data(cbviewv, cb.name());
    vtk.add_scalar_point_data(scviewv, sc.name());
    vtk.add_scalar_point_data(mvviewv, mv.name());
    vtk.add_scalar_point_data(ltviewv, lt.name());

    vtk.add_scalar_cell_data(ghviewf, gh.name());
    vtk.add_scalar_cell_data(cbviewf, cb.name());
    vtk.add_scalar_cell_data(scviewf, sc.name());
    vtk.add_scalar_cell_data(mvviewf, mv.name());
    vtk.add_scalar_cell_data(ltviewf, lt.name());

    vtk.write("tracer_init_test_cubed_sphere.vtp");
#endif
    const auto sc_verts = tracer_verts.at("SphericalSlottedCylinders").view;
    typename Kokkos::MinMax<Real>::value_type sc_minmax_verts;
    Kokkos::parallel_reduce(pm.vertices.nh(),
      KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
        if (sc_verts(i) > mm.max_val) mm.max_val = sc_verts(i);
        if (sc_verts(i) < mm.min_val) mm.min_val = sc_verts(i);
      }, Kokkos::MinMax<Real>(sc_minmax_verts));

    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.min_val, 0.1));
    REQUIRE(FloatingPoint<Real>::equiv(sc_minmax_verts.max_val, 1.0));
  }
}
