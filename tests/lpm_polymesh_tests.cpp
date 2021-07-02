#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "lpm_constants.hpp"
#include "catch.hpp"
#include <memory>
#include <sstream>

using namespace Lpm;

TEST_CASE("polymesh2d tests", "[mesh]") {

  Comm comm;

  Logger<> logger("faces_test", Log::level::info, comm);

  const int tree_lev = 3;

  SECTION("planar triangles") {
    MeshSeed<TriHexSeed> thseed;

    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    thseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    PolyMesh2d<TriHexSeed> triplane(nmaxverts, nmaxedges, nmaxfaces);
    triplane.tree_init(tree_lev, thseed);

    REQUIRE(triplane.vertices.nh() == nmaxverts);
    REQUIRE(triplane.edges.nh() == nmaxedges);
    REQUIRE(triplane.faces.nh() == nmaxfaces);

    triplane.output_vtk("triplane_test.vtk");
    triplane.update_device();
    logger.info("TriHexSeed mesh info:\n {}", triplane.info_string());

    REQUIRE(FloatingPoint<Real>::equiv(triplane.surface_area_host(), 2.59807621135331512,
      constants::ZERO_TOL));
  }

  SECTION("planar quads") {
    MeshSeed<QuadRectSeed> qrseed(4);
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;

    qrseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    PolyMesh2d<QuadRectSeed> quadplane(nmaxverts, nmaxedges, nmaxfaces);
    quadplane.tree_init(tree_lev, qrseed);

    REQUIRE(quadplane.vertices.nh() == nmaxverts);
    REQUIRE(quadplane.edges.nh() == nmaxedges);
    REQUIRE(quadplane.faces.nh() == nmaxfaces);


    quadplane.output_vtk("quadplane_test.vtk");
    quadplane.update_device();
    logger.info("QuadRectSeed mesh info:\n {}", quadplane.info_string("radius = 4"));

    REQUIRE(FloatingPoint<Real>::equiv(quadplane.surface_area_host(), 64));
  }

  SECTION("spherical triangles") {
    MeshSeed<IcosTriSphereSeed> icseed;

    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;

    icseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    PolyMesh2d<IcosTriSphereSeed> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.tree_init(tree_lev, icseed);

    REQUIRE(trisphere.vertices.nh() == nmaxverts);
    REQUIRE(trisphere.edges.nh() == nmaxedges);
    REQUIRE(trisphere.faces.nh() == nmaxfaces);

    trisphere.output_vtk("trisphere_test.vtk");
    trisphere.update_device();
    logger.info("IcosTriSphereSeed mesh info:\n {}", trisphere.info_string());

    REQUIRE(FloatingPoint<Real>::equiv(trisphere.surface_area_host(), 4*constants::PI,
      31*constants::ZERO_TOL));
  }

  SECTION("cubed sphere") {
    MeshSeed<CubedSphereSeed> csseed;
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    csseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    PolyMesh2d<CubedSphereSeed> quadsphere(nmaxverts, nmaxedges, nmaxfaces);
    quadsphere.tree_init(tree_lev, csseed);

    REQUIRE(quadsphere.vertices.nh() == nmaxverts);
    REQUIRE(quadsphere.edges.nh() == nmaxedges);
    REQUIRE(quadsphere.faces.nh() == nmaxfaces);

    quadsphere.output_vtk("quadsphere_test.vtk");
    quadsphere.update_device();
    logger.info("CubedSphereSeed mesh info:\n {}", quadsphere.info_string());

    logger.debug("cubed sphere area = {}, |area = 4*pi| = {}",
      quadsphere.surface_area_host(), abs(quadsphere.surface_area_host() - 4*constants::PI));

    REQUIRE(FloatingPoint<Real>::equiv(quadsphere.surface_area_host(), 4*constants::PI,
      3.5*constants::ZERO_TOL));
  }

}
