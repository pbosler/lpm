  #include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_floating_point.hpp"
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif
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
    auto triplane = std::shared_ptr<PolyMesh2d<TriHexSeed>>(new PolyMesh2d<TriHexSeed>(nmaxverts, nmaxedges, nmaxfaces));
    triplane->tree_init(tree_lev, thseed);

    REQUIRE(triplane->vertices.nh() == nmaxverts);
    REQUIRE(triplane->edges.nh() == nmaxedges);
    REQUIRE(triplane->faces.nh() == nmaxfaces);

    triplane->output_vtk("triplane_test.vtk");
    triplane->update_device();
    logger.info("TriHexSeed mesh info:\n {}", triplane->info_string());

    REQUIRE(FloatingPoint<Real>::equiv(triplane->surface_area_host(), 2.59807621135331512,
      constants::ZERO_TOL));

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<TriHexSeed> vtk(triplane);
    vtk.write("triplane_vtk_test.vtp");
#endif
  }

  SECTION("planar quads") {
    MeshSeed<QuadRectSeed> qrseed(4);
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;

    qrseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);

    auto quadplane = std::shared_ptr<PolyMesh2d<QuadRectSeed>>(new
      PolyMesh2d<QuadRectSeed>(nmaxverts, nmaxedges, nmaxfaces));
    quadplane->tree_init(tree_lev, qrseed);

    REQUIRE(quadplane->vertices.nh() == nmaxverts);
    REQUIRE(quadplane->edges.nh() == nmaxedges);
    REQUIRE(quadplane->faces.nh() == nmaxfaces);


    quadplane->output_vtk("quadplane_test.vtk");
    quadplane->update_device();
    logger.info("QuadRectSeed mesh info:\n {}", quadplane->info_string("radius = 4"));

    REQUIRE(FloatingPoint<Real>::equiv(quadplane->surface_area_host(), 64));

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<QuadRectSeed> vtk(quadplane);
    vtk.write("quadplane_vtk_test.vtp");
#endif
  }

  SECTION("spherical triangles") {
    MeshSeed<IcosTriSphereSeed> icseed;

    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;

    icseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    auto trisphere = std::shared_ptr<PolyMesh2d<IcosTriSphereSeed>>(new
      PolyMesh2d<IcosTriSphereSeed>(nmaxverts, nmaxedges, nmaxfaces));
    trisphere->tree_init(tree_lev, icseed);

    REQUIRE(trisphere->vertices.nh() == nmaxverts);
    REQUIRE(trisphere->edges.nh() == nmaxedges);
    REQUIRE(trisphere->faces.nh() == nmaxfaces);

    trisphere->output_vtk("trisphere_test.vtk");
    trisphere->update_device();
    logger.info("IcosTriSphereSeed mesh info:\n {}", trisphere->info_string());

    REQUIRE(FloatingPoint<Real>::equiv(trisphere->surface_area_host(), 4*constants::PI,
      31*constants::ZERO_TOL));

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<IcosTriSphereSeed> vtk(trisphere);
    vtk.write("icosstri_vtk_test.vtp");
#endif
  }

  SECTION("cubed sphere") {
    MeshSeed<CubedSphereSeed> csseed;
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;
    csseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
    auto quadsphere = std::shared_ptr<PolyMesh2d<CubedSphereSeed>>(new
      PolyMesh2d<CubedSphereSeed>(nmaxverts, nmaxedges, nmaxfaces));
    quadsphere->tree_init(tree_lev, csseed);

    REQUIRE(quadsphere->vertices.nh() == nmaxverts);
    REQUIRE(quadsphere->edges.nh() == nmaxedges);
    REQUIRE(quadsphere->faces.nh() == nmaxfaces);

    quadsphere->output_vtk("quadsphere_test.vtk");
    quadsphere->update_device();
    logger.info("CubedSphereSeed mesh info:\n {}", quadsphere->info_string());

    logger.debug("cubed sphere area = {}, |area = 4*pi| = {}",
      quadsphere->surface_area_host(), abs(quadsphere->surface_area_host() - 4*constants::PI));

    REQUIRE(FloatingPoint<Real>::equiv(quadsphere->surface_area_host(), 4*constants::PI,
      3.5*constants::ZERO_TOL));

#ifdef LPM_USE_VTK
    VtkPolymeshInterface<CubedSphereSeed> vtk(quadsphere);
    vtk.write("cubed_sphere_vtk_test.vtp");
#endif
  }

}

TEST_CASE("polymesh/netcdf", "[mesh]") {

  // typedef CubedSphereSeed seed_type;
  typedef IcosTriSphereSeed seed_type;

  Comm comm;

  Logger<> logger("polymesh/netcdf test", Log::level::info, comm);

  const int tree_lev = 3;

  MeshSeed<seed_type> csseed;
  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;
  csseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, tree_lev);
  auto quadsphere = std::shared_ptr<PolyMesh2d<seed_type>>(new
    PolyMesh2d<seed_type>(nmaxverts, nmaxedges, nmaxfaces));
  quadsphere->tree_init(tree_lev, csseed);

  ScalarField<FaceField> ones("ones", nmaxfaces, ekat::units::Units::nondimensional());
  ko::deep_copy(ones.hview, 1.0);

  VectorField<SphereGeometry, VertexField> twos("twos", nmaxverts, ekat::units::Units::nondimensional());
  ko::deep_copy(twos.hview, 2.0);

  REQUIRE(FloatingPoint<Real>::equiv(quadsphere->faces.surface_area_host(),
    4*constants::PI, 20*constants::ZERO_TOL));

  SECTION("NcWriter") {
    NcWriter<SphereGeometry> ncwriter("polymesh_netcdf_test.nc");

    ncwriter.define_polymesh(*quadsphere);
    ncwriter.define_scalar_field(ones);
    ncwriter.put_scalar_field(0, ones);
    ncwriter.define_vector_field(twos);
    ncwriter.put_vector_field(0, twos);

    logger.info(ncwriter.info_string());
  }

  SECTION("NcReader") {

    PolymeshReader ncreader("polymesh_netcdf_test.nc");

    auto mesh = ncreader.init_polymesh<seed_type>();
    logger.info(ncreader.info_string());

    REQUIRE(mesh);

    // check that vertices are the same
    REQUIRE(quadsphere->vertices.n_max() == mesh->vertices.n_max());
    REQUIRE(quadsphere->vertices.nh() == mesh->vertices.nh());
    for (Index i=0; i<quadsphere->n_vertices_host(); ++i) {
      REQUIRE(quadsphere->vertices.host_crd_ind(i) == mesh->vertices.host_crd_ind(i));
      for (Index j=0; j<SphereGeometry::ndim; ++j) {
        REQUIRE(FloatingPoint<Real>::equiv(quadsphere->vertices.phys_crds->get_crd_component_host(i,j),
          mesh->vertices.phys_crds->get_crd_component_host(i,j)));
        REQUIRE(FloatingPoint<Real>::equiv(quadsphere->vertices.lag_crds->get_crd_component_host(i,j),
          mesh->vertices.lag_crds->get_crd_component_host(i,j)));
      }
    }

    // check that edges are the same
    REQUIRE(quadsphere->edges.n_max() == mesh->edges.n_max());
    REQUIRE(quadsphere->edges.nh() == mesh->edges.nh());
    REQUIRE(quadsphere->edges.n_leaves_host() == mesh->edges.n_leaves_host());
    for (Index i=0; i<quadsphere->n_edges_host(); ++i) {
      REQUIRE(quadsphere->edges.orig_host(i) == mesh->edges.orig_host(i));
      REQUIRE(quadsphere->edges.dest_host(i) == mesh->edges.dest_host(i));
      REQUIRE(quadsphere->edges.left_host(i) == mesh->edges.left_host(i));
      REQUIRE(quadsphere->edges.parent_host(i) == mesh->edges.parent_host(i));
      for (int k=0; k<2; ++k) {
        REQUIRE(quadsphere->edges.kid_host(i,k) == mesh->edges.kid_host(i,k));
      }
    }

    // check that faces are the same
    REQUIRE(quadsphere->faces.n_max() == mesh->faces.n_max());
    REQUIRE(quadsphere->faces.nh() == mesh->faces.nh());
    REQUIRE(quadsphere->faces.n_leaves_host() == mesh->faces.n_leaves_host());
    REQUIRE(FloatingPoint<Real>::equiv(quadsphere->faces.surface_area_host(),
      mesh->faces.surface_area_host()));

    for (Index i=0; i<quadsphere->n_faces_host(); ++i) {
      REQUIRE(quadsphere->faces.parent_host(i) == mesh->faces.parent_host(i));
      REQUIRE(quadsphere->faces.crd_idx_host(i) == mesh->faces.crd_idx_host(i));
      REQUIRE(FloatingPoint<Real>::equiv(quadsphere->faces.area_host(i),
        mesh->faces.area_host(i)));
      REQUIRE(quadsphere->faces.level_host(i) == mesh->faces.level_host(i));
      for (Index j=0; j<seed_type::faceKind::nverts; ++j) {
        REQUIRE(quadsphere->faces.vert_host(i,j) == mesh->faces.vert_host(i,j));
        REQUIRE(quadsphere->faces.edge_host(i,j) == mesh->faces.edge_host(i,j));
      }
      for (Index j=0; j<4; ++j) {
        REQUIRE(quadsphere->faces.kid_host(i,j) == mesh->faces.kid_host(i,j));
      }
      for (Index j=0; j<SphereGeometry::ndim; ++j) {
        REQUIRE(FloatingPoint<Real>::equiv(quadsphere->faces.phys_crds->get_crd_component_host(i,j),
          mesh->faces.phys_crds->get_crd_component_host(i,j)));
        REQUIRE(FloatingPoint<Real>::equiv(quadsphere->faces.lag_crds->get_crd_component_host(i,j),
          mesh->faces.lag_crds->get_crd_component_host(i,j)));
      }
    }
  }
}

