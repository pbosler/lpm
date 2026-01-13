#include <iostream>
#include <sstream>

#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coords.hpp"
#include "lpm_coords_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "mesh/lpm_faces_impl.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_vertices_impl.hpp"
#include "util/lpm_floating_point.hpp"

using namespace Lpm;

TEST_CASE("faces test", "[mesh]") {
  Comm comm;

  Logger<> logger("faces_test", Log::level::info, comm);

  SECTION("triangle, plane") {
    using coords_type = Coords<PlaneGeometry>;

    const MeshSeed<TriHexSeed> thseed;
    Int nmaxverts, nmaxedges, nmaxfaces;
    thseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, 1);

    logger.info(
        "plane, tri memory allocations: {} vertices, {} edges, {} faces.",
        nmaxverts, nmaxedges, nmaxfaces);

    const int nmax_verts = 11;
    Vertices<Coords<PlaneGeometry>> tri_hex_verts(nmax_verts);
    tri_hex_verts.init_from_seed(thseed);
    logger.info("verts:\n {}", tri_hex_verts.info_string());

    Edges tri_hex_edges(nmaxedges);
    tri_hex_edges.init_from_seed(thseed);
    logger.info("edges:\n {}", tri_hex_edges.info_string());

    Faces<TriFace, PlaneGeometry> tri_hex_faces(11);
    tri_hex_faces.init_from_seed(thseed);

    logger.info("surface area = {}", tri_hex_faces.surface_area_host());
    REQUIRE(FloatingPoint<Real>::equiv(tri_hex_faces.surface_area_host(),
                                       2.59807621135331512,
                                       constants::ZERO_TOL));

    typedef FaceDivider<PlaneGeometry, TriFace> tri_hex_divider;
    logger.debug("calling divide.");
    tri_hex_divider::divide(0, tri_hex_verts, tri_hex_edges, tri_hex_faces);
    logger.debug("returned from divide.");

    REQUIRE(FloatingPoint<Real>::equiv(tri_hex_faces.surface_area_host(),
                                       2.59807621135331512,
                                       constants::ZERO_TOL));
    logger.info("tri hex faces:\n {}",
                tri_hex_faces.info_string("divide face 0"));

    tri_hex_faces.update_device();
    auto fcrds       = tri_hex_faces.phys_crds.get_host_crd_view();
    auto leaf_crds   = tri_hex_faces.leaf_crd_view();
    auto h_leaf_crds = Kokkos::create_mirror_view(leaf_crds);
    auto h_leaf_idx  = Kokkos::create_mirror_view(tri_hex_faces.leaf_idx);
    Kokkos::deep_copy(h_leaf_crds, leaf_crds);
    Kokkos::deep_copy(h_leaf_idx, tri_hex_faces.leaf_idx);

    for (int i = 0; i < tri_hex_faces.nh(); ++i) {
      logger.info(
          "face idx {} leaf idx {} (x, y) = ({}, {}), leaf(x,y) = ({}, {})", i,
          h_leaf_idx(i), fcrds(i, 0), fcrds(i, 1), leaf_crds(h_leaf_idx(i), 0),
          leaf_crds(h_leaf_idx(i), 1));
    }

    std::cout << "face tree:" << std::endl;
    ko::parallel_for(
        1, KOKKOS_LAMBDA(const int& i) {
          printf("------------Parallel region------------\n");
          printf("face 0 has kids = %s\n",
                 (tri_hex_faces.has_kids(0) ? "true" : "false"));
          printf("face 0 has mask = %s\n",
                 (tri_hex_faces.mask(0) ? "true" : "false"));
          for (int j = 0; j < tri_hex_faces.n(); ++j) {
            printf("face %d has parent %d and kids = (%d,%d,%d,%d)\n", j,
                   tri_hex_faces.parent(j), tri_hex_faces.kids(j, 0),
                   tri_hex_faces.kids(j, 1), tri_hex_faces.kids(j, 2),
                   tri_hex_faces.kids(j, 3));
          }
          printf("-----------end parallel region----------\n");
        });
    std::ostringstream ss;
    tri_hex_verts.phys_crds.write_matlab(ss, "vert_crds1");
    logger.info("matlab output:\n {}", ss.str());
    ss.str("");
    tri_hex_faces.phys_crds.write_matlab(ss, "face_crds1");
    logger.info("matlab output:\n {}", ss.str());
  }

  SECTION("quad, sphere") {
    typedef MeshSeed<CubedSphereSeed> seed_type;
    const seed_type seed;
    typedef SphereGeometry geo;
    typedef QuadFace face_type;
    using coords_type = Coords<SphereGeometry>;
    Index nmaxverts;
    Index nmaxfaces;
    Index nmaxedges;
    const int maxlev = 2;
    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, maxlev);
    logger.info(
        "quad, sphere memory allocations: {} vertices, {} edges, {} faces.",
        nmaxverts, nmaxedges, nmaxfaces);

    Vertices<coords_type> csverts(nmaxverts);
    csverts.init_from_seed(seed);
    logger.info("verts:\n {}", csverts.info_string());

    Edges csedges(nmaxedges);
    csedges.init_from_seed(seed);
    logger.info("edges:\n {}", csedges.info_string());

    Faces<QuadFace, SphereGeometry> csfaces(nmaxfaces);
    csfaces.init_from_seed(seed);

    logger.debug("cubed sphere init:\n {}", csfaces.info_string("", 0, true));

    REQUIRE(FloatingPoint<Real>::equiv(csfaces.surface_area_host(),
                                       4 * constants::PI, constants::ZERO_TOL));
    REQUIRE(csfaces.nh() == csfaces.phys_crds.nh());

    typedef FaceDivider<geo, face_type> csdiv;
    for (int i = 0; i < maxlev; ++i) {
      logger.debug(
          "tree level {}: faces.nh = {}, edges.nh = {}, verts.nh = {}; "
          "faces.n_max = {}",
          i, csfaces.nh(), csedges.nh(), csverts.nh(), csfaces.n_max());
      Index startInd = 0;
      Index stopInd  = csfaces.nh();
      for (Index j = startInd; j < stopInd; ++j) {
        if (!csfaces.has_kids_host(j)) {
          csdiv::divide(j, csverts, csedges, csfaces);
        }
      }
    }
    logger.info("cubed sphere surface area = {}; |area - 4*pi| = {}",
                csfaces.surface_area_host(),
                abs(csfaces.surface_area_host() - 4 * constants::PI));
    REQUIRE(FloatingPoint<Real>::equiv(csfaces.surface_area_host(),
                                       4 * constants::PI,
                                       1.5 * constants::ZERO_TOL));

#ifdef LPM_USE_VTK
    VtkInterface<SphereGeometry, QuadFace> vtk;
    auto pd = vtk.toVtkPolyData(csfaces, csedges, csverts);
    logger.debug("vtk data conversion done.");
    vtk.writePolyData("cs_test.vtk", pd);
#endif
  }

  SECTION("tri, sphere") {
    const MeshSeed<IcosTriSphereSeed> seed;
    Index nmaxverts;
    Index nmaxfaces;
    Index nmaxedges;
    const int maxlev = 5;
    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, maxlev);

    using coords_type = Coords<SphereGeometry>;
    Vertices<coords_type> icverts(nmaxverts);
    icverts.init_from_seed(seed);

    Edges icedges(nmaxedges);
    icedges.init_from_seed(seed);

    Faces<TriFace, SphereGeometry> icfaces(nmaxfaces);
    icfaces.init_from_seed(seed);

    REQUIRE(icfaces.nh() == IcosTriSphereSeed::nfaces);
    typedef FaceDivider<SphereGeometry, TriFace> icdiv;
    for (int i = 0; i < maxlev; ++i) {
      const Index stopInd = icfaces.nh();
      for (Index j = 0; j < stopInd; ++j) {
        if (!icfaces.has_kids_host(j)) {
          icdiv::divide(j, icverts, icedges, icfaces);
        }
      }
    }

    logger.info("icos. tri. sphere surface area = {}; |area - 4*pi| = {}",
                icfaces.surface_area_host(),
                abs(icfaces.surface_area_host() - 4 * constants::PI));

    REQUIRE(FloatingPoint<Real>::equiv(icfaces.surface_area_host(),
                                       4 * constants::PI,
                                       31 * constants::ZERO_TOL));

#ifdef LPM_USE_VTK
    VtkInterface<SphereGeometry, TriFace> vtk;
    logger.debug("writing vtk output.");
    vtkSmartPointer<vtkPolyData> pd =
        vtk.toVtkPolyData(icfaces, icedges, icverts);
    logger.debug("conversion to vtk polydata complete.");
    vtk.writePolyData("ic_test.vtk", pd);
#endif
  }
}
