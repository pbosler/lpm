#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "catch.hpp"
#include <memory>

using namespace Lpm;

TEST_CASE ("faces test", "[mesh]") {

  Comm comm;

  Logger<> logger("faces_test", Log::level::info, comm);

  SECTION("triangle, plane") {
    using coords_type = Coords<PlaneGeometry>;

    Faces<TriFace, PlaneGeometry> plane_tri(11);
    const MeshSeed<TriHexSeed> thseed;
    Index nmaxverts;
    Index nmaxfaces;
    Index nmaxedges;
    thseed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, 1);

    logger.info("plane, tri memory allocations: {} vertices, {} edges, {} faces.",
      nmaxverts, nmaxedges, nmaxfaces);

    const int nmax_verts = 11;

    auto tri_hex_pcrd_verts = std::shared_ptr<coords_type>(
      new coords_type(nmax_verts));
    auto tri_hex_lcrd_verts = std::shared_ptr<coords_type>(
      new coords_type(nmax_verts));
    auto tri_hex_pcrd_face_crds = std::shared_ptr<coords_type>(new coords_type(nmax_verts));
    auto tri_hex_lcrd_face_crds = std::shared_ptr<coords_type>(new coords_type(nmax_verts));
    Edges tri_hex_edges(24);

    tri_hex_pcrd_verts->init_vert_crds_from_seed(thseed);
    tri_hex_lcrd_verts->init_vert_crds_from_seed(thseed);
    std::stringstream ss;
    tri_hex_pcrd_verts->write_matlab(ss, "vert_crds0");
    logger.info("matlab output:\n {}", ss.str());
    ss.str("");

    Vertices<Coords<PlaneGeometry>> verts(nmax_verts, tri_hex_pcrd_verts, tri_hex_lcrd_verts);
    logger.info("verts:\n {}", verts.info_string());


    tri_hex_pcrd_face_crds->init_interior_crds_from_seed(thseed);
    tri_hex_lcrd_face_crds->init_interior_crds_from_seed(thseed);
    tri_hex_edges.init_from_seed(thseed);
    logger.info("edges:\n {}", tri_hex_edges.info_string());

    plane_tri.phys_crds = tri_hex_pcrd_face_crds;
    plane_tri.lag_crds = tri_hex_lcrd_face_crds;
    plane_tri.init_from_seed(thseed);

    logger.debug("surface area = {}", plane_tri.surface_area_host());
    REQUIRE(FloatingPoint<Real>::equiv(plane_tri.surface_area_host(), 2.59807621135331512,
      constants::ZERO_TOL));

    typedef FaceDivider<PlaneGeometry,TriFace> tri_hex_divider;
    tri_hex_divider::divide(0, verts, tri_hex_edges, plane_tri);

    REQUIRE(FloatingPoint<Real>::equiv(plane_tri.surface_area_host(), 2.59807621135331512,
      constants::ZERO_TOL));
    logger.info("tri hex faces:\n {}", plane_tri.info_string("divide face 0"));


    plane_tri.update_device();
    std::cout << "face tree:" << std::endl;
    ko::parallel_for(1, KOKKOS_LAMBDA (const int& i) {
        printf("------------Parallel region------------\n");
        printf("face 0 has kids = %s\n", (plane_tri.has_kids(0) ? "true" : "false"));
        printf("face 0 has mask = %s\n", (plane_tri.mask(0) ? "true" : "false"));
        for (int j=0; j<plane_tri.n(); ++j) {
            printf("face %d has parent %d and kids = (%d,%d,%d,%d)\n", j, plane_tri.parent(j),
                plane_tri.kids(j,0), plane_tri.kids(j,1), plane_tri.kids(j,2), plane_tri.kids(j,3));
        }
        printf("-----------end parallel region----------\n");
    });
    tri_hex_pcrd_verts->write_matlab(ss, "vert_crds1");
    logger.info("matlab output:\n {}", ss.str());
    ss.str("");
    tri_hex_pcrd_face_crds->write_matlab(ss, "face_crds1");
    logger.info("matlab output:\n {}", ss.str());
  }

  SECTION("quad, sphere")  {
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
        logger.info("quad, sphere memory allocations: {} vertices, {} edges, {} faces.",
      nmaxverts, nmaxedges, nmaxfaces);

        auto cs_verts_pcrd = std::shared_ptr<coords_type>(new coords_type(nmaxverts));
        auto cs_verts_lcrd = std::shared_ptr<coords_type>(new coords_type(nmaxverts));
        cs_verts_pcrd->init_vert_crds_from_seed(seed);
        cs_verts_lcrd->init_vert_crds_from_seed(seed);

        Vertices<coords_type> csverts(nmaxverts, cs_verts_pcrd, cs_verts_lcrd);
        logger.info("verts:\n {}", csverts.info_string());

        Edges csedges(nmaxedges);
        csedges.init_from_seed(seed);
        logger.info("edges:\n {}", csedges.info_string());

        Faces<QuadFace, SphereGeometry> csfaces(nmaxfaces);
        auto csfacecrds = std::shared_ptr<coords_type>(new coords_type(nmaxfaces));
        auto cslagfacecrds = std::shared_ptr<coords_type>(new coords_type(nmaxfaces));
        csfacecrds->init_interior_crds_from_seed(seed);
        cslagfacecrds->init_interior_crds_from_seed(seed);
        csfaces.init_from_seed(seed);

        csfaces.phys_crds = csfacecrds;
        csfaces.lag_crds = cslagfacecrds;

        logger.debug("cubed sphere init:\n {}", csfaces.info_string("", 0, true));

        REQUIRE(FloatingPoint<Real>::equiv(csfaces.surface_area_host(),
          4*constants::PI, constants::ZERO_TOL));
        REQUIRE(csfaces.nh() == csfaces.phys_crds->nh());

        typedef FaceDivider<geo,face_type> csdiv;
        for (int i=0; i<maxlev; ++i) {
          logger.debug("tree level {}: faces.nh = {}, edges.nh = {}, verts.nh = {}; faces.n_max = {}",
            i, csfaces.nh(), csedges.nh(), csverts.nh(), csfaces.n_max());
            Index startInd = 0;
            Index stopInd = csfaces.nh();
            for (Index j=startInd; j<stopInd; ++j) {
                if (!csfaces.has_kids_host(j)) {
                    csdiv::divide(j, csverts, csedges, csfaces);
                }
            }
        }
        logger.info("cubed sphere surface area = {}; |area - 4*pi| = {}", csfaces.surface_area_host(),
          abs(csfaces.surface_area_host() - 4*constants::PI));
        REQUIRE(FloatingPoint<Real>::equiv(csfaces.surface_area_host(), 4*constants::PI,
          1.5*constants::ZERO_TOL));

        VtkInterface<SphereGeometry, QuadFace> vtk;
        auto pd = vtk.toVtkPolyData(csfaces, csedges, csverts);
        logger.debug("vtk data conversion done.");
        vtk.writePolyData("cs_test.vtk", pd);
    }

  SECTION("tri, sphere") {
        const MeshSeed<IcosTriSphereSeed> seed;
        Index nmaxverts;
        Index nmaxfaces;
        Index nmaxedges;
        const int maxlev = 5;
        seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, maxlev);

        using coords_type = Coords<SphereGeometry>;

        auto icvertcrds = std::shared_ptr<coords_type>(new coords_type(nmaxverts));
        auto icvertlagcrds = std::shared_ptr<coords_type>(new coords_type(nmaxverts));
        icvertcrds->init_vert_crds_from_seed(seed);
        icvertlagcrds->init_vert_crds_from_seed(seed);
        Vertices<coords_type> icverts(nmaxverts, icvertcrds, icvertlagcrds);

        Edges icedges(nmaxedges);

        auto icfacecrds = std::shared_ptr<coords_type>(new coords_type(nmaxfaces));
        auto icfacelagcrds = std::shared_ptr<coords_type>(new coords_type(nmaxfaces));
        icfacecrds->init_interior_crds_from_seed(seed);
        icfacelagcrds->init_interior_crds_from_seed(seed);
        Faces<TriFace,SphereGeometry> icfaces(nmaxfaces, icfacecrds, icfacelagcrds);

        icfaces.init_from_seed(seed);
        icedges.init_from_seed(seed);
        typedef FaceDivider<SphereGeometry,TriFace> icdiv;
        for (int i=0; i<maxlev; ++i) {
            const Index stopInd = icfaces.nh();
            for (Index j=0; j<stopInd; ++j) {
                if (!icfaces.has_kids_host(j)) {
                    icdiv::divide(j, icverts, icedges, icfaces);
                }
            }
        }

        logger.info("icos. tri. sphere surface area = {}; |area - 4*pi| = {}", icfaces.surface_area_host(),
          abs(icfaces.surface_area_host() - 4*constants::PI));

        REQUIRE(FloatingPoint<Real>::equiv(icfaces.surface_area_host(), 4*constants::PI,
          31*constants::ZERO_TOL));

        VtkInterface<SphereGeometry, TriFace> vtk;
        logger.debug("writing vtk output.");
        vtkSmartPointer<vtkPolyData> pd = vtk.toVtkPolyData(icfaces, icedges, icverts);
        logger.debug("conversion to vtk polydata complete.");
        vtk.writePolyData("ic_test.vtk", pd);
    }
}

