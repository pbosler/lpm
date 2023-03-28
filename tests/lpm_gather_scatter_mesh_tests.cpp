#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_field.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif
#include "catch.hpp"
#include <memory>
#include <sstream>

using namespace Lpm;

template <typename SeedType>
struct GatherScatterTest {
  using Divider = FaceDivider<typename SeedType::geo, typename SeedType::faceKind>;

  void run() {
    Comm comm;

    Logger<> logger("gather scatter test - " + SeedType::id_string(),
      Log::level::debug, comm);

    constexpr Int init_depth = 0;
    constexpr Real r = 1;
    constexpr Int amr_limit = 1;

    PolyMeshParameters<SeedType> pm_params(init_depth, r, amr_limit);
    const auto pm = std::make_shared<PolyMesh2d<SeedType>>(pm_params);
    Divider::divide(0, pm->vertices, pm->edges, pm->faces);
    Divider::divide(pm->faces.kid_host(0,0), pm->vertices, pm->edges, pm->faces);
    pm->update_device();

    ScalarField<VertexField> vert_scalar("scalar", pm->n_vertices_host());
    ScalarField<FaceField> face_scalar("scalar", pm->n_faces_host());
    Kokkos::deep_copy(vert_scalar.view, 2.0);
    Kokkos::deep_copy(face_scalar.view, 2.0);
    std::map<std::string, ScalarField<VertexField>> vert_scalar_fields;
    std::map<std::string, ScalarField<FaceField>> face_scalar_fields;
    vert_scalar_fields.emplace("scalar", vert_scalar);
    face_scalar_fields.emplace("scalar", face_scalar);

    Int tab_level = 0;
    bool verbose = false;
    logger.info(pm->info_string("2 divides", tab_level, verbose));

    GatherMeshData<SeedType> gather(pm);
    gather.unpack_coordinates();
    gather.init_scalar_fields(vert_scalar_fields, face_scalar_fields);
    gather.gather_scalar_fields(vert_scalar_fields, face_scalar_fields);
    auto new_scalar = gather.scalar_fields.at("scalar");
    Kokkos::deep_copy(new_scalar, 3.0);

    gather.update_host();

    REQUIRE(gather.n() == pm->n_vertices_host() + pm->faces.n_leaves_host());

    logger.debug(gather.info_string(tab_level, verbose));

    Index n_duplicates = 0;
    Kokkos::parallel_reduce(gather.n(),
      KOKKOS_LAMBDA (const Index i, Index& n) {
        for (Int j=0; j<gather.n(); ++j) {
          if (j != i) {
            Real xi[3];
            Real xj[3];
            xi[0] = gather.x(i);
            xi[1] = gather.y(i);
            xj[0] = gather.x(j);
            xj[1] = gather.y(j);
            if constexpr (SeedType::geo::ndim == 3) {
              xi[2] = gather.z(i);
              xj[2] = gather.z(j);
            }
            const Real dist = SeedType::geo::distance(xi, xj);
            if (FloatingPoint<Real>::zero(dist)) {
              ++n;
            }
          }
        }
      }, n_duplicates);
    REQUIRE(n_duplicates == 0);

    ScatterMeshData<SeedType> scatter(gather, pm);
    scatter.scatter_fields(vert_scalar_fields, face_scalar_fields);

    Index n_vert_threes = 0;
    Kokkos::parallel_reduce(pm->n_vertices_host(),
      KOKKOS_LAMBDA (const Index i, Index& n) {
        if ( FloatingPoint<Real>::equiv(vert_scalar(i), 3) ) {
          ++n;
        }
      }, n_vert_threes);
    REQUIRE(n_vert_threes == pm->n_vertices_host());

    Index n_face_threes = 0;
    const auto face_mask = pm->faces.mask;
    Kokkos::parallel_reduce(pm->n_faces_host(),
      KOKKOS_LAMBDA (const Index i, Index& n) {
        if ( FloatingPoint<Real>::equiv(face_scalar(i), 3) ) {
          ++n;
        }
      }, n_face_threes);
    Index n_face_twos = 0;
    Kokkos::parallel_reduce(pm->n_faces_host(),
      KOKKOS_LAMBDA (const Index i, Index& n) {
        if (FloatingPoint<Real>::equiv(face_scalar(i), 2)) {
          ++n;
        }
      }, n_face_twos);
     REQUIRE(n_face_threes == pm->faces.n_leaves_host());
     REQUIRE(n_face_twos == pm->faces.nh() - pm->faces.n_leaves_host());


#ifdef LPM_USE_VTK
  logger.debug("starting vtk output.");
  VtkPolymeshInterface<SeedType> vtk(pm);
  vtk.add_scalar_point_data(vert_scalar.view);
  vtk.add_scalar_cell_data(face_scalar.view);
  vtk.write("gather_scatter_divide2.vtp");
  logger.debug("vtk output complete.");
#endif
}
};

TEST_CASE("gather_scatter", "") {
  SECTION("tri_hex") {
    typedef TriHexSeed seed_type;
    GatherScatterTest<seed_type> gs_test;
    gs_test.run();
  }
  SECTION("quad_rect") {
    typedef QuadRectSeed seed_type;
    GatherScatterTest<seed_type> gs_test;
    gs_test.run();
  }
  SECTION("icos_tri") {
    typedef IcosTriSphereSeed seed_type;
    GatherScatterTest<seed_type> gs_test;
    gs_test.run();
  }
  SECTION("cubed_sph") {
    typedef CubedSphereSeed seed_type;
    GatherScatterTest<seed_type> gs_test;
    gs_test.run();
  }
}
