#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "tree/lpm_box3d.hpp"
#include "tree/lpm_cpu_tree.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_filename.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::tree;

TEST_CASE("cpu_tree_test", "[tree]") {
  Comm comm;

  Logger <> logger("cpu_tree_test", Log::level::debug, comm);

  typedef IcosTriSphereSeed seed_type;
  //typedef CubedSphereSeed seed_type;

  MeshSeed<seed_type> seed;

  const Int mesh_tree_depth = 5;

  /**
    Set memory allocations
  */
  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;
  seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, mesh_tree_depth);

  /**
    Build the particle/panel mesh
  */
  auto sphere = std::shared_ptr<PolyMesh2d<seed_type>>(new
    PolyMesh2d<seed_type>(nmaxverts, nmaxedges, nmaxfaces));
  sphere->tree_init(mesh_tree_depth, seed);
  sphere->update_device();

  /** Output mesh to a vtk file */
  VtkPolymeshInterface<seed_type> vtk(sphere);
  vtk.write("cpu_tree_test_sphere.vtp");

  SECTION("Max particles per leaf") {
    /**
      Build the CPU trees
    */
    const auto depth_ctl = CpuTree<>::TreeDepthControl::MaxCoordsPerLeaf;
    const int max_leaf_crds = 20;
    const bool shrink_boxes = false;
    CpuTree<> vertex_tree(sphere->vertices.phys_crds,
      depth_ctl, max_leaf_crds, shrink_boxes);

    logger.debug(vertex_tree.info_string());
    logger.debug("root box: {}", vertex_tree.root->box);

    const auto vert_tree_fname = "cpu_tree_verts_n" +
      std::to_string(max_leaf_crds) + ".vtu";

    vertex_tree.write_vtk(vert_tree_fname);

//     CpuTree<SphereGeometry> face_tree(sphere->faces.phys_crds,
//       depth_ctl, max_leaf_crds, shrink_boxes);
//
//     face_tree.write_vtk("cpu_tree_faces_n200.vtp");

  }
//   SECTION("Max tree depth") {
//     auto depth_ctl = CpuTree<Lpm::SphereGeometry>::TreeDepthControl::MaxDepth;
//     const int max_cpu_tree_depth = 3;
//     const bool shrink_boxes = false;
//     CpuTree<SphereGeometry> vertex_tree(sphere->vertices.phys_crds,
//       depth_ctl, max_cpu_tree_depth, shrink_boxes);
//
//     vertex_tree.write_vtk("cpu_tree_verts_d3.vtp");
//
//     CpuTree<SphereGeometry> face_tree(sphere->faces.phys_crds,
//       depth_ctl, max_cpu_tree_depth, shrink_boxes);
//
//     face_tree.write_vtk("cpu_tree_faces_d3.vtp");
//   }
}
