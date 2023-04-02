#ifndef LPM_POLYMESH_2D_HPP
#define LPM_POLYMESH_2D_HPP

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_logger.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_vertices.hpp"
#include "util/lpm_floating_point.hpp"

#ifdef LPM_USE_COMPOSE
#include "compose/siqk_sqr.hpp"
#endif

#ifndef NDEBUG
#include <iostream>
#include <sstream>

#include "util/lpm_string_util.hpp"
#endif

#include <memory>

namespace Lpm {

/** @brief Parameters that define a mesh for initialization.

  @param depth initial depth of mesh quadtree, uniform resolution
  @param r radius of initial mesh in physical space
  @param amr values > 0 allocate additional memory for adaptive refinement
*/
template <typename SeedType>
struct PolyMeshParameters {
  Index nmaxverts;  /// max number of vertices to allocate in memory
  Index nmaxedges;  /// max number of edges to allocate in memory
  Index nmaxfaces;  /// max number of faces to allocated in memory
  Int init_depth;   /// initial depth of mesh quadtree
  Real radius;      /// radius of initial mesh in physical space
  Int amr_limit;    /// if > 0, the allocated memory includes space for adaptive
                    /// refinement
  MeshSeed<SeedType> seed;  /// instance of the MeshSeed that initializes the
                            /// particles and panels

  PolyMeshParameters() = default;

  PolyMeshParameters(const PolyMeshParameters& other) = default;

  PolyMeshParameters(const Index nmv, const Index nme, const Index nmf)
      : nmaxverts(nmv),
        nmaxedges(nme),
        nmaxfaces(nmf),
        init_depth(0),
        radius(1),
        amr_limit(0) {}

  PolyMeshParameters(const Int depth, const Real r = 1, const Int amr = 0)
      : init_depth(depth), amr_limit(amr), seed(r) {
    seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, depth + amr);
  }
};

/** @brief Class for organizing a topologically 2D mesh of particles and panels

  Provides a single access point for a collection of:
  1. Vertices, represented by physical and Lagrangian Coords
  2. Edges
  3. Faces, and a coincident set of physical and Lagrangian Coords to represent
  face centers.
*/
template <typename SeedType>
class PolyMesh2d {
 public:
  /*

      Type defs

  */
  typedef SeedType seed_type;
  typedef typename SeedType::geo Geo;
  typedef typename SeedType::faceKind FaceType;
  typedef Coords<Geo> coords_type;
  typedef std::shared_ptr<Coords<Geo>> coords_ptr;
  typedef FaceDivider<Geo, FaceType> divider;

  /*

      Member variables

  */
  // mesh topology & coordinates
  Vertices<Coords<Geo>> vertices;  /// vertex particles, aka "passive," because
                                   /// they don't participate in quadrature
  Edges edges;  /// edges, typically only used for mesh initialization and
                /// refinement
  Faces<FaceType, Geo> faces;  /// face particles, aka "active," because they do
                               /// contribute to quadrature.

  /** @brief initial refinement level.  All panels in the MeshSeed will be
    refined to this level.

    Adaptive refinement adds to this level.
  */
  Int base_tree_depth;

  /// initial radius of mesh
  Real radius;

  PolyMeshParameters<SeedType> params_;
  /*

      Member functions

  */

  /** @brief Constructor.  Allocates memory for a PolyMesh2d instance.

    @param nmaxverts Maximum number of vertices that will be allowed in memory
    @param nmaxedges Maximum number of edges allowed in memory
    @param nmaxfaces Maximum number of faces allowed in memory

    @see MeshSeed::setMaxAllocations()
  */
  PolyMesh2d(const Index nmaxverts, const Index nmaxedges,
             const Index nmaxfaces)
      : vertices(nmaxverts),
        edges(nmaxedges),
        faces(nmaxfaces),
        radius(1),
        params_(nmaxverts, nmaxedges, nmaxfaces) {
    params_.nmaxverts = nmaxverts;
    params_.nmaxedges = nmaxedges;
    params_.nmaxfaces = nmaxfaces;
    params_.radius = radius;
  }

  /** @brief Constructor.  Allocates memory, does not initialize values.

    @param params PolyMeshParameters that define a mesh.
  */
  PolyMesh2d(const PolyMeshParameters<SeedType>& params)
      : vertices(params.nmaxverts),
        edges(params.nmaxedges),
        faces(params.nmaxfaces),
        radius(params.radius),
        params_(params) {
    tree_init(params.init_depth, params.seed);
  }

  /** @brief Constructor.  Allocates memory, does not initialize values.

    @param params PolyMeshParameters that define a mesh.
  */
  PolyMesh2d(const std::shared_ptr<PolyMeshParameters<SeedType>> params)
      : vertices(params->nmaxverts),
        edges(params->nmaxedges),
        faces(params->nmaxfaces),
        radius(params->radius),
        params_() {
    tree_init(params->init_depth, params->seed);
  }

  /// Destructor
  virtual ~PolyMesh2d() {}

  /** @brief Return a subview of all initialized vertices' physical coordinates

    @device
  */
  auto get_vert_crds() const {
    return ko::subview(vertices.phys_crds.view,
                       std::make_pair(0, vertices.nh()), ko::ALL());
  }

  /** @brief Return a subview of all initialized face particles' physical
  coordinates

  @device
  */
  auto get_face_crds() const {
    return ko::subview(faces.phys_crds.view, std::make_pair(0, faces.nh()),
                       ko::ALL());
  }

  /** @brief Return a view face masks; leaves of the face tree are not masked.
    Internal nodes are masked.

    @device
  */
  mask_view_type faces_mask() const {
    return mask_view_type(faces.mask, std::make_pair(0, faces.nh()));
  }

  /** @brief Return a view of face masks on host.

   @todo This function seems useless without a deep copy

  @hostfn
  */
  typename mask_view_type::HostMirror faces_mask_host() const {
    return faces.leaf_mask_host();
  }

  /** @brief Return a subview of all initialized face areas

    @device
  */
  scalar_view_type faces_area() const {
    return scalar_view_type(faces.area, std::make_pair(0, faces.nh()));
  }

  /** @brief Number of initialized vertices

  @device
  */
  KOKKOS_INLINE_FUNCTION
  Index n_vertices() const { return vertices.n(); }

  /** @brief Number of initialized faces

  @device
  */
  KOKKOS_INLINE_FUNCTION
  Index n_faces() const { return faces.n(); }

  template <typename VT>
  void get_leaf_face_crds(VT leaf_crds) const;

  typename SeedType::geo::crd_view_type get_leaf_face_crds() const;

  /** @brief Number of initialized vertices

  @hostfn
  */
  Index n_vertices_host() const { return vertices.nh(); }

  /** @brief Number of initialized edges

  @hostfn
  */
  Index n_edges_host() const { return edges.nh(); }

  /** @brief Number of initialized faces

  @hostfn
  */
  Index n_faces_host() const { return faces.nh(); }

  /** @brief Returns true if an edge has positive (CCW) orientation
    relative to a face.

    @warning This function does not verify connectivity; it will not detect
    whether or not edge(edge_idx) is associated with face(face_idx). It will
    return false in such a case.

    @param edge_idx
    @param face_idx
  */
  KOKKOS_INLINE_FUNCTION
  bool edge_is_positive(const Index edge_idx, const Index face_idx) const {
    return face_idx == edges.lefts(edge_idx);
  }

  /** @brief Returns a list of a panel's leaf edges.

    Required for AMR meshes.  If a panel's neighbor has been refined, its own
    edges will have been divided.

    @param [out] edge_list  List of leaves that are children of parent_edge_idx,
    with the same orientation as their parents.
    @param [out] n_leaves number of leaf edges, >= number of panel edges
    @param [in] parent_edge_idx index of edge that may have been divided
  */
  template <typename EdgeList = Index*>
  KOKKOS_INLINE_FUNCTION void get_leaf_edges_from_parent(
      EdgeList& edge_list, Int& n_leaves, const Index parent_edge_idx) const {
    LPM_KERNEL_ASSERT_MSG(parent_edge_idx != LPM_NULL_IDX,
                          "PolyMesh2d::get_leaf_edges_from_parent");
    edge_list[0] = parent_edge_idx;
    n_leaves = 1;
    bool keep_going = edges.has_kids(parent_edge_idx);
    while (keep_going) {
      Int n_new = 0;
      keep_going = false;
      for (int i = 0; i < n_leaves; ++i) {
        if (edges.has_kids(edge_list[i])) {
          // replace parent idx in list with its kids
          const auto kid0 = edges.kids(edge_list[i], 0);
          const auto kid1 = edges.kids(edge_list[i], 1);
          // kid0 will replace its parent, but
          // need to make room for kid1 at idx i + 1
          for (int j = i + 1; j < n_leaves; ++j) {
            edge_list[j + 1] = edge_list[j];
          }
          edge_list[i] = kid0;
          edge_list[i + 1] = kid1;
          if (edges.has_kids(kid0) or edges.has_kids(kid1)) {
            keep_going = true;
          }
          ++n_new;
        }
      }
      n_leaves += n_new;
    }
  }

  /** @brief Returns a counter-clockwise list of leaf edges around a face.

    @param [out] face_leaf_edges ccw list of edges
    @param [out] n_leaf_edges number of leaf edges
    @param [in] face_idx
  */
  template <typename EdgeList = Index*>
  KOKKOS_INLINE_FUNCTION void ccw_edges_around_face(
      EdgeList& face_leaf_edges, Int& n_leaf_edges,
      const Index face_idx) const {
    LPM_KERNEL_ASSERT_MSG(face_idx != LPM_NULL_IDX,
                          "PolyMesh2d::ccw_edges_around_face");
    n_leaf_edges = 0;
    for (int i = 0; i < SeedType::nfaceverts; ++i) {
      Index leaf_edge_list[2 * LPM_MAX_AMR_LIMIT];
      Int n_leaves_this_edge = 0;
      get_leaf_edges_from_parent(leaf_edge_list, n_leaves_this_edge,
                                 faces.edges(face_idx, i));
      if (edge_is_positive(faces.edges(face_idx, i), face_idx)) {
        for (int j = 0; j < n_leaves_this_edge; ++j) {
          face_leaf_edges[n_leaf_edges + j] = leaf_edge_list[j];
        }
      } else {
        const Int last_idx = n_leaf_edges + n_leaves_this_edge - 1;
        for (int j = 0; j < n_leaves_this_edge; ++j) {
          face_leaf_edges[last_idx - j] = leaf_edge_list[j];
        }
      }
      n_leaf_edges += n_leaves_this_edge;
    }
  }

  /** @brief Returns a counter-clockwise list of adjacent faces

    @param [out] adj_faces ccw list of face(face_idx)'s neighbors
    @param [out] n_adj number of neighbors
    @param [in] face_idx index of face whose neighbors are needed
  */
  template <typename FaceList = Index*>
  KOKKOS_INLINE_FUNCTION void ccw_adjacent_faces(FaceList& adj_faces,
                                                 Int& n_adj,
                                                 const Index face_idx) const {
    LPM_KERNEL_ASSERT_MSG(face_idx != LPM_NULL_IDX,
                          "PolyMesh2d::ccw_adjacent_faces");
    Index leaf_edge_list[2 * SeedType::nfaceverts * LPM_MAX_AMR_LIMIT];
    Int n_leaf_edges = 0;
    ccw_edges_around_face(leaf_edge_list, n_leaf_edges, face_idx);

    n_adj = 0;
    for (int i = 0; i < n_leaf_edges; ++i) {
      if (edge_is_positive(leaf_edge_list[i], face_idx)) {
        adj_faces[n_adj++] = edges.rights(leaf_edge_list[i]);
      } else {
        adj_faces[n_adj++] = edges.lefts(leaf_edge_list[i]);
      }
    }
  }

  /** @brief Returns the index of a face that contains the query_pt

    Worst-case scaling as @f$\sqrt(N)@f$, where N is the number of panels.

    @param [in] query_pt physical coordinates of point to locate within the mesh
    @param [in] face_start_idx initial guess for face containing point
    @return index of face containing query_pt
  */
  template <typename Point>
  KOKKOS_INLINE_FUNCTION Index locate_pt_walk_search(
      const Point& query_pt, const Index face_start_idx) const {
    LPM_KERNEL_ASSERT_MSG(face_start_idx != LPM_NULL_IDX,
                          "PolyMesh2d::locate_pt_walk_search");
    LPM_KERNEL_ASSERT_MSG(!faces.has_kids(face_start_idx),
                          "PolyMesh2d::locate_pt_walk_search is leaf-only.");
    Index result;
    Index current_idx = face_start_idx;
    auto fcrd = Kokkos::subview(faces.phys_crds.view, current_idx, Kokkos::ALL);
    Real dist = SeedType::geo::distance(query_pt, fcrd);
    // search all adjacent faces
    bool keep_going = true;
    while (keep_going) {
      Index adj_face_list[8 * LPM_MAX_AMR_LIMIT];
      Int n_adj;
      result = current_idx;
      ccw_adjacent_faces(adj_face_list, n_adj, current_idx);
      for (int i = 0; i < n_adj; ++i) {
        if (adj_face_list[i] != LPM_NULL_IDX) {  // skip external faces
          const auto poly =
              Kokkos::subview(faces.verts, adj_face_list[i], Kokkos::ALL);
          Real face_centroid[Geo::ndim];
          Geo::barycenter(face_centroid, vertices.phys_crds.view, poly,
                          SeedType::nfaceverts);
          Real test_dist = SeedType::geo::distance(face_centroid, query_pt);
          // continue search at adjacent face, if it's closer than current
          if (test_dist < dist) {
            dist = test_dist;
            current_idx = adj_face_list[i];
          }
        }
      }
      keep_going = (current_idx != result);
    }
    return result;
  }

  /** @brief Returns the index of the root face containing query_pt.
  Used to initialize a tree search.

    @param [in] query_pt physical coordinates of point to locate in a face
    @return index of root face containing query_pt
  */
  template <typename Point>
  KOKKOS_INLINE_FUNCTION Index nearest_root_face(const Point& query_pt) const {
    Index result = 0;
    auto x0 = Kokkos::subview(faces.phys_crds.view, 0, Kokkos::ALL);
    Real dist = SeedType::geo::distance(query_pt, x0);
    for (int i = 1; i < SeedType::nfaces; ++i) {
      x0 = Kokkos::subview(faces.phys_crds.view, i, Kokkos::ALL);
      const Real test_dist = SeedType::geo::distance(x0, query_pt);
      if (test_dist < dist) {
        dist = test_dist;
        result = i;
      }
    }
    return result;
  }

  /** @brief Returns the index of the leaf face closest to query_pt.

    @warning A face's kids may have moved outside of the convex region enclosed
    by its edges.

    @param [in] query_pt physical coordinates of point to locate
    @param [in] index of root face that's closest to query_pt
    @return index of closest leaf face to query pt
  */
  template <typename Point>
  KOKKOS_INLINE_FUNCTION Index
  locate_pt_tree_search(const Point& query_pt, const Index root_face) const {
    Index current_idx = root_face;
    Index next_idx;
    bool keep_going = faces.has_kids(current_idx);
    while (keep_going) {
      // initialize with kid 0
      auto fx = Kokkos::subview(faces.phys_crds.view,
                                faces.kids(current_idx, 0), Kokkos::ALL);
      Real dist = Geo::distance(fx, query_pt);
      next_idx = faces.kids(current_idx, 0);
      for (int k = 1; k < 4; ++k) {
        // replace kid 0 with any closer kids
        fx = Kokkos::subview(faces.phys_crds.view, faces.kids(current_idx, k),
                             Kokkos::ALL);
        const Real test_dist = Geo::distance(query_pt, fx);
        if (test_dist < dist) {
          next_idx = faces.kids(current_idx, k);
          dist = test_dist;
        }
      }
      current_idx = next_idx;
      // continue if not at a leaf
      keep_going = faces.has_kids(current_idx);
    }
    return current_idx;
  }

  /** @brief Returns true if a query_pt is located outside the boundaries of a
    mesh.

    @param [in] pt physical coordinates of point
    @param [in] face_idx index of facec closest to query_pt, output from
    locate_pt_walk_search or locate_face_containing_pt
    @return false if point is contained inside a mesh, true if not.
  */
  template <typename Point>
  KOKKOS_INLINE_FUNCTION bool pt_is_outside_mesh(const Point& pt,
                                                 const Index face_idx) const {
    bool result = false;
    if constexpr (std::is_same<Geo, PlaneGeometry>::value) {
      Index leaf_edge_list[2 * SeedType::nfaceverts * LPM_MAX_AMR_LIMIT];
      Int n_leaf_edges = 0;
      ccw_edges_around_face(leaf_edge_list, n_leaf_edges, face_idx);
      Int boundary_edges = 0;
      for (int i = 0; i < n_leaf_edges; ++i) {
        if (edges.on_boundary(leaf_edge_list[i])) {
          ++boundary_edges;
        }
      }
      if (boundary_edges > 0) {
        const auto poly = Kokkos::subview(faces.verts, face_idx, Kokkos::ALL);
        Real face_centroid[2];
        Geo::barycenter(face_centroid, vertices.phys_crds.view, poly,
                        SeedType::nfaceverts);
        const Real intr_dist = Geo::distance(pt, face_centroid);
        for (int i = 0; i < n_leaf_edges; ++i) {
          if (edges.on_boundary(leaf_edge_list[i])) {
            Real q_vec[2];
            edges.edge_vector(q_vec, vertices.phys_crds.view,
                              leaf_edge_list[i]);
            auto v0 =
                Kokkos::subview(vertices.phys_crds.view,
                                edges.origs(leaf_edge_list[i]), Kokkos::ALL);
            if (!edge_is_positive(leaf_edge_list[i], face_idx)) {
              Geo::negate(q_vec);
              v0 = Kokkos::subview(vertices.phys_crds.view,
                                   edges.dests(leaf_edge_list[i]), Kokkos::ALL);
            }
            Geo::normalize(q_vec);
            Real p_vec[2];
            for (int i = 0; i < 2; ++i) {
              p_vec[i] = face_centroid[i] - v0[i];
            }
            Real reflection[2];
            const Real dotp = Geo::dot(p_vec, q_vec);
            for (int i = 0; i < 2; ++i) {
              reflection[i] =
                  face_centroid[i] - 2 * (p_vec[i] - dotp * q_vec[i]);
            }
            const Real extr_dist = Geo::distance(pt, reflection);
            if (extr_dist < intr_dist) {
              result = true;
            }
          }
        }
      }
    }
    return result;
  }

  /** @brief Returns the index of the face whose barycenter is closest to
    query_pt

    @param [in] query_pt physical coordinates of point to locate
  */
  template <typename Point>
  KOKKOS_INLINE_FUNCTION Index
  locate_face_containing_pt(const Point& query_pt) const {
    const auto tree_start = nearest_root_face(query_pt);
    const auto walk_start = locate_pt_tree_search(query_pt, tree_start);
    if constexpr (std::is_same<Geo, PlaneGeometry>::value) {
      if (pt_is_outside_mesh(query_pt, walk_start)) {
        return LPM_NULL_IDX;
      }
    }
    return locate_pt_walk_search(query_pt, walk_start);
  }

  /** @brief Computes the barycentric coordinates of a point within a triangular
    panel.

    Ensures that a point is inside the mesh by verifying that all barycentric
    coordinates are nonnegative.

    @param [out] bc barycentric coordinates of point
    @param [in] pt physical coordinates of point
  */
  template <typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION void triangular_barycentric_coords(
      VT& bc, const CVT& pt) const {
    const auto f_idx = locate_face_containing_pt(pt);
    LPM_KERNEL_ASSERT(f_idx != LPM_NULL_IDX);
    const auto tri = Kokkos::subview(faces.verts, f_idx, Kokkos::ALL);
    const auto vertexA =
        Kokkos::subview(vertices.phys_crds.view, tri[0], Kokkos::ALL);
    const auto vertexB =
        Kokkos::subview(vertices.phys_crds.view, tri[1], Kokkos::ALL);
    const auto vertexC =
        Kokkos::subview(vertices.phys_crds.view, tri[2], Kokkos::ALL);
    const Real total_area = Geo::tri_area(vertexA, vertexB, vertexC);
    const Real area_a = Geo::tri_area(pt, vertexB, vertexC);
    const Real area_b = Geo::tri_area(pt, vertexC, vertexA);
    const Real area_c = Geo::tri_area(pt, vertexA, vertexB);
    LPM_KERNEL_ASSERT(
        FloatingPoint<Real>::equiv(total_area, faces.area[f_idx]));
    LPM_KERNEL_ASSERT(area_a >= 0);
    LPM_KERNEL_ASSERT(area_b >= 0);
    LPM_KERNEL_ASSERT(area_c >= 0);
    LPM_KERNEL_ASSERT(total_area >= 0);
    bc[0] = area_a / total_area;
    bc[1] = area_b / total_area;
    bc[2] = area_c / total_area;
  }

  /** @brief Finds the coordinates of a point within reference space.

    @param [out] ref reference coordinates
    @param [in] pt physical coordinates of a point
  */
  template <typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION void ref_elem_coords(VT& ref, const CVT& pt) const {
    return ref_elem_impl<FaceType, VT, CVT>(ref, pt);
  }

  /** @brief Implementation (private) of reference coordinates for triangular
   * faces.
   */
  template <typename Face, typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_same<Face, TriFace>::value, void>::type
      ref_elem_impl(VT& ref, const CVT& pt) const {
    triangular_barycentric_coords(ref, pt);
  }

  /** @brief Implementation (private) of reference coordinates for quadrilateral
   * faces.
   */
  template <typename Face, typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_same<Face, QuadFace>::value, void>::type
      ref_elem_impl(VT& ref, const CVT& pt) const {
    quad_ref<Geo, VT, CVT>(ref, pt);
  }

  /** @brief Private. Solves the quadratic equation. Required for planar
    quadrilateral reference coordinates.

    Solves a x^2 + bx + c == 0.

    @param [in] a coefficient a
    @param [in] b coefficient b
    @param [in] c coefficient c
  */
  Real KOKKOS_INLINE_FUNCTION quad_quadratic_solve(const Real a, const Real b,
                                                   const Real c) const {
    const Real disc = square(b) - 4 * a * c;
    LPM_KERNEL_ASSERT(disc >= 0);
    const Real r0 = (-b + sqrt(disc)) / (2 * a);
    const Real r1 = (-b - sqrt(disc)) / (2 * a);
    if (FloatingPoint<Real>::in_bounds(r0, -1, 1)) {
      LPM_KERNEL_ASSERT(!FloatingPoint<Real>::in_bounds(r1, -1, 1));
      return r0;
    } else {
      LPM_KERNEL_ASSERT(FloatingPoint<Real>::in_bounds(r1, -1, 1));
      return r1;
    }
  }

  /** @brief Private. Uses SIQK to find reference coordinates of a point on a
    sphere in a mesh of quadrilateral panels.
  */
  template <typename GeoType, typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_same<GeoType, SphereGeometry>::value,
                              void>::type
      quad_ref(VT& ref, const CVT& pt) const {
#ifndef LPM_USE_COMPOSE
    static_assert(false, "Compose TPL required.");
#else
    const auto f_idx = locate_face_containing_pt(pt);
    LPM_KERNEL_ASSERT(f_idx != LPM_NULL_IDX);
    auto quad = Kokkos::subview(faces.verts, f_idx, Kokkos::ALL);
    Index quad_cyc[4];
    for (Int i = 0; i < 4; ++i) {
      quad_cyc[i] = quad[(i + 1) % 4];
    }
    Real rpt[3];
    for (int i = 0; i < 3; ++i) {
      rpt[i] = pt[i];
    }
    siqk::sqr::calc_sphere_to_ref(vertices.phys_crds.view, quad_cyc, rpt,
                                  ref[0], ref[1]);
#endif
  }

  /** @brief Private. Computes the reference coordinates of a point in a
    planar quadrilateral mesh.

    Follows C. Hua, 1990, An inverse transformation for quadrilateral
    isoparametric elements: Analysis and application, Finite Elements in
    Analysis and Design 7:159--166.
  */
  template <typename GeoType, typename VT, typename CVT>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_same<GeoType, PlaneGeometry>::value,
                              void>::type
      quad_ref(VT& ref, const CVT& pt) const {
    Real a1, a2;
    Real b1, b2;
    Real c1, c2;
    Real d1, d2;
    const auto f_idx = locate_face_containing_pt(pt);
    LPM_KERNEL_ASSERT(f_idx != LPM_NULL_IDX);
    const auto quad = Kokkos::subview(faces.verts, f_idx, Kokkos::ALL);
    Real vx[4];
    Real vy[4];
    d1 = 4 * pt[0];
    d2 = 4 * pt[1];
    for (int i = 0; i < 4; ++i) {
      vx[i] = vertices.phys_crds.view(quad[i], 0);
      vy[i] = vertices.phys_crds.view(quad[i], 1);
      d1 -= vx[i];
      d2 -= vy[i];
    }
    a1 = -vx[0] + vx[1] - vx[2] + vx[3];
    a2 = -vy[0] + vy[1] - vy[2] + vy[3];
    b1 = -vx[0] - vx[1] + vx[2] + vx[3];
    b2 = -vy[0] - vy[1] + vy[2] + vy[3];
    c1 = vx[0] - vx[1] - vx[2] + vx[3];
    c2 = vy[0] - vy[1] - vy[2] + vy[3];
    if (FloatingPoint<Real>::zero(a1)) {
      // Case I
      if (FloatingPoint<Real>::zero(a2)) {
        // Case I.A
        ref[0] = (d1 * c2 - d2 * c1) / (b1 * c2 - b2 * c1);
        ref[1] = (b1 * d2 - b2 * d1) / (b1 * c2 - b2 * c1);
      } else {
        // Case I.B
        LPM_KERNEL_ASSERT(!FloatingPoint<Real>::zero(b1));
        if (FloatingPoint<Real>::zero(c1)) {
          // Case I.B.a
          ref[0] = d1 / b1;
          ref[1] = (b1 * d2 - b2 * d1) / (a2 * d1 + b1 * c2);
        } else {
          // Case I.B.b
          const Real qa = a2 * b1;
          const Real qb = c2 * b1 - a2 * d1 - b2 * c1;
          const Real qc = d2 * c1 - c2 * d1;
          ref[0] = quad_quadratic_solve(qa, qb, qc);
          ref[1] = (d1 - b1 * ref[0]) / c1;
        }
      }
    } else {
      // Case II
      if (!FloatingPoint<Real>::zero(a2)) {
        // Case II.A
        const Real ab = a1 * b2 - a2 * b1;
        const Real ac = a1 * c2 - a2 * c1;
        const Real ad = a1 * d2 - d2 * a1;
        const Real cb = c1 * b2 - c2 * b1;
        const Real dc = d1 * c2 - d2 * c1;
        const Real db = d1 * b2 - d2 * b1;
        if (!FloatingPoint<Real>::zero(ab)) {
          // Case II.A.a
          if (!FloatingPoint<Real>::zero(ac)) {
            // Case II.A.a.1
            const Real qa = ab;
            const Real qb = cb - ad;
            const Real qc = dc;
            ref[0] = quad_quadratic_solve(qa, qb, qc);
            ref[1] = (ad - ab * ref[0]) / ac;
          } else {
            // Case II.A.a.2
            ref[0] = ad / ab;
            ref[1] = a1 * db / (c1 * ab + a1 * ad);
          }
        } else {
          // Case II.A.b
          ref[0] = a1 * dc / (b1 * ac + a1 * ad);
          ref[1] = ad / ac;
        }
      } else {
        // Case II.B
        if (FloatingPoint<Real>::zero(b2)) {
          // Case II.B.a
          ref[0] = (d1 * c2 - c1 * d2) / (a1 * d2 + b1 * c2);
          ref[1] = d2 / c2;
        } else {
          // Case II.B.b
          const Real qa = a1 * b2;
          const Real qb = c1 * b2 - a1 * d2 - b1 * c2;
          const Real qc = d1 * c2 - c1 * d2;
          ref[0] = quad_quadratic_solve(qa, qb, qc);
          ref[1] = (d2 - b2 * ref[0]) / c2;
        }
      }
    }
  }

  /** @brief Interpolates a scalar field using "native" interpolation.

  "Native" implies the highest order interpolation degree for each face type,
  e.g., linear interpolation on triangles, bilinear interpolation on
  quadrilaterals.

    @param [out] dst destination view for interpolated values.
    @param [in] dst_crds physical coordinates of interpolation output points.
    @param [in] src scalar field values defined on this mesh.
  */
  void scalar_interpolate(scalar_view_type& dst,
                          const typename Coords<Geo>::crd_view_type dst_crds,
                          const ScalarField<VertexField>& src) const {
    Kokkos::parallel_for(
        dst.extent(0), KOKKOS_LAMBDA(const Index i) {
          Real ref[3];
          const auto mcrd = Kokkos::subview(dst_crds, i, Kokkos::ALL);
          const auto f_idx = locate_face_containing_pt(mcrd);
          if (f_idx != LPM_NULL_IDX) {
            ref_elem_coords(ref, mcrd);
            const auto poly = Kokkos::subview(faces.verts, f_idx, Kokkos::ALL);
            dst(i) = 0;
            if (FaceType::nverts == 3) {
              for (Int j = 0; j < 3; ++j) {
                dst(i) += ref[j] * src.view(poly[j]);
              }
            } else if (FaceType::nverts == 4) {
              dst(i) += 0.25 * (1 - ref[0]) * (1 + ref[1]) * src.view(poly[0]);
              dst(i) += 0.25 * (1 - ref[0]) * (1 - ref[1]) * src.view(poly[1]);
              dst(i) += 0.25 * (1 + ref[0]) * (1 - ref[1]) * src.view(poly[2]);
              dst(i) += 0.25 * (1 + ref[0]) * (1 + ref[1]) * src.view(poly[3]);
            }
          } else {
            dst(i) = 0;
          }
        });
  }

  /** @brief @return surface area

    @hostfn
  */
  Real surface_area_host() const { return faces.surface_area_host(); }

  void reset_face_centroids();

  /** @brief Returns a pointer to the view of face coordinates

  @hostfn

  */
  typename Coords<Geo>::crd_view_type::HostMirror faces_phys_crds() {
    return faces.phys_crds.get_host_crd_view();
  }

  /** @brief Starting with a MeshSeed, uniformly refines each panel until the
  desired level of initial refinement is reached.

  @hostfn

  @param initDepth Max depth of initially refined mesh
  @param seed Mesh seed used to initialize particles and panels
  */
  void tree_init(const Int initDepth, const MeshSeed<SeedType>& seed);

  /** @brief divides (refines) a face

  @param [in] face_idx index of face to divide
  @param [in/out] logger console output logger
  */
  template <typename LoggerType>
  void divide_face(const Index face_idx, LoggerType& logger);

#ifdef LPM_USE_VTK
  /// @brief Construct relevant Vtk objects for visualization of a PolyMesh2d
  /// instance
  virtual void output_vtk(const std::string& fname) const;
#endif

  /// @brief Copies data from host to device
  virtual void update_device() const;

  /// @brief Copies data from device to host
  virtual void update_host() const;

  /** @brief Returns the approximate mesh size; assumes quasi-uniform
   * discretization in space (not adaptively refined meshes).
   */
  inline Real appx_mesh_size() const { return faces.appx_mesh_size(); }

  /** @brief Writes basic info about a PolyMesh2d instance to a string.

  @hostfn
  */
  virtual std::string info_string(const std::string& label = "",
                                  const int& tab_level = 0,
                                  const bool& dump_all = false) const;

  /** @brief Divides faces that have been flagged for refinement.

    @param [in] flags flags(i) = true for faces that need to be divided
    @param [in/out] logger console output
  */
  template <typename LoggerType = Logger<>>
  void divide_flagged_faces(const Kokkos::View<bool*> flags,
                            LoggerType& logger) {
    if (!params_) {
      logger.warn(
          "divide_flagged_faces: mesh parameters not stored; AMR is disabled, "
          "exiting.");
      return;
    }

    Index flag_count;
    Kokkos::parallel_reduce(
        n_faces_host(),
        KOKKOS_LAMBDA(const Index i, Index& s) { s += (flags(i) ? 1 : 0); },
        flag_count);
    const Index space_left = params_.nmaxfaces - n_faces_host();

    if (flag_count > space_left / 4) {
      logger.warn(
          "divide_flagged_faces: not enough memory (flag count = {}, nfaces = "
          "{}, nmaxfaces = {})",
          flag_count, n_faces_host(), params_.nmaxfaces);
      return;
    }
    const Index n_faces_in = n_faces_host();
    auto host_flags = Kokkos::create_mirror_view(flags);
    Kokkos::deep_copy(host_flags, flags);
    Index refine_count = 0;
    bool limit_reached = false;
    for (Index i = 0; i < n_faces_in; ++i) {
      if (host_flags(i)) {
        if (faces.host_level(i) < params_.init_depth + params_.amr_limit) {
          divide_face(i, logger);
          ++refine_count;
        } else {
          limit_reached = true;
        }
      }
    }
    if (limit_reached) {
      logger.warn(
          "divide_flagged_faces: local refinement limit reached; divided {} of "
          "{} flagged faces.",
          refine_count, flag_count);
    } else {
      LPM_ASSERT(refine_count == flag_count);
      logger.info("divide_flagged_faces: {} faces divided.", refine_count);
    }
  }

 protected:
  /** @brief Initializes a mesh from a MeshSeed.

    @param [in] seed MeshSeed instance.
  */
  void seed_init(const MeshSeed<SeedType>& seed);
};

/** @brief Resets faces' physical coordinates to the barycenter of the polygon
   defined by their vertices
*/
template <typename SeedType>
struct FaceCentroidFunctor {
  typedef typename SeedType::geo::crd_view_type crd_view;
  crd_view face_crds;
  crd_view vert_crds;
  index_view_type face_crd_inds;
  index_view_type vert_crd_inds;
  ko::View<Index * [SeedType::nfaceverts]> face_verts;
  ko::View<Real[SeedType::nfaceverts][SeedType::geo::ndim]> local_vcrds;

  FaceCentroidFunctor(crd_view fc, const index_view_type fi, const crd_view vc,
                      const index_view_type vi,
                      const ko::View<Index * [SeedType::nfaceverts]>& fv)
      : face_crds(fc),
        face_crd_inds(fi),
        vert_crds(vc),
        vert_crd_inds(vi),
        face_verts(fv),
        local_vcrds("local_vcrds") {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& i) const {
    for (Int j = 0; j < SeedType::nfaceverts; ++j) {
      for (Int k = 0; k < SeedType::geo::ndim; ++k) {
        local_vcrds(j, k) = vert_crds(vert_crd_inds(face_verts(i, j)), k);
      }
    }
    auto fcrd = ko::subview(face_crds, face_crd_inds(i), ko::ALL);
    SeedType::geo::barycenter(fcrd, local_vcrds, SeedType::nfaceverts);
  }
};

/** @brief Collects physical coordinates from all particles (vertices and faces)
  for use as a source of interpolation data.

  @todo rewrite to handle faces in parallel

  @param pm PolyMesh2d mesh used as data source
*/
template <typename SeedType>
ko::View<Real* [SeedType::geo::ndim]> source_coords(
    const PolyMesh2d<SeedType>& pm) {
  const Index nv = pm.nvertsHost();
  const Index nl = pm.faces.nLeavesHost();
  ko::View<Real * [SeedType::geo::ndim]> result("source_coords", nv + nl);
  ko::parallel_for(
      nv, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < SeedType::geo::ndim; ++j) {
          result(i, j) = pm.vertices.phys_crds.view(i, j);
        }
      });
  ko::parallel_for(
      1, KOKKOS_LAMBDA(int i) {
        Int offset = nv;
        for (int j = 0; j < pm.nfaces(); ++j) {
          if (!pm.faces.mask(j)) {
            for (int k = 0; k < SeedType::geo::ndim; ++k) {
              result(offset, k) = pm.faces.phys_crds.view(j, k);
            }
            ++offset;
          }
        }
      });
  return result;
}

}  // namespace Lpm
#endif
