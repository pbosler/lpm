#ifndef LPM_POLYMESH_2D_HPP
#define LPM_POLYMESH_2D_HPP

#include "LpmConfig.h"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "lpm_logger.hpp"
#include "Kokkos_Core.hpp"

#ifdef LPM_USE_COMPOSE
#include "compose/siqk_sqr.hpp"
#endif

#ifndef NDEBUG
#include "util/lpm_string_util.hpp"
#include <iostream>
#include <sstream>
#endif

#include <memory>

namespace Lpm {

template <typename SeedType>
struct PolyMeshParameters {
  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;
  Int init_depth;
  Real radius;
  Int amr_limit;
  MeshSeed<SeedType> seed;

  PolyMeshParameters(const Int depth, const Real r=1, const Int amr = 0) :
    init_depth(depth),
    amr_limit(amr),
    seed(r)
    {
      seed.set_max_allocations(nmaxverts, nmaxedges, nmaxfaces, depth + amr);
    }
};

/** @brief Class for organizing a topologically 2D mesh of particles and panels

  Provides a single access point for a collection of:
  1. Vertices, represented by physical and Lagrangian Coords
  2. Edges
  3. Faces, and a coincident set of physical and Lagrangian Coords to represent face centers.
*/
template <typename SeedType> class PolyMesh2d {
  public:
  typedef SeedType seed_type;
  typedef typename SeedType::geo Geo;
  typedef typename SeedType::faceKind FaceType;
  typedef Coords<Geo> coords_type;
  typedef std::shared_ptr<Coords<Geo>> coords_ptr;

  /** @brief Constructor.  Allocates memory for a PolyMesh2d instance.

    @param nmaxverts Maximum number of vertices that will be allowed in memory
    @param nmaxedges Maximum number of edges allowed in memory
    @param nmaxfaces Maximum number of faces allowed in memory

    @see MeshSeed::setMaxAllocations()
  */
  PolyMesh2d(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) :
    vertices(nmaxverts),
    edges(nmaxedges),
    faces(nmaxfaces) {
      vertices.phys_crds = coords_ptr(new coords_type(nmaxverts));
      vertices.lag_crds = coords_ptr(new coords_type(nmaxverts));
      faces.phys_crds = coords_ptr(new coords_type(nmaxfaces));
      faces.lag_crds = coords_ptr(new coords_type(nmaxfaces));
    }

  PolyMesh2d(const PolyMeshParameters<SeedType>& params) :
    vertices(params.nmaxverts),
    edges(params.nmaxedges),
    faces(params.nmaxfaces) {
      vertices.phys_crds = coords_ptr(new coords_type(params.nmaxverts));
      vertices.lag_crds = coords_ptr(new coords_type(params.nmaxverts));
      faces.phys_crds = coords_ptr(new coords_type(params.nmaxfaces));
      faces.lag_crds = coords_ptr(new coords_type(params.nmaxfaces));
      tree_init(params.init_depth, params.seed);
    }

    /// Destructor
    virtual ~PolyMesh2d() {}

    /** @brief initial refinement level.  All panels in the MeshSeed will be refined to this level.

      Adaptive refinement adds to this level.
    */
    Int base_tree_depth;

    /** @brief Return a subview of all initialized vertices' physical coordinates

      @device
    */
    auto get_vert_crds() const {
      return ko::subview(vertices.phys_crds->crds,
         std::make_pair(0, vertices.nh()), ko::ALL());}


    /** @brief Return a subview of all initialized face particles' physical coordinates

    @device
    */
    auto get_face_crds() const {
      return ko::subview(faces.phys_crds->crds,
        std::make_pair(0,faces.nh()), ko::ALL());}

    /** @brief Return a view face masks; leaves of the face tree are not masked. Internal nodes are masked.

      @device
    */
    mask_view_type faces_mask() const {
      return mask_view_type(faces.mask, std::make_pair(0,faces.nh()));}


    /** @brief Return a view of face masks on host.

     @todo This function seems useless without a deep copy

    @hostfn
    */
    typename mask_view_type::HostMirror faces_mask_host() const {
      return faces.leaf_mask_host();}


    /** @brief Return a subview of all initialized face areas

      @device
    */
    scalar_view_type faces_area() const {
      return scalar_view_type(faces.area, std::make_pair(0,faces.nh()));}


    /** @brief Number of initialized vertices

    @device
    */
    KOKKOS_INLINE_FUNCTION
    Index n_vertices() const {return vertices.n();}

    /** @brief Number of initialized faces

    @device
    */
    KOKKOS_INLINE_FUNCTION
    Index n_faces() const {return faces.n();}

    template <typename VT>
    void get_leaf_face_crds(VT leaf_crds) const;

    typename SeedType::geo::crd_view_type get_leaf_face_crds() const;


    /** @brief Number of initialized vertices

    @hostfn
    */
    Index n_vertices_host() const {return vertices.nh();}

    /** @brief Number of initialized edges

    @hostfn
    */
    Index n_edges_host() const {return edges.nh();}


    /** @brief Number of initialized faces

    @hostfn
    */
    Index n_faces_host() const {return faces.nh();}

    // mesh topology & coordinates
    Vertices<Coords<Geo>> vertices;
    Edges edges;
    Faces<FaceType, Geo> faces;

    KOKKOS_INLINE_FUNCTION
    bool edge_is_positive(const Index edge_idx, const Index face_idx) const {
      LPM_KERNEL_ASSERT(face_idx == edges.lefts(edge_idx) or
                        face_idx == edges.rights(edge_idx));
      return face_idx == edges.lefts(edge_idx);
    }

    template <typename EdgeList=Index*> KOKKOS_INLINE_FUNCTION
    void get_leaf_edges_from_parent(EdgeList& edge_list, Int& n_leaves,
      const Index parent_edge_idx) const {
      edge_list[0] = parent_edge_idx;
      n_leaves = 1;
      bool keep_going = edges.has_kids(parent_edge_idx);
      while (keep_going) {
        Int n_new = 0;
        keep_going = false;
        for (int i=0; i<n_leaves; ++i) {
          if (edges.has_kids(edge_list[i])) {
            // replace parent idx in list with its kids
            const auto kid0 = edges.kids(edge_list[i], 0);
            const auto kid1 = edges.kids(edge_list[i], 1);
            // kid0 will replace its parent, but
            // need to make room for kid1 at idx i + 1
            for (int j=i+1; j<n_leaves; ++j) {
              edge_list[j+1] = edge_list[j];
            }
            edge_list[i] = kid0;
            edge_list[i+1] = kid1;
            if (edges.has_kids(kid0) or edges.has_kids(kid1)) {
              keep_going = true;
            }
            ++n_new;
          }
        }
        n_leaves += n_new;
      }
    }

    template <typename EdgeList=Index*> KOKKOS_INLINE_FUNCTION
    void ccw_edges_around_face(EdgeList& face_leaf_edges, Int& n_leaf_edges,
      const Index face_idx) const {
      n_leaf_edges = 0;
      for (int i=0; i<SeedType::nfaceverts; ++i) {
        Index leaf_edge_list[2*LPM_MAX_AMR_LIMIT];
        Int n_leaves_this_edge = 0;
        get_leaf_edges_from_parent(leaf_edge_list, n_leaves_this_edge,
          faces.edges(face_idx, i));
#ifndef NDEBUG
        std::ostringstream ss;
        ss << "face " << face_idx << " edge " << i << " ("
           << faces.edges(face_idx,i) << ") : n_leaves_this_edge = "
           << n_leaves_this_edge << " "
           << sprarr("leaf_edge_list", leaf_edge_list, n_leaves_this_edge) << "\n";
        std::cout << ss.str();
#endif
        if (edge_is_positive(faces.edges(face_idx, i), face_idx)) {
          for (int j=0; j<n_leaves_this_edge; ++j) {
            face_leaf_edges[n_leaf_edges+j] = leaf_edge_list[j];
          }
        }
        else {
          const Int last_idx = n_leaf_edges + n_leaves_this_edge -1;
          for (int j=0; j<n_leaves_this_edge; ++j) {
            face_leaf_edges[last_idx - j] = leaf_edge_list[j];
          }
        }
        n_leaf_edges += n_leaves_this_edge;
      }
    }

    template <typename FaceList=Index*> KOKKOS_INLINE_FUNCTION
    void ccw_adjacent_faces(FaceList& adj_faces, Int& n_adj, const Index face_idx) const {
      Index leaf_edge_list[2*SeedType::nfaceverts*LPM_MAX_AMR_LIMIT];
      Int n_leaf_edges = 0;
      ccw_edges_around_face(leaf_edge_list, n_leaf_edges, face_idx);

      n_adj = n_leaf_edges;

      for (int i=0; i<n_leaf_edges; ++i) {
        if (edge_is_positive(leaf_edge_list[i], face_idx)) {
          adj_faces[i] = edges.rights(leaf_edge_list[i]);
        }
        else {
          adj_faces[i] = edges.lefts(leaf_edge_list[i]);
        }
      }
    }

    template <typename Point> KOKKOS_INLINE_FUNCTION
    Index locate_pt_walk_search(const Point& query_pt, const Index face_start_idx) const {
      Index result = LPM_NULL_IDX;
      Index current_idx = result;
      auto fcrd = Kokkos::subview(faces.phys_crds->crds, current_idx, Kokkos::ALL);
      Real dist = SeedType::geo::distance(query_pt, fcrd);

      bool keep_going = true;
      while (keep_going) {
        Index adj_face_list[8*LPM_MAX_AMR_LIMIT];
        Int n_adj;
        result = current_idx;
        ccw_adjacent_faces(adj_face_list, n_adj, current_idx);
        for (int i=0; i<n_adj; ++i) {
          fcrd = Kokkos::subview(faces.phys_crds->crds, adj_face_list[i], Kokkos::ALL);
          Real test_dist = SeedType::geo::distance(fcrd, query_pt);
          if (test_dist < dist) {
            dist = test_dist;
            current_idx = adj_face_list[i];
          }
        }
        keep_going = (current_idx != result);
      }
      return result;
    }

    template <typename Point> KOKKOS_INLINE_FUNCTION
    Index nearest_root_face(const Point& query_pt) const {
      Index result = 0;
      auto x0 = Kokkos::subview(faces.phys_crds->crds, 0, Kokkos::ALL);
      Real dist = SeedType::geo::distance(query_pt, x0);
      for (int i=1; i<SeedType::nfaces; ++i) {
        x0 = Kokkos::subview(faces.phys_crds->crds, i, Kokkos::ALL);
        const Real test_dist = SeedType::geo::distance(x0, query_pt);
        if (test_dist < dist) {
          dist = test_dist;
          result = i;
        }
      }
      return result;
    }

    template <typename Point> KOKKOS_INLINE_FUNCTION
    Index locate_pt_tree_search(const Point& query_pt, const Index root_face) const {
      bool keep_going = true;
      Index current_idx = root_face;
      Index next_idx = LPM_NULL_IDX;
      Real dist = std::numeric_limits<Real>::max();
      while (keep_going) {
        if (faces.has_kids(current_idx) > 0) {
          for (int k=0; k<4; ++k) {
            const auto fx = Kokkos::subview(faces.phys_crds->crds, faces.kids(current_idx, k), Kokkos::ALL);
            const Real test_dist = SeedType::geo::distance(query_pt, fx);
            if (test_dist < dist) {
              next_idx = faces.kids(current_idx, k);
              dist = test_dist;
            }
          }
          current_idx = next_idx;
        }
        else {
          keep_going = false;
        }
      }
      return current_idx;
    }

    template <typename Point> KOKKOS_INLINE_FUNCTION
    Index locate_face_containing_pt(const Point& query_pt) const {
      const auto tree_start = nearest_root_face(query_pt);
      const auto walk_start = locate_pt_tree_search(query_pt, tree_start);
      return locate_pt_walk_search(query_pt, walk_start);
    }

    Real surface_area_host() const {
      return faces.surface_area_host();
    }

    void reset_face_centroids();

    /** @brief Returns a pointer to the view of face coordinates

    @hostfn

    */
    typename Coords<Geo>::crd_view_type::HostMirror faces_phys_crds() {
      return faces.phys_crds->get_host_crd_view();}


    /** @brief Starting with a MeshSeed, uniformly refines each panel until the desired level of initial refinement is reached.

    @hostfn

    @param initDepth Max depth of initially refined mesh
    @param seed Mesh seed used to initialize particles and panels
    */
    void tree_init(const Int initDepth, const MeshSeed<SeedType>& seed);

    template <typename LoggerType>
    void divide_face(const Index face_idx, LoggerType& logger);

#ifdef LPM_USE_VTK
    /// @brief Construct relevant Vtk objects for visualization of a PolyMesh2d instance
    virtual void output_vtk(const std::string& fname) const;
#endif

    /// @brief Copies data from host to device
    virtual void update_device() const;

    /// @brief Copies data from device to host
    virtual void update_host() const;


    inline Real appx_mesh_size() const {return faces.appx_mesh_size();}

    /** @brief Writes basic info about a PolyMesh2d instance to a string.

    @hostfn
    */
    virtual std::string info_string(const std::string& label="",
      const int& tab_level = 0, const bool& dump_all=false) const;

  protected:
    typedef FaceDivider<Geo,FaceType> divider;

    void seed_init(const MeshSeed<SeedType>& seed);


};

/** @brief Resets faces' physical coordinates to the barycenter of the polygon defined
    by their vertices
*/
template <typename SeedType> struct FaceCentroidFunctor {
  typedef typename SeedType::geo::crd_view_type crd_view;
  crd_view face_crds;
  crd_view vert_crds;
  index_view_type face_crd_inds;
  index_view_type vert_crd_inds;
  ko::View<Index*[SeedType::nfaceverts]> face_verts;
  ko::View<Real[SeedType::nfaceverts][SeedType::geo::ndim]> local_vcrds;

  FaceCentroidFunctor(crd_view fc, const index_view_type fi, const crd_view vc,
    const index_view_type vi, const ko::View<Index*[SeedType::nfaceverts]>& fv) :
      face_crds(fc),
      face_crd_inds(fi),
      vert_crds(vc),
      vert_crd_inds(vi),
      face_verts(fv),
      local_vcrds("local_vcrds") {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    for (Int j=0; j<SeedType::nfaceverts; ++j) {
      for (Int k=0; k<SeedType::geo::ndim; ++k) {
        local_vcrds(j,k) = vert_crds(vert_crd_inds(face_verts(i,j)),k);
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
ko::View<Real*[3]> source_coords(const PolyMesh2d<SeedType>& pm) {
  const Index nv = pm.nvertsHost();
  const Index nl = pm.faces.nLeavesHost();
  ko::View<Real*[3]> result("source_coords", nv + nl);
  ko::parallel_for(nv, KOKKOS_LAMBDA (int i) {
    for (int j=0; j<3; ++j) {
      result(i,j) = pm.vertices.phys_crds->crds(i,j);
    }
  });
  ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
    Int offset = nv;
    for (int j=0; j<pm.nfaces(); ++j) {
      if (!pm.faces.mask(j)) {
        result(offset,0) = pm.faces.phys_crds->crds(j,0);
        result(offset,1) = pm.faces.phys_crds->crds(j,1);
        result(offset++,2) = pm.faces.phys_crds->crds(j,2);
      }
    }
  });
  return result;
}

}
#endif
