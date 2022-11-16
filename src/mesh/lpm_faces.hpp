#ifndef LPM_FACES_HPP
#define LPM_FACES_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_mesh_seed.hpp"

#include "Kokkos_Core.hpp"

namespace Lpm {

#ifdef LPM_USE_NETCDF
template <typename Geo> class NcWriter; // fwd decl
class PolymeshReader;
#endif


/** @brief Faces define panels.  Connected to Coords and Edges.

Most faces have only vertices (define by two sets of Coords, one physical, one Lagrangian)
and single particles initialized at their barycrd_inds (also 2 sets of Coords, but separate from the vertcies).
High order faces have multiple particles in their interior.

Vertices (Coords) and Edges are listed in counter-clockwise order.

All Faces (even the high order ones) are currently assumed to be convex.

All initialization / changes occur on host.  Device arrays are const.

*/
template <typename FaceKind, typename Geo> class Faces {
  public:
    typedef FaceKind faceKind;
    typedef Geo geo;
    typedef ko::View<Index*[FaceKind::nverts]> vertex_view_type;
    typedef vertex_view_type edge_view_type;
    typedef ko::View<Index*[4]> face_tree_view;
    static constexpr Int nverts = FaceKind::nverts;
    typedef typename face_tree_view::HostMirror face_tree_host;
    typedef typename index_view_type::HostMirror host_index_view;
    typedef typename vertex_view_type::HostMirror host_vertex_view;
    typedef host_vertex_view host_edge_view;
    typedef typename scalar_view_type::HostMirror host_scalar;

    template <typename Geom, typename FaceType> friend struct FaceDivider;
    template <typename MeshSeedType> friend class PolyMesh2d;

#ifdef LPM_USE_NETCDF
    friend class NcWriter<Geo>;
    friend class PolymeshReader;
#endif

    mask_view_type mask; ///< non-leaf faces are masked
    vertex_view_type verts;  ///< indices to Coords on face edges, ccw order per face
    edge_view_type edges; ///< indices to Edges, ccw order per face
    index_view_type crd_inds; ///< indices to Coords<Geo> inside faces
    ko::View<Int*,Dev> level; ///< level of faces in tree
    index_view_type parent; ///< indices to Faces<FaceKind>
    face_tree_view kids; ///< indices to Faces<FaceKind>
    n_view_type n; ///< number of Faces currently defined
    n_view_type n_leaves; ///< number of leaf Faces
    scalar_view_type area; ///< Areas of each face
    index_view_type leaf_idx; ///< index of leaf in leaf-only face array

    /** @brief Constructor.

      @param nmax Maximum number of faces to allocate space.
      @see MeshSeed::setMaxAllocations()
    */
    explicit Faces(const Index nmax) :
      verts("faceverts", nmax),
      edges("faceedges", nmax),
      crd_inds("crd_inds",nmax),
      parent("parent", nmax),
      kids("kids", nmax),
      n("n"),
      _nmax(nmax),
      area("area", nmax),
      n_leaves("n_leaves"),
      mask("mask",nmax),
      level("level",nmax),
      leaf_idx("leaf_idx", nmax) {
      _hostverts = ko::create_mirror_view(verts);
      _hostedges = ko::create_mirror_view(edges);
      _host_crd_inds = ko::create_mirror_view(crd_inds);
      _hostparent = ko::create_mirror_view(parent);
      _hostkids = ko::create_mirror_view(kids);
      _nh = ko::create_mirror_view(n);
      _hostarea = ko::create_mirror_view(area);
      _hn_leaves = ko::create_mirror_view(n_leaves);
      _nh() = 0;
      _hn_leaves() = 0;
      _hmask = ko::create_mirror_view(mask);
      _hlevel = ko::create_mirror_view(level);
    }

    Faces(const Index nmax, std::shared_ptr<Coords<Geo>> pcrds,
                            std::shared_ptr<Coords<Geo>> lcrds) :
      verts("faceverts", nmax),
      edges("faceedges", nmax),
      crd_inds("crd_inds",nmax),
      parent("parent", nmax),
      kids("kids", nmax),
      n("n"),
      _nmax(nmax),
      area("area", nmax),
      n_leaves("n_leaves"),
      mask("mask",nmax),
      level("level",nmax),
      phys_crds(pcrds),
      lag_crds(lcrds) {
        _hostverts = ko::create_mirror_view(verts);
        _hostedges = ko::create_mirror_view(edges);
        _host_crd_inds = ko::create_mirror_view(crd_inds);
        _hostparent = ko::create_mirror_view(parent);
        _hostkids = ko::create_mirror_view(kids);
        _nh = ko::create_mirror_view(n);
        _hostarea = ko::create_mirror_view(area);
        _hn_leaves = ko::create_mirror_view(n_leaves);
        _nh() = 0;
        _hn_leaves() = 0;
        _hmask = ko::create_mirror_view(mask);
        _hlevel = ko::create_mirror_view(level);
      }

#ifdef LPM_USE_NETCDF
  explicit Faces(const PolymeshReader& reader);
#endif

    inline host_vertex_view verts_host() const {return _hostverts;}
    inline host_edge_view edges_host() const {return _hostedges;}
    inline face_tree_host kids_host() const {return _hostkids;}
    inline host_index_view parents_host() const {return _hostparent;}
    inline host_index_view crd_inds_host() const {return _host_crd_inds;}
    inline typename ko::View<Int*,Dev>::HostMirror levels_host() const {
       return _hlevel;}
    inline Int host_level(const Index idx) const {return _hlevel(idx);}

    /** @brief Copies data from host to device
    */
    void update_device() const {
      ko::deep_copy(verts, _hostverts);
      ko::deep_copy(edges, _hostedges);
      ko::deep_copy(crd_inds, _host_crd_inds);
      ko::deep_copy(parent, _hostparent);
      ko::deep_copy(kids, _hostkids);
      ko::deep_copy(area, _hostarea);
      ko::deep_copy(n, _nh);
      ko::deep_copy(n_leaves, _hn_leaves);
      ko::deep_copy(mask, _hmask);
      ko::deep_copy(level, _hlevel);
      phys_crds->update_device();
      lag_crds->update_device();
      scan_leaves();
    }

    /** @brief Copies data from device to Host
    */
    void update_host() const {
      ko::deep_copy(_hostarea, area);
      phys_crds->update_host();
      lag_crds->update_host();
    }


    /** @brief Returns true if a face has been divided.

      @param ind Face to query
      @return true if face(ind) has children
    */
    KOKKOS_INLINE_FUNCTION
    bool has_kids(const Index ind) const {
      LPM_KERNEL_ASSERT( ind < n() );
      return kids(ind,0) > 0;
    }

    /** @brief returns a view to all face areas.

    @hostfn If the device has updated face areas, this function will not see it unless update_host() is called first.
    */
    typename scalar_view_type::HostMirror area_host() const {return _hostarea;}

    typename mask_view_type::HostMirror leaf_mask_host() const {return _hmask;}

    /** @brief Returns the maximum number of faces allowed in memory.
    */
    inline Index n_max() const {return verts.extent(0);}

    /** @brief Updates the mask value of a face

    \hostfn

    @param i index of face to update
    @param val new value for mask(i)
    */
    inline void set_leaf_mask(const Index i, const bool val) {_hmask(i) = val;}

    /** @brief Return the number of currently initialized faces.

      @hostfn
    */
    inline Index nh() const {return _nh();}

    /** @brief Inserts a new face into a Faces instance.

    @hostfn

    @param ctr_ind pointer to center particle in a Coords object
    @param vertinds pointers to vertex indices in a Coords object
    @param edgeinds pointers to edge indices in an Edges object
    @param prt pointer to parent face
    @param ar area value of new face
    */
    void insert_host(const Index ctr_ind, ko::View<Index*,Host> vertinds,
      ko::View<Index*,Host> edgeinds, const Index prt=constants::NULL_IND, const Real ar = 0.0);


    /** Populate a view of coordinates that only includes leaf faces.

      View must have been allocated already.
    */
    void leaf_crd_view(const typename Geo::crd_view_type leaf_crds) const;

    /** Allocate and populate a view of coordinates that only includes
      leaf faces.
    */
    typename Geo::crd_view_type leaf_crd_view() const;


    void leaf_field_vals(const scalar_view_type vals, const ScalarField<FaceField>& field) const;

    scalar_view_type leaf_field_vals(const ScalarField<FaceField>& field) const;

    void leaf_field_vals(const typename Geo::vec_view_type vals, const VectorField<Geo,FaceField>& field) const;

    typename Geo::vec_view_type leaf_field_vals(const VectorField<Geo,FaceField>& field) const;


    /** @brief Returns true if a face has been divided

      @hostfn

      @param ind index of face to query
    */
    inline bool has_kids_host(const Index ind) const {
      LPM_ASSERT(ind < _nh());
      return _hostkids(ind, 0) > 0;
    }

    /** @brief Overwrite the children of a face

    @hostfn

    @param parent index of face whose children need to be updated.
    @param kids data values for new kids
    */
    template <typename CV>
    void set_kids_host(const Index ind, const CV v) {
      LPM_ASSERT(ind < _nh());
      for (int i=0; i<4; ++i) {
        _hostkids(ind, i) = v(i);
      }
    }

    /** @brief Get a particular vertex from a host

      @hostfn

      @param ind index of face
      @param relInd relative index of vertex (relative to face(ind) in Faces object); see MeshSeed
    */
    Index vert_host(const Index ind, const Int relInd) const {return _hostverts(ind, relInd);}

    Index parent_host(const Index ind) const {return _hostparent(ind);}

    Index level_host(const Index idx) const {return _hlevel(idx);}

    Index kid_host(const Index idx, const Int kid) const {return _hostkids(idx, kid);}

    template <typename CV>
    void set_verts_host(const Index& ind, const CV v) {
      LPM_ASSERT(ind < _nh());
      for (Short i=0; i<nverts; ++i) {
        _hostverts(ind,i) = v[i];
      }
    }

    template <typename CV>
    void set_edges_host(const Index& ind, const CV v) {
      LPM_ASSERT(ind < _nh());
      for (Short i=0; i<nverts; ++i) {
        _hostedges(ind,i) = v[i];
      }
    }

     /** @brief Get a particular edge from a host

      @hostfn

      @param ind index of face
      @param relInd relative index of edge (relative to face(ind) in Faces object); see MeshSeed
    */
    Index edge_host(const Index ind, const Int relInd) const {return _hostedges(ind, relInd);}

    /** @brief Get the index of the center particle (from Coords) associated with a face

    @hostfn

    @param ind face index
    @return index to Coords for face(ind)'s center particle
    */
    inline Index crd_idx_host(const Index ind) const {return _host_crd_inds(ind);}

    /** @brief Returns true if an edge is positive oriented about a face


      @hostfn

      @param faceInd face whose edge needs querying
      @param relEdgeInd relative edge index on face (see MeshSeed)
      @param edges collection of Edges
    */
    inline bool edge_is_positive(const Index faceInd, const Int relEdgeInd, const Edges& edges) const {
      return faceInd == edges.left_host(_hostedges(faceInd, relEdgeInd));
    }


    /** Overwrites a face's area

    @hostfn

    @param ind index of face whose area needs updating
    @param ar new value for area
    */
    inline void set_area_host(const Index ind, const Real ar) {_hostarea(ind)= ar;}

    inline Real area_host(const Index& ind) const {return _hostarea(ind);}

    /** @brief Decreases the number of leaves by one.

    @hostfn
    */
    inline void decrement_leaves() {_hn_leaves() -= 1;}

    /** @brief Increases the face leaf count

    @hostfn
    */
    inline void increment_leaves(const Short& i=1) {_hn_leaves() += i;}

    /** @brief Writes basic info about a Faces object's state to a string.

    @hostfn

    @param label name for Faces object
    */
    std::string info_string(const std::string& label="", const int& tab_level=0, const bool& dump_all=false) const;

    /** @brief Initialize a collection of Faces from a MeshSeed

    @hostfn

    @param seed
    */
    template <typename SeedType>
    void init_from_seed(const MeshSeed<SeedType>& seed);

    /** @brief Returns the number of leaves in the Faces tree.

    @hostfn
    */
    Index n_leaves_host() const {return _hn_leaves();}

    /** @brief Returns the surface area, as seen from the host

    @hostfn If the device updates face areas, this function will  not see it unless update_host() is called first.
    */
    Real surface_area_host() const;

    inline Real appx_mesh_size() const {return std::sqrt(surface_area_host() / _hn_leaves());}

    std::shared_ptr<Coords<Geo>> phys_crds;

    std::shared_ptr<Coords<Geo>> lag_crds;

  protected:

    host_vertex_view _hostverts;
    host_edge_view _hostedges;
    host_index_view _host_crd_inds;
    host_index_view _hostparent;
    face_tree_host _hostkids;
    ko::View<Index>::HostMirror _nh;
    ko::View<Index>::HostMirror _hn_leaves;
    typename mask_view_type::HostMirror _hmask;
    typename ko::View<Int*,Dev>::HostMirror _hlevel;
    host_scalar _hostarea;

    Index _nmax;

    void scan_leaves() const;
};

template <typename Geo, typename FaceType> struct FaceDivider {
  static void divide(const Index faceInd, Vertices<Coords<Geo>>& verts,
    Edges& edges, Faces<FaceType, Geo>& faces) {}
};

template <typename Geo> struct FaceDivider<Geo, TriFace> {
  static void divide(const Index faceInd, Vertices<Coords<Geo>>& verts,
    Edges& edges, Faces<TriFace,Geo>& faces) ;
};

template <typename Geo> struct FaceDivider<Geo, QuadFace> {
  static void divide(const Index faceInd, Vertices<Coords<Geo>>& verts,
    Edges& edges, Faces<QuadFace, Geo>& faces);
};

}
#endif
