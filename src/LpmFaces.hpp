#ifndef LPM_FACES_HPP
#define LPM_FACES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmMeshSeed.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

#ifdef LPM_HAVE_NETCDF
  class PolyMeshReader; /// fwd. decl.
#endif

/** @brief Faces define panels.  Connected to Coords and Edges.

Most faces have only vertices (define by two sets of Coords, one physical, one Lagrangian)
and single particles initialized at their barycenters (also 2 sets of Coords, but separate from the vertcies).
High order faces have multiple particles in their interior.

Vertices (Coords) and Edges are listed in counter-clockwise order.

All Faces (even the high order ones) are currently assumed to be convex.

All initialization / changes occur on host.  Device arrays are const.

*/
template <typename FaceKind> class Faces {
  public:
    typedef ko::View<Index*[FaceKind::nverts]> vertex_view_type;
    typedef vertex_view_type edge_view_type;
    typedef ko::View<Index*[4]> face_tree_view;
    template <typename Geo, typename FaceType> friend struct FaceDivider;
    static constexpr Int nverts = FaceKind::nverts;
    typedef typename face_tree_view::HostMirror face_tree_host;
    typedef typename index_view_type::HostMirror host_index_view;
    typedef typename vertex_view_type::HostMirror host_vertex_view;
    typedef host_vertex_view host_edge_view;
    typedef typename scalar_view_type::HostMirror host_scalar;

    mask_view_type mask; ///< non-leaf faces are masked
    vertex_view_type verts;  ///< indices to Coords on face edges, ccw order per face
    edge_view_type edges; ///< indices to Edges, ccw order per face
    index_view_type centers; ///< indices to Coords<Geo> inside faces
    ko::View<Int*,Dev> level; ///< level of faces in tree
    index_view_type parent; ///< indices to Faces<FaceKind>
    face_tree_view kids; ///< indices to Faces<FaceKind>
    n_view_type n; ///< number of Faces currently defined
    n_view_type nLeaves; ///< number of leaf Faces
    scalar_view_type area; ///< Areas of each face

    /** @brief Constructor.

      @param nmax Maximum number of faces to allocate space.
      @see MeshSeed::setMaxAllocations()
    */
    Faces(const Index nmax) : verts("faceverts", nmax), edges("faceedges", nmax), centers("centers",nmax),
      parent("parent", nmax), kids("kids", nmax), n("n"), _nmax(nmax), area("area", nmax), nLeaves("nLeaves"),
      mask("mask",nmax), level("level",nmax) {
      _hostverts = ko::create_mirror_view(verts);
      _hostedges = ko::create_mirror_view(edges);
      _hostcenters = ko::create_mirror_view(centers);
      _hostparent = ko::create_mirror_view(parent);
      _hostkids = ko::create_mirror_view(kids);
      _nh = ko::create_mirror_view(n);
      _hostarea = ko::create_mirror_view(area);
      _hnLeaves = ko::create_mirror_view(nLeaves);
      _nh() = 0;
      _hnLeaves() = 0;
      _hmask = ko::create_mirror_view(mask);
      _hlevel = ko::create_mirror_view(level);
    }

#ifdef LPM_HAVE_NETCDF
  Faces(const PolyMeshReader& reader);
#endif

    inline host_vertex_view getVertsHost() const {return _hostverts;}
    inline host_edge_view getEdgesHost() const {return _hostedges;}
    inline face_tree_host getKidsHost() const {return _hostkids;}
    inline host_index_view getParentsHost() const {return _hostparent;}
    inline host_index_view getCentersHost() const {return _hostcenters;}
    inline typename ko::View<Int*,Dev>::HostMirror getLevelsHost() const {
       return _hlevel;}

    /** @brief Copies data from host to device
    */
    void updateDevice() const {
      ko::deep_copy(verts, _hostverts);
      ko::deep_copy(edges, _hostedges);
      ko::deep_copy(centers, _hostcenters);
      ko::deep_copy(parent, _hostparent);
      ko::deep_copy(kids, _hostkids);
      ko::deep_copy(area, _hostarea);
      ko::deep_copy(n, _nh);
      ko::deep_copy(nLeaves, _hnLeaves);
      ko::deep_copy(mask, _hmask);
      ko::deep_copy(level, _hlevel);
    }

    /** @brief Copies data from device to Host
    */
    void updateHost() const {
      ko::deep_copy(_hostarea, area);
    }


    /** @brief Returns true if a face has been divided.

      @param ind Face to query
      @return true if face(ind) has children
    */
    KOKKOS_INLINE_FUNCTION
    bool hasKids(const Index ind) const {return ind < n() && kids(ind,0) > 0;}

    /** @brief returns a view to all face areas.

    @hostfn If the device has updated face areas, this function will not see it unless updateHost() is called first.
    */
    typename scalar_view_type::HostMirror getAreaHost() const {return _hostarea;}

    typename mask_view_type::HostMirror getMaskHost() const {return _hmask;}

    /** @brief Returns the maximum number of faces allowed in memory.
    */
    inline Index nMax() const {return verts.extent(0);}

    /** @brief Updates the mask value of a face

    \hostfn

    @param i index of face to update
    @param val new value for mask(i)
    */
    inline void setMask(const Index i, const bool val) {_hmask(i) = val;}

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
    void insertHost(const Index ctr_ind, ko::View<Index*,Host> vertinds,
      ko::View<Index*,Host> edgeinds, const Index prt=NULL_IND, const Real ar = 0.0);

    /** @brief Overwrite the children of a face

    @hostfn

    @deprecated in favor of setKidsHost() since Faces cannot be updated on device

    @param parent index of face whose children need to be updated.
    @param kids data values for new kids
    */
    void setKids(const Index parent, const Index* kids);

    /** @brief Returns true if a face has been divided

      @hostfn

      @param ind index of face to query
    */
    inline bool hasKidsHost(const Index ind) const {return ind < _nh() && _hostkids(ind, 0) > 0;}

    /** @brief Overwrite the children of a face

    @hostfn

    @param parent index of face whose children need to be updated.
    @param kids data values for new kids
    */
    template <typename CV>
    void setKidsHost(const Index ind, const CV v) {
      for (int i=0; i<4; ++i) {
        _hostkids(ind, i) = v(i);
      }
    }

    /** @brief Get a particular vertex from a host

      @hostfn

      @param ind index of face
      @param relInd relative index of vertex (relative to face(ind) in Faces object); see MeshSeed
    */
    Index getVertHost(const Index ind, const Int relInd) const {return _hostverts(ind, relInd);}

     /** @brief Get a particular edge from a host

      @hostfn

      @param ind index of face
      @param relInd relative index of edge (relative to face(ind) in Faces object); see MeshSeed
    */
    Index getEdgeHost(const Index ind, const Int relInd) const {return _hostedges(ind, relInd);}

    /** @brief Get the index of the center particle (from Coords) associated with a face

    @hostfn

    @param ind face index
    @return index to Coords for face(ind)'s center particle
    */
    inline Index getCenterIndHost(const Index ind) const {return _hostcenters(ind);}

    /** @brief Returns true if an edge is positive oriented about a face


      @hostfn

      @param faceInd face whose edge needs querying
      @param relEdgeInd relative edge index on face (see MeshSeed)
      @param edges collection of Edges
    */
    inline bool edgeIsPositive(const Index faceInd, const Int relEdgeInd, const Edges& edges) const {
      return faceInd == edges.getLeftHost(_hostedges(faceInd, relEdgeInd));
    }


    /** Overwrites a face's area

    @hostfn

    @param ind index of face whose area needs updating
    @param ar new value for area
    */
    inline void setAreaHost(const Index ind, const Real ar) {_hostarea(ind)= ar;}

    inline Real getAreaHost(const Index& ind) const {return _hostarea(ind);}

    /** @brief Decreases the number of leaves by one.

    @hostfn
    */
    inline void decrementnLeaves() {_hnLeaves() -= 1;}

    /** @brief Increases the face leaf count

    @hostfn
    */
    inline void incrementnLeaves(const Short& i=1) {_hnLeaves() += i;}

    /** @brief Writes basic info about a Faces object's state to a string.

    @hostfn

    @param label name for Faces object
    */
    std::string infoString(const std::string& label, const int& tab_level=0, const bool& dump_all=false) const;

    /** @brief Initialize a collection of Faces from a MeshSeed

    @hostfn

    @param seed
    */
    template <typename SeedType>
    void initFromSeed(const MeshSeed<SeedType>& seed);

    /** @brief Returns the number of leaves in the Faces tree.

    @hostfn
    */
    Index nLeavesHost() const {return _hnLeaves();}

    /** @brief Returns the surface area, as seen from the host

    @hostfn If the device updates face areas, this function will  not see it unless updateHost() is called first.
    */
    Real surfAreaHost() const;

//     / Host function
//     void setCenterInd(const Index faceInd, const Index crdInd) {_hostcenters(faceInd) = crdInd;}
//
//     / Host function
//     ko::View<const Index[FaceKind::nverts], Host> getVertsHostConst(const Index ind) const {
//       return ko::subview(_hostverts, ind, ko::ALL());
//     }
//



  protected:

    host_vertex_view _hostverts;
    host_edge_view _hostedges;
    host_index_view _hostcenters;
    host_index_view _hostparent;
    face_tree_host _hostkids;
    ko::View<Index>::HostMirror _nh;
    ko::View<Index>::HostMirror _hnLeaves;
    typename mask_view_type::HostMirror _hmask;
    typename ko::View<Int*,Dev>::HostMirror _hlevel;
    host_scalar _hostarea;

    Index _nmax;
};

template <typename Geo, typename FaceType> struct FaceDivider {
  static void divide(const Index faceInd, Coords<Geo>& physVerts, Coords<Geo>& lagVerts,
    Edges& edges, Faces<FaceType>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces) {}
};

template <typename Geo> struct FaceDivider<Geo, TriFace> {
  static void divide(const Index faceInd, Coords<Geo>& physVerts, Coords<Geo>& lagVerts,
    Edges& edges, Faces<TriFace>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces) ;
};

template <typename Geo> struct FaceDivider<Geo, QuadFace> {
  static void divide(const Index faceInd, Coords<Geo>& physVerts, Coords<Geo>& lagVerts,
    Edges& edges, Faces<QuadFace>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces) ;
};

template <> struct FaceDivider<CircularPlaneGeometry,QuadFace> {
  static void divide(const Index faceInd, Coords<CircularPlaneGeometry>& physVerts,
    Coords<CircularPlaneGeometry>& lagVerts, Edges& edges,
    Faces<QuadFace>& faces, Coords<CircularPlaneGeometry>& physFaces,
    Coords<CircularPlaneGeometry>& lagFaces) ;
};

}
#endif
