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

struct TriFace {
    static constexpr Int nverts = 3;
};

struct QuadFace {
    static constexpr Int nverts = 4;
};

/** All initialization / changes occur on host.  Device arrays are const.

*/
template <typename FaceKind> class Faces {
    public:
        typedef ko::View<Index*[FaceKind::nverts]> vertex_view_type;
        typedef vertex_view_type edge_view_type;
        typedef typename vertex_view_type::HostMirror host_vertex_view;
        typedef host_vertex_view host_edge_view;
        typedef ko::View<Index*[4]> face_tree_view;
        typedef typename face_tree_view::HostMirror face_tree_host;
        typedef ko::View<Index*> index_view;
        typedef typename index_view::HostMirror host_index_view;
        typedef ko::View<Real*> scalar_view;
        typedef typename scalar_view::HostMirror host_scalar;
#ifdef HAVE_CUDA
        typedef ko::View<Index[FaceKind::nverts], ko::LayoutStride, Dev, ko::MemoryTraits<ko::Unmanaged>> vert_ind_view;
        typedef vert_ind_view edge_ind_view;
        typedef ko::View<Index[FaceKind::nverts], ko::LayoutStride, Host, ko::MemoryTraits<ko::Unmanaged>> host_vert_inds;
        typedef host_vert_inds host_edge_inds;
#else
        typedef Index* vert_ind_view;
        typedef vert_ind_view edge_ind_view;
        typedef Index* host_vert_inds;
        typedef host_vert_inds host_edge_inds;
#endif

        Faces(const Index nmax) : _verts("face_verts", nmax), _edges("face_edges", nmax), _centers("centers",nmax),
            _parent("parent", nmax), _kids("kids", nmax), _n("n"), _nmax(nmax), _area("area", nmax), _nActive("nActive") {
            _hostverts = ko::create_mirror_view(_verts);
            _hostedges = ko::create_mirror_view(_edges);
            _hostcenters = ko::create_mirror_view(_centers);
            _hostparent = ko::create_mirror_view(_parent);
            _hostkids = ko::create_mirror_view(_kids);
            _nh = ko::create_mirror_view(_n);
            _hostarea = ko::create_mirror_view(_area);
            _hnActive = ko::create_mirror_view(_nActive);
            _nh(0) = 0;
            _hnActive(0) = 0;
        }
        
        void updateDevice() {
            ko::deep_copy(_verts, _hostverts);
            ko::deep_copy(_edges, _hostedges);
            ko::deep_copy(_centers, _hostcenters);
            ko::deep_copy(_parent, _hostparent);
            ko::deep_copy(_kids, _hostkids);
            ko::deep_copy(_area, _hostarea);
            ko::deep_copy(_n, _nh);
            ko::deep_copy(_nActive, _hnActive);
        }
        
        void updateHost() {
            ko::deep_copy(_hostarea, _area);
        }
        
        KOKKOS_INLINE_FUNCTION
        Index getCenterInd(const Index ind) const {return _centers(ind);}
        
        KOKKOS_INLINE_FUNCTION
        vert_ind_view getVerts(const Index ind) {return slice(_verts, ind);}
        
        KOKKOS_INLINE_FUNCTION
        edge_ind_view getEdges(const Index ind) {return slice(_edges, ind);}
                
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n(0);}
        
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < _n(0) && _kids(ind, 0) >= 0;}
        
        KOKKOS_INLINE_FUNCTION
        void setArea(const Index ind, const Real ar) {_area(ind) = ar;}
        
        /// Host function
        inline Index nMax() const {return _verts.extent(0);}
        
        /// Host function
        inline Index nh() const {return _nh(0);}
        
        /// Host function
        void insertHost(const Index ctr_ind, ko::View<Index*,Host> vertinds, ko::View<Index*,Host> edgeinds, const Index prt=NULL_IND, const Real ar = 0.0);
        
        /// Host function
        void setKids(const Index parent, const Index* kids);

        /// Host function
        inline bool hasKidsHost(const Index ind) const {return ind < _nh(0) && _hostkids(ind, 0) >= 0;}
        
        /// Host function
        template <typename CV>
        void setKidsHost(const Index ind, const CV v) {
            for (int i=0; i<4; ++i) {
                _hostkids(ind, i) = v(i);
            }
        }
        
        /// Host function
        Index getVertHost(const Index ind, const Int relInd) const {return _hostverts(ind, relInd);}

        /// Host function
        Index getEdgeHost(const Index ind, const Int relInd) const {return _hostedges(ind, relInd);}
        
        /// Host function
        inline Index getCenterIndHost(const Index ind) const {return _hostcenters(ind);}
        
        /// Host function
        inline bool edgeIsPositive(const Index faceInd, const Int relEdgeInd, const Edges& edges) const { 
            return faceInd == edges.getLeftHost(_hostedges(faceInd, relEdgeInd));
        }
        
        /// Host function
        inline void setAreaHost(const Index ind, const Real ar) {_hostarea(ind)= ar;}
        
        /// Host function
        inline void decrementNActive() {_hnActive(0) -= 1;}
        
        /// Host function
        std::string infoString(const std::string& label) const;
        
        /// Host function
        template <typename SeedType>
        void initFromSeed(const MeshSeed<SeedType>& seed);
        
        /// Host function
        Real surfAreaHost() const;
        
        /// Host function
        template <typename Geo>
        void divide(const Index ind, Edges& edges, Coords<Geo>& intr_crds, Coords<Geo>& intr_lagcrds, 
            Coords<Geo>& bndry_crds, Coords<Geo>& bndry_lagcrds);
        
    protected:
        vertex_view_type _verts;  /// indices to Coords<Geo> on face edges
        edge_view_type _edges; /// indices to Edges
        index_view _centers; /// indices to Coords<Geo> inside faces
        index_view _parent; /// indices to Faces<FaceKind>
        face_tree_view _kids; /// indices to Faces<FaceKind>
        ko::View<Index> _n; /// number of Faces currently defined
        ko::View<Index> _nActive; /// number of leaf Faces
        scalar_view _area; /// Areas of each face
        
        host_vertex_view _hostverts;
        host_edge_view _hostedges;
        host_index_view _hostcenters;
        host_index_view _hostparent;
        face_tree_host _hostkids;
        ko::View<Index>::HostMirror _nh;
        ko::View<Index>::HostMirror _hnActive;
        host_scalar _hostarea;
        
        Index _nmax;
};

template <typename Geo, typename FaceKind> struct FaceDivider {
    static void divide(const Index faceInd, Faces<FaceKind>& faces, Edges& edges, 
        Coords<Geo>& intr_crds, Coords<Geo>& intr_lagcrds, 
        Coords<Geo>& bndry_crds, Coords<Geo>& bndry_lagcrds) {}
};

template <typename Geo> struct FaceDivider<Geo, TriFace> {
    static void divide(const Index faceInd, Faces<TriFace>& faces, Edges& edges, 
        Coords<Geo>& intr_crds, Coords<Geo>& intr_lagcrds, 
        Coords<Geo>& bndry_crds, Coords<Geo>& bndry_lagcrds) ;
};

template <typename Geo> struct FaceDivider<Geo, QuadFace> {
    static void divide(const Index faceInd, Faces<QuadFace>& faces, Edges& edges, 
        Coords<Geo>& intr_crds, Coords<Geo>& intr_lagcrds, 
        Coords<Geo>& bndry_crds, Coords<Geo>& bndry_lagcrds);
};

}
#endif
