#ifndef LPM_FACES_HPP
#define LPM_FACES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"

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
        typedef typename vertex_view_type::HostMirror host_vertex_view;
        typedef ko::View<Index*[4]> face_tree_view;
        typedef typename face_tree_view::HostMirror face_tree_host;
        typedef ko::View<Index*> index_view;
        typedef typename index_view::HostMirror host_index_view;
        typedef ko::View<Real*> scalar_view;
        typedef typename scalar_view::HostMirror host_scalar;

#ifdef HAVE_CUDA
        typedef ko::View<Index*, ko::LayoutStride, typename vertex_view_type::device_type,
            ko::MemoryTraits<ko::Unmanaged>> ind_slice;
        typedef ko::View<const Index*, ko::LayoutStride, typename vertex_view_type::device_type,
            ko::MemoryTraits<ko::Unmanaged>> const_ind_slice;
        typedef ko::View<Index*, ko::LayoutStride, typename vertex_view_type::host_mirror_space,
            ko::MemoryTraits<ko::Unmanaged>> host_ind_slice;
        typedef ko::View<const Index*, ko::LayoutStride, typename vertex_view_type::host_mirror_space,
            ko::MemoryTraits<ko::Unmanaged>> const_host_ind_slice;
#else
        typedef typename vertex_view_type::value_type* ind_slice;
        typedef typename vertex_view_type::const_value_type* const_ind_slice;
        typedef ind_slice host_ind_slice;
        typedef const_ind_slice const_host_ind_slice;
#endif 

        Faces(const Index nmax) : _verts("face_verts", nmax), _edges("face_edges", nmax),
            _parent("parent", nmax), _kids("kids", nmax), _n("n"), _nmax(nmax), _area("area", nmax) {
            _hostverts = ko::create_mirror_view(_verts);
            _hostedges = ko::create_mirror_view(_edges);
            _hostparent = ko::create_mirror_view(_parent);
            _hostkids = ko::create_mirror_view(_kids);
            _nh = ko::create_mirror_view(_n);
            _hostarea = ko::create_mirror_view(_area);
        }
        
        void updateDevice() {
            ko::deep_copy(_verts, _hostverts);
            ko::deep_copy(_edges, _hostedges);
            ko::deep_copy(_parent, _hostparent);
            ko::deep_copy(_kids, _hostkids);
            ko::deep_copy(_area, _hostarea);
            ko::deep_copy(_n, _nh);
        }
        
        void updateHost() {
            ko::deep_copy(_hostarea, _area);
        }
        
        KOKKOS_INLINE_FUNCTION
        ind_slice getVerts(const Index ind) {return slice(_verts, ind);}
        
        KOKKOS_INLINE_FUNCTION
        const_ind_slice getConstVerts(const Index ind) const {return const_slice(_verts, ind);}
        
        KOKKOS_INLINE_FUNCTION
        ind_slice getEdges(const Index ind) {return slice(_edges, ind);}
        
        KOKKOS_INLINE_FUNCTION
        const_ind_slice getConstEdges(const Index ind) const {return const_slice(_edges, ind);}
        
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
        void insertHost(const Index* vertinds, const Index* edgeinds, const Index prt=NULL_IND, const Real ar = 0.0);
        
        /// Host function
        void setKids(const Index parent, const Index* kids);

        /// Host function
        inline bool hasKidsHost(const Index ind) const {return ind < _nh(0) && _hostkids(ind, 0) >= 0;}
        
        /// Host function
        host_ind_slice getVertsHost(const Index ind) {return slice(_hostverts, ind);}
        const_host_ind_slice getConstVertsHost(const Index ind) const {return const_slice(_hostverts, ind);}

        /// Host function
        host_ind_slice getEdgesHost(const Index ind) {return slice(_hostedges, ind);}
        const_host_ind_slice getConstEdgesHost(const Index ind) const {return const_slice(_hostedges, ind);}
        
        /// Host function
        inline bool edgeHasPositiveOrientation(const Index faceInd, const Int relEdgeInd, const Edges& edges) const { 
            return faceInd == edges.getLeftHost(_hostedges(faceInd, relEdgeInd));
        }
        
        /// Host function
        inline void setAreaHost(const Index ind, const Real ar) {_hostarea(ind)= ar;}
        
    protected:
        vertex_view_type _verts;
        vertex_view_type _edges;
        index_view _parent;
        face_tree_view _kids;
        ko::View<Index> _n;
        scalar_view _area;
        
        host_vertex_view _hostverts;
        host_vertex_view _hostedges;
        host_index_view _hostparent;
        face_tree_host _hostkids;
        ko::View<Index>::HostMirror _nh;
        host_scalar _hostarea;
        
        Index _nmax;
};

template <typename Geo, typename FaceKind> struct FaceDivider {
    void divide(const Index faceInd, Faces<FaceKind>& faces, Edges& edges, Coords<Geo>& crds, Coords<Geo>& lagcrds) const {}
};

template <typename Geo> struct FaceDivider<Geo, TriFace> {
    void divide(const Index faceInd, Faces<TriFace>& faces, Edges& edges, Coords<Geo>& crds, Coords<Geo>& lagcrds) const;
};

template <typename Geo> struct FaceDivider<Geo, QuadFace> {
    void divide(const Index faceInd, Faces<QuadFace>& faces, Edges& edges, Coords<Geo>& crds, Coords<Geo>& lagcrds) const {}
};

}
#endif
