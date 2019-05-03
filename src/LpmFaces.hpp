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

/** All initialization / changes occur on host.  Device arrays are const.

*/
template <typename FaceKind> class Faces {
    public:
        typedef ko::View<Index*[FaceKind::nverts]> vertex_view_type;
        typedef vertex_view_type edge_view_type;
        typedef ko::View<Index*[4]> face_tree_view;
        template <typename Geo, typename FaceType> friend struct FaceDivider;
        static constexpr Int nverts = FaceKind::nverts;
        
        vertex_view_type verts;  /// indices to Coords<Geo> on face edges
        edge_view_type edges; /// indices to Edges
        index_view_type centers; /// indices to Coords<Geo> inside faces
        index_view_type parent; /// indices to Faces<FaceKind>
        face_tree_view kids; /// indices to Faces<FaceKind>
        n_view_type n; /// number of Faces currently defined
        n_view_type nLeaves; /// number of leaf Faces
        scalar_view_type area; /// Areas of each face
        
        Faces(const Index nmax) : verts("faceverts", nmax), edges("faceedges", nmax), centers("centers",nmax),
            parent("parent", nmax), kids("kids", nmax), n("n"), _nmax(nmax), area("area", nmax), nLeaves("nLeaves") {
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
        }
        
        void updateDevice() const {
            ko::deep_copy(verts, _hostverts);
            ko::deep_copy(edges, _hostedges);
            ko::deep_copy(centers, _hostcenters);
            ko::deep_copy(parent, _hostparent);
            ko::deep_copy(kids, _hostkids);
            ko::deep_copy(area, _hostarea);
            ko::deep_copy(n, _nh);
            ko::deep_copy(nLeaves, _hnLeaves);
        }
        
        void updateHost() const {
            ko::deep_copy(_hostarea, area);
        }
        
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < n() && kids(ind,0) >= 0;}
        

/*/////  HOST FUNCTIONS ONLY BELOW THIS LINE         
    
        todo: make them protected, not public    
        
*/      
        typename scalar_view_type::HostMirror getAreaHost() const {return _hostarea;}
         
        /// Host function
        inline Index nMax() const {return _nmax;}
        
        /// Host function
        inline Index nh() const {return _nh();}
        
        /// Host function
        void insertHost(const Index ctr_ind, ko::View<Index*,Host> vertinds, ko::View<Index*,Host> edgeinds, const Index prt=NULL_IND, const Real ar = 0.0);
        
        /// Host function
        void setKids(const Index parent, const Index* kids);

        /// Host function
        inline bool hasKidsHost(const Index ind) const {return ind < _nh() && _hostkids(ind, 0) >= 0;}
        
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
        inline void decrementnLeaves() {_hnLeaves() -= 1;}
        
        /// Host function
        std::string infoString(const std::string& label) const;
        
        /// Host function
        template <typename SeedType>
        void initFromSeed(const MeshSeed<SeedType>& seed);
        
        Index nLeavesHost() const {return _hnLeaves();}
        
        /// Host function
        Real surfAreaHost() const;
        
//         / Host function
//         void setCenterInd(const Index faceInd, const Index crdInd) {_hostcenters(faceInd) = crdInd;}
//         
//         / Host function
//         ko::View<const Index[FaceKind::nverts], Host> getVertsHostConst(const Index ind) const {
//             return ko::subview(_hostverts, ind, ko::ALL());
//         }
//         
    protected:
        typedef typename face_tree_view::HostMirror face_tree_host;
        typedef typename index_view_type::HostMirror host_index_view;
        typedef typename vertex_view_type::HostMirror host_vertex_view;
        typedef host_vertex_view host_edge_view;
        typedef typename scalar_view_type::HostMirror host_scalar;
        host_vertex_view _hostverts;
        host_edge_view _hostedges;
        host_index_view _hostcenters;
        host_index_view _hostparent;
        face_tree_host _hostkids;
        ko::View<Index>::HostMirror _nh;
        ko::View<Index>::HostMirror _hnLeaves;
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

}
#endif
