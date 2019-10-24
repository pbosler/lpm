#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"
#include "Kokkos_Core.hpp"
#include <cassert>
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkPoints.h"

namespace Lpm {
namespace Octree {

class Tree {
    public:
        typedef typename ko::View<Index*>::HostMirror index_view_host;
        
        // point arrays
        ko::View<Index*> pt_in_leaf;
        ko::View<Index*> pt_orig_id;
        ko::View<Real*[3]> presorted_pts;
        ko::View<Real*[3]> sorted_pts;

        // node arrays
        ko::View<key_type*> node_keys;
        ko::View<Index*[2]> node_pt_inds;
        ko::View<Index*> node_parents;
        ko::View<Index*[8]> node_kids;
        ko::View<Index*[27]> node_neighbors;
        
        // connectivity arrays
        ko::View<Index*[8]> node_vertices;
        ko::View<Index*[12]> node_edges;
        ko::View<Index*[6]> node_faces;
        
        ko::View<Index*[8]> vertex_nodes;
        ko::View<Index*[2]> edge_vertices;
        ko::View<Index*[4]> face_edges;
        
        // tree arrays 
        ko::View<Index*> base_address;
        ko::View<Index*> nnodes_per_level;

        // tree scalars
        Int max_depth;        
        ko::View<BBox> box;     
        bool do_connectivity;
        
        Tree(const ko::View<Real*[3]>& p, const Int& md, const bool& do_conn=false) : presorted_pts(p),
            sorted_pts("sorted_pts", p.extent(0)), pt_in_leaf("pt_in_leaf", p.extent(0)), 
            pt_orig_id("pt_orig_id", p.extent(0)), max_depth(md), 
            base_address("base_address", md+1), nnodes_per_level("nnodes_per_level", md+1), 
            box("bbox"), do_connectivity(do_conn) {
                initNodes();
            }
        
        std::string infoString() const;
        
    //protected:
        /**
            Initializes the node arrays
        */
        void initNodes();
        /**
            Initializes node-vertex connectivity relations.  Must be called after init().
        */
        void initVertices(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase);
        
        /** 
            Initializes node-edge connectivity.  Must be called after initVertices().
        */
        void initEdges(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase);
        
        /** 
            Initializes node-face connectivity.  Must be called after initEdges().
        */
        void initFaces(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase);
};

// struct NeighborhoodFunctor {
//     // output
//     ko::View<Index*[27]> neighbors;
//     // input
//     ko::View<key_type*> keys;
//     ko::View<Index*[8]> kids;
//     ko::View<Index*> parents;
//     ko::View<Index*> base_address;
//     ko::View<ParentLUT> ptable;
//     ko::View<ChildLUT> ctable;
//     Int level; // assume > 0
//     Int max_depth;
//     
//     NeighborhoodFunctor(ko::View<Index*[27]>& n, const ko::View<key_type*>& k, const ko::View<Index*[8]>& c,
//         const ko::View<Index*>& p, const ko::View<Index*>& b, const Int& l, const Int& m) :
//         neighbors(n), keys(k), kids(c), parents(p), base_address(b), level(l), max_depth(m), 
//         ptable("ParentLUT"), ctable("ChildLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& mbr) const {
//         const Index t = base_address(level) + mbr.league_rank();
//         const Index p = parents(t);
//         const key_type i = local_key(keys(t), level, max_depth);
//         ko::parallel_for(ko::TeamThreadRange(mbr, 27), KOKKOS_LAMBDA (const Index& j) {
//             const Index plut = table_val(i,j, ptable);
//             if ( neighbors(p,plut) != NULL_IND) {
//                 const Index h = neighbors(p, plut);
//                 neighbors(t,j) = kids(h, table_val(i,j, ctable));
//             }
//             else {
//                 neighbors(t,j) = NULL_IND;
//             }
//         });
//     }
// };
// 
// KOKKOS_INLINE_FUNCTION
// Index nvertsAtLevel(const Int& lev) {
//     Index points_per_edge = 2;
//     for (Int l=1; l<=lev; ++l) {
//         points_per_edge = 2*points_per_edge-1;
//     }
//     return cube(points_per_edge);
// }
// 
// 
// 
// struct VertexOwnerFunctor {
//     // output
//     ko::View<Index*[8]> owner;
//     // input
//     ko::View<key_type*> keys;
//     ko::View<Index*[27]> neighbors;
//     // local
//     ko::View<NeighborsAtVertexLUT> nvtable;
//     typedef typename ko::MinLoc<key_type,Index>::value_type minloc_type;
//     
//     VertexOwnerFunctor(ko::View<Index*[8]>& o, const ko::View<key_type*>& k, const ko::View<Index*[27]>& n) : 
//         owner(o), keys(k), neighbors(n), nvtable("NeighborsAtVertexLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& mbr) const {
//         const Index t = mbr.league_rank();  // each team is assigned a node
//         ko::parallel_for(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Int& i) {
//             // each thread gets a vertex of the team's node
//             minloc_type result;
//             // determine node that owns vertex v (the node sharing v with lowest key)
//             ko::parallel_reduce(ko::ThreadVectorRange(mbr,8), [=] (const Int& j, minloc_type& loc) {
//                const Index nbr_ind = neighbors(t, table_val(i,j,nvtable));
//                if (nbr_ind != NULL_IND) {
//                     if (keys(nbr_ind) < loc.val) {
//                         loc.val = keys(nbr_ind);
//                         loc.loc = nbr_ind;
//                     }
//                }
//                }, ko::MinLoc<key_type,Index>(result));
//             owner(t,i) = result.loc;
//         });
//     }
// };
// 
// struct EdgeOwnerFunctor {
//     // output
//     ko::View<Index*[12]> owner;
//     // input
//     ko::View<key_type*> keys;
//     ko::View<Index*[27]> neighbors;
//     // local
//     ko::View<NeighborsAtEdgeLUT> netable;
//     typedef typename ko::MinLoc<key_type,Index>::value_type minloc_type;
//     
//     EdgeOwnerFunctor(ko::View<Index*[12]>& o, const ko::View<key_type*>& k, const ko::View<Index*[27]>& n) :
//         owner(o), keys(k), neighbors(n), netable("NeighborsAtEdgeLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator () (const member_type& mbr) const {
//         const Index t = mbr.league_rank();
//         ko::parallel_for(ko::TeamThreadRange(mbr,12), KOKKOS_LAMBDA (const Int& i) {
//             minloc_type result;
//             ko::parallel_reduce(ko::ThreadVectorRange(mbr,4), [=] (const Int& j, minloc_type& loc) {
//                 const Index nbr_ind = neighbors(t, table_val(i,j,netable));
//                 if (nbr_ind != NULL_IND) {
//                     if (keys(nbr_ind) < loc.val) {
//                         loc.val = keys(nbr_ind);
//                         loc.loc = nbr_ind;
//                     }
//                 }
//             }, ko::MinLoc<key_type,Index>(result));
//             owner(t,i) = result.loc;
//         });    
//     }
// };
// 
// struct FaceOwnerFunctor {
//     // output
//     ko::View<Index*[6]> owner;
//     // input
//     ko::View<key_type*> keys;
//     ko::View<Index*[27]> neighbors;
//     // local
//     ko::View<NeighborsAtFaceLUT> nftable;
//     typedef typename ko::MinLoc<key_type, Index>::value_type minloc_type;
//     
//     FaceOwnerFunctor(ko::View<Index*[6]>& o, const ko::View<key_type*>& k, const ko::View<Index*[27]>& n) : 
//         owner(o), keys(k), neighbors(n), nftable("NeighborsAtFaceLUT") {}
// 
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& team) const {
//         const Index t = team.league_rank();
//         ko::parallel_for(ko::TeamThreadRange(team,6), KOKKOS_LAMBDA (const Int& i) {
//             const Index nbr1 = neighbors(t, table_val(i,0, nftable));
//             const Index nbr2 = neighbors(t, table_val(i,1, nftable));
//             const key_type key1 = keys(nbr1);
//             const key_type key2 = keys(nbr2);
//             owner(t,i) = (key1<key2 ? nbr1 : nbr2);
//         });
//     }
// };
// 
// 
// 
// struct VertexFlagFunctor {
//     // output
//     ko::View<Int*[8]> flags;
//     ko::View<Int*> nverts_at_node;
//     // input
//     ko::View<Index*[8]> owner; 
//     
//     VertexFlagFunctor(ko::View<Index*[8]>& f, ko::View<Int*>& nv, const ko::View<Index*[8]>& o) : flags(f), 
//         nverts_at_node(nv), owner(o) {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& mbr) const {
//         const Index t = mbr.league_rank();
//         ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i) {
//             flags(t,i) = (owner(t,i) == t ? 1 : 0);
//         });
//         mbr.team_barrier();
//         ko::parallel_reduce(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Int& i, Int& nv) {
//             nv += flags(t,i);
//         }, nverts_at_node(t));
//     }
// };
// 
// struct EdgeFlagFunctor {
//     // output
//     ko::View<Int*[12]> flags;
//     ko::View<Int*> nedges_at_node;
//     // input
//     ko::View<Index*[12]> owner;
//     
//     EdgeFlagFunctor(ko::View<Int*[12]>& f, ko::View<Int*>& ne, const ko::View<Index*[12]>& o) : flags(f),
//         nedges_at_node(ne), owner(o) {}
//         
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& mbr) const {
//         const Index t = mbr.league_rank();
//         ko::parallel_for(ko::TeamThreadRange(mbr,12), KOKKOS_LAMBDA (const Int& i) {
//             flags(t,i) = (owner(t,i) == t ? 1 : 0);
//         });
//         mbr.team_barrier();
//         ko::parallel_reduce(ko::TeamThreadRange(mbr,12), KOKKOS_LAMBDA (const Int& i, Int& ne) {
//             ne += flags(t,i);
//         },nedges_at_node(t));
//     }
// };
// 
// struct FaceFlagFunctor {
//     // output
//     ko::View<Int*[6]> flags;
//     ko::View<Int*> nfaces_at_node;
//     // input
//     ko::View<Index*[6]> owner;
//     
//     FaceFlagFunctor(ko::View<Int*[6]>& f, ko::View<Int*> nf, const ko::View<Index*[6]>& o) : flags(f),
//         nfaces_at_node(nf), owner(o) {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& team) const {
//         const Index t = team.league_rank();
//         ko::parallel_for(ko::TeamThreadRange(team,6), KOKKOS_LAMBDA (const Int& i) {
//             flags(t,i) = (owner(t,i) == t ? 1 : 0);
//         });
//         team.team_barrier();
//         ko::parallel_reduce(ko::TeamThreadRange(team,6), KOKKOS_LAMBDA (const Int& i, Int& nf) {
//             nf += flags(t,i);
//         }, nfaces_at_node(t));
//     }
// };
// 
// struct VertexNodeFunctor {
//     // output
//     ko::View<Index*[8]> vert_nodes;
//     ko::View<Index*[8]> node_verts;
//     // input
//     ko::View<Int*[8]> flags;
//     ko::View<Index*[8]> vert_owner;
//     ko::View<Index*> vert_address;
//     ko::View<Index*[27]> neighbors;
//     ko::View<NeighborsAtVertexLUT> nvtable;
// 
//     VertexNodeFunctor(ko::View<Index*[8]>& vn, ko::View<Index*[8]>& nv, const ko::View<Int*[8]>& f, 
//         const ko::View<Index*[8]>& vo, const ko::View<Index*>& va, const ko::View<Index*[27]>& n) : 
//         vert_nodes(vn), node_verts(nv), flags(f), vert_owner(vo), vert_address(va), neighbors(n),
//         nvtable("NeighborsAtVertexLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& mbr) const {
//         const Index t = mbr.league_rank();
//         
//         Int* local_scan = (Int*)mbr.team_shmem().get_shmem(8*sizeof(Int));
// 
//         ko::parallel_for(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Int& i) {
//             Int flagi = 0;
//             ko::parallel_reduce(ko::ThreadVectorRange(mbr, i), [=] (const Int& j, Int& ct) {
//                 ct += flags(t,j);
//             }, flagi);
//             ko::single(ko::PerThread(mbr), [=] () {
//                 local_scan[i] = flagi;
//             });
//         });
//         mbr.team_barrier();
//         
//         ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i) {
//             if (vert_owner(t,i) == t) {
//                 const Index v = vert_address(t) + local_scan[i];
//                 ko::parallel_for(ko::ThreadVectorRange(mbr,8), [=] (const Int& j) {
//                     const Index nbr_ind = neighbors(t, table_val(i,j,nvtable));
//                     vert_nodes(v,j) = nbr_ind;
//                     node_verts(nbr_ind,7-j) = v;
//                 });
//             }
//         });
//     }
//     
//     size_t team_shmem_size(int team_size) const {
//         return 8*sizeof(Int);
//     }
// };
// 
// struct EdgeNodeFunctor {
//     // output
//     ko::View<Index*[2]> edge_verts;
//     ko::View<Index*[12]> node_edges;
//     // input
//     ko::View<Int*[12]> flags;
//     ko::View<Index*[12]> owner;
//     ko::View<Index*> address;
//     ko::View<Index*[27]> neighbors;
//     ko::View<Index*[8]> vertices;
//     // local
//     ko::View<NeighborsAtEdgeLUT> netable;
//     ko::View<EdgeVerticesLUT> evtable;
//     ko::View<NeighborEdgeComplementLUT> nectable;
//     
//     EdgeNodeFunctor(ko::View<Index*[2]>& ev, ko::View<Index*[12]>& ne, const ko::View<Int*[12]>& f,
//         const ko::View<Index*[12]>& o, const ko::View<Index*>& a, const ko::View<Index*[27]>& nn,
//         const ko::View<Index*[8]>& nv) : edge_verts(ev), node_edges(ne), flags(f), owner(o), 
//             address(a), neighbors(nn), vertices(nv), netable("NeighborsAtEdgeLUT"),
//             evtable("EdgeVerticesLUT"), nectable("NeighborEdgeComplementLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator () (const member_type& mbr) const {
//         const Index t = mbr.league_rank();
//         
//         Int* local_scan = (Int*)mbr.team_shmem().get_shmem(12*sizeof(Int));
//         
//         ko::parallel_for(ko::TeamThreadRange(mbr, 12), KOKKOS_LAMBDA (const Int& i) {
//             Int flagi = 0;
//             ko::parallel_reduce(ko::ThreadVectorRange(mbr, i), [=] (const Int& j, Int& ct)  {
//                 ct += flags(t,j);
//             }, flagi);
//             ko::single(ko::PerThread(mbr), [=] () {
//                 local_scan[i] = flagi;
//             });
//         });
//         mbr.team_barrier();
//         
//         ko::parallel_for(ko::TeamThreadRange(mbr, 12), KOKKOS_LAMBDA (const Int& i) {
//             if (owner(t,i) == t) {
//                 const Index e = address(t) + local_scan[i];
//                 edge_verts(e,0) = vertices(t, table_val(i,0, evtable));
//                 edge_verts(e,1) = vertices(t, table_val(i,1, evtable));
//                 ko::parallel_for(ko::ThreadVectorRange(mbr, 4), [=] (const Int& j) {
//                     const Index nbr_ind = neighbors(t, table_val(i,j,netable));
//                     node_edges(nbr_ind, table_val(i,j, nectable)) = e;
//                 });
//             }
//         });
//     }
//     
//     size_t team_shmem_size(int team_size) const {
//         return 12*sizeof(Int);
//     }
// };
// 
// struct FaceNodeFunctor {
//     // output
//     ko::View<Index*[4]> face_edges;
//     ko::View<Index*[6]> node_faces;
//     // input
//     ko::View<Int*[6]> flags;
//     ko::View<Index*[6]> owner;
//     ko::View<Index*> address;
//     ko::View<Index*[27]> neighbors;
//     ko::View<Index*[12]> edges;
//     // local
//     ko::View<NeighborsAtFaceLUT> nftable;
//     ko::View<FaceEdgesLUT> fetable;
//     ko::View<NeighborFaceComplementLUT> nfctable;
//     
//     FaceNodeFunctor(ko::View<Index*[4]>& fe, ko::View<Index*[6]>& nf, const ko::View<Int*[6]>& f,
//         const ko::View<Index*[6]>& o, const ko::View<Index*>& a, const ko::View<Index*[27]>& nn, 
//         const ko::View<Index*[12]>& ne) : face_edges(fe), node_faces(nf), flags(f), owner(o),
//             address(a), neighbors(nn), edges(ne), nftable("NeighborsAtFaceLUT"), fetable("FaceEdgesLUT"),
//             nfctable("NeighborFaceComplementLUT") {}
//     
//     KOKKOS_INLINE_FUNCTION
//     void operator() (const member_type& team) const {
//         const Index t = team.league_rank();
//         
//         Int* local_scan = (Int*)team.team_shmem().get_shmem(6*sizeof(Int));
//         
//         ko::parallel_for(ko::TeamThreadRange(team, 6), KOKKOS_LAMBDA (const Int& i) {
//             Int flagi = 0;
//             ko::parallel_reduce(ko::ThreadVectorRange(team,i), [=] (const Int& j, Int& ct) {
//                 ct += flags(t,j);
//             }, flagi);
//             ko::single(ko::PerThread(team), [=] () {
//                 local_scan[i] = flagi;
//             });
//         });
//         team.team_barrier();
//         
//         ko::parallel_for(ko::TeamThreadRange(team, 6), KOKKOS_LAMBDA (const Int& i) {
//             if (owner(t,i) == t) {
//                 const Index f = address(t) + local_scan[i];
//                 ko::parallel_for(ko::ThreadVectorRange(team, 4), [=] (const Int& j) {
//                     face_edges(f,j) = edges(t, table_val(i,j, fetable));
//                 });
//                 const Index nbr0 = neighbors(t, table_val(i,0, nftable));
//                 const Index nbr1 = neighbors(t, table_val(i,1, nftable));
//                 node_faces(nbr0, table_val(i,0, nfctable)) = f;
//                 node_faces(nbr1, table_val(i,1, nfctable)) = f;
//             }
//         });   
//     }
//     
//     size_t team_shmem_size(int team_size) const {
//         return 6*sizeof(Int);
//     }
// };

}}
#endif

