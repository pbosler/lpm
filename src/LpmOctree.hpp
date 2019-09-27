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

namespace Lpm {
namespace Octree {



class Octree {
    public:
        ko::View<Real*[3]> pts;
        Int max_depth;
        
        ko::View<BBox> box;
        
        ko::View<key_type*> node_keys;
        ko::View<Index*> node_pt_idx;
        ko::View<Index*> node_pt_ct;
        ko::View<Index*> node_parents;
        ko::View<Index*[8]> node_kids;
        ko::View<Index*[27]> node_neighbors;
        ko::View<Index*[8]> node_vertices;
        ko::View<Index*[12]> node_edges;
        ko::View<Index*[6]> node_faces;
        
        ko::View<Index*> pt_in_leaf;
        ko::View<Index*[8]> vertex_nodes;
        ko::View<Index*[2]> edge_vertices;
        ko::View<Index*[4]> face_edges;
        ko::View<Index*> base_address;
        
        ko::View<Index[8][27]> parent_table;
        ko::View<Index[8][27]> child_table;
        
        Octree(const ko::View<Real*[3]>& p, const Int& md) : pts(p), max_depth(md), 
            pt_in_leaf("pt_in_leaf", p.extent(0)), base_address("base_address", max_depth+1),
            box("bbox"), parent_table("parent_table"), child_table("child_table") {
                init();
            }
        
    protected:
        void init();
    
};

}}
#endif
