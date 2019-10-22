#ifndef LPM_NODE_ARRAYD_HPP
#define LPM_NODE_ARRAYD_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {
namespace Octree {
/** 
    Implements Listing 1 from: 
    
    K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction, 
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75 
*/

/**
    Node array at depth D
    Input: pts = kokkos view of 3d xyz coordinates
    
    Step 1: Bounding box
    Step 2: compute the key of each point and encode the pt id with its key
    Step 3: Sort by key, rearrange points into sorted order.
    Step 4: Consolidate unique nodes.
    Step 5: Reserve space for full sibling sets  
    Step 6: Build NodeArrayD
*/
class NodeArrayD {
    public:
    ko::View<Real*[3]> sorted_pts; /// point coordinates in R3 (input).
    Int depth; /// maximum depth of octree.
    
    ko::View<BBox> box; /// Bounding box
    ko::View<key_type*> node_keys; /// node_keys(i) = shuffled xyz key of node i
    ko::View<Index*[2]> node_pt_inds; /** node_pt_inds(i,0) = address of first point (in sorted_pts) contained by node i
                                          node_pt_inds(i,1) = number of points contained by node i */
    ko::View<Index*> node_parents; /// allocated here; set by level D-1
    
    ko::View<Index*> pt_in_node; /// pt_in_node(i) = index of the node that contains point i
    ko::View<Index*> orig_ids; /// original (presort) locations of points
        
    NodeArrayD(const ko::View<Real*[3]>& p, const Int& d) : depth(d), sorted_pts("sorted_points", p.extent(0)),
        pt_in_node("pt_in_node", p.extent(0)), orig_ids("original_pt_locs", p.extent(0)), box("bbox") {init(p);}
    
    /**
        Listing 1:  Initializer for lowest level of octree
    */
    void init(const ko::View<Real*[3]>& presorted_points);
    
    std::string infoString() const;
};

}}
#endif
