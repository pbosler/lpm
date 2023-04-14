#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include <cassert>

#include "Kokkos_Core.hpp"
#include "LpmBox3d.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"
#include "LpmOctreeKernels.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmUtilities.hpp"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkSmartPointer.h"

namespace Lpm {
namespace Octree {

class Tree {
 public:
  typedef typename ko::View<Index*>::HostMirror index_view_host;

  // point arrays
  ko::View<Index*> pt_in_leaf;
  ko::View<Index*> pt_orig_id;
  ko::View<Real* [3]> presorted_pts;
  ko::View<Real* [3]> sorted_pts;

  // node arrays
  ko::View<key_type*> node_keys;
  ko::View<Index* [2]> node_pt_inds;
  ko::View<Index*> node_parents;
  ko::View<Index* [8]> node_kids;
  ko::View<Index* [27]> node_neighbors;

  // connectivity arrays
  ko::View<Index* [8]> node_vertices;
  ko::View<Index* [12]> node_edges;
  ko::View<Index* [6]> node_faces;

  ko::View<Index* [8]> vertex_nodes;
  ko::View<Index* [2]> edge_vertices;
  ko::View<Index* [4]> face_edges;

  // tree arrays
  ko::View<Index*> base_address;
  ko::View<Index*> nnodes_per_level;
  typename ko::View<Index*>::HostMirror base_address_host;
  typename ko::View<Index*>::HostMirror nnodes_per_level_host;
  Index nnodes_total;

  // tree scalars
  Int max_depth;
  ko::View<BBox> box;
  bool do_connectivity;

  Tree(const ko::View<Real* [3]>& p, const Int& md, const bool& do_conn = false)
      : presorted_pts(p),
        sorted_pts("sorted_pts", p.extent(0)),
        pt_in_leaf("pt_in_leaf", p.extent(0)),
        pt_orig_id("pt_orig_id", p.extent(0)),
        max_depth(md),
        base_address("base_address", md + 1),
        nnodes_per_level("nnodes_per_level", md + 1),
        box("bbox"),
        do_connectivity(do_conn) {
    initNodes();
  }

  std::string infoString() const;

  // protected:
  /**
      Initializes the node arrays
  */
  void initNodes();
  /**
      Initializes node-vertex connectivity relations.  Must be called after
     initNodes().
  */
  void initVertices();

  /**
      Initializes node-edge connectivity.  Must be called after initVertices().
  */
  //         void initEdges(const std::vector<Index>& nnodes_at_level, const
  //         hbase_type& hbase);

  /**
      Initializes node-face connectivity.  Must be called after initEdges().
  */
  //         void initFaces(const std::vector<Index>& nnodes_at_level, const
  //         hbase_type& hbase);
};

}  // namespace Octree
}  // namespace Lpm
#endif
