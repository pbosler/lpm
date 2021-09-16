#ifndef LPM_CPU_TREE_HPP
#define LPM_CPU_TREE_HPP

#include "LpmConfig.h"
#include "lpm_box3d.hpp"
#include "lpm_coords.hpp"

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkCellArray.h"

#include <memory>
#include <vector>

namespace Lpm {
namespace tree {

struct Node {
  Node(const Box3d& bb, Node* p = NULL,
    const std::vector<Index>& inds = std::vector<Index>()) :
    box(bb),
    level((p ? p->level+1 : 0)),
    parent(p),
    crd_inds(inds) {}

  virtual ~Node() {}

  inline Index n() const {return crd_inds.size();}
  inline bool is_leaf() const {return kids.empty();}
  inline bool has_kids() const {return !is_leaf();}
  inline bool empty() const {return crd_inds.empty();}

  Box3d box;
  int level;
  Node* parent;
  std::vector<std::unique_ptr<Node>> kids;
  std::vector<Index> crd_inds;

};


template <typename Geo>
struct CpuTree {
  public:
    enum TreeDepthControl {MaxCoordsPerNode, MaxDepth};

    static_assert(!std::is_same<Geo,PlaneGeometry>::value,
      "PlaneGeometry not implemented yet.");

    CpuTree(const std::shared_ptr<Coords<Geo>> crds);

    virtual ~CpuTree() {}

    int depth;
    Index n_nodes;
    std::unique_ptr<Node> root;

    void write_vtk(const std::string& ofname) const;

  protected:
    CpuTree() : depth(0), n_nodes(0) {}

    void generate_tree_max_coords_per_node(Node* node,
      const Index max_coords_per_node,
      const bool do_shrink);

    void generate_tree_max_depth(Node* node, int max_depth, const bool do_shrink);

    void shrink_box(Node* node);

    void divide_node(Node* node, const bool do_shrink);

    void insert_vtk_cell_points(
      vtkSmartPointer<vtkPoints> pts, const Index pt_offset,
      vtkSmartPointer<vtkCellArray> cells, const Index cell_offset,
      vtkSmartPointer<vtkIntArray> levels, const Node* node);

    std::shared_ptr<Coords<Geo>> _crds;
};


} // namespace tree
} // namespace Lpm

#endif
