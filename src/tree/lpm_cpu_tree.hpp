#ifndef LPM_CPU_TREE_HPP
#define LPM_CPU_TREE_HPP

#include "LpmConfig.h"
#include "lpm_coords.hpp"

#include "tree/lpm_tree_node.hpp"
#include "tree/lpm_box3d.hpp"

#ifdef LPM_USE_VTK
#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkCellArray.h"
#include "vtkUnstructuredGrid.h"
#endif

#include <memory>

namespace Lpm {
namespace tree {

template <typename Geo=SphereGeometry, typename NodeType=Node>
struct CpuTree {
  public:
    static constexpr int MAX_CPU_TREE_DEPTH = 10;
    enum TreeDepthControl {MaxCoordsPerLeaf, MaxDepth};

    static_assert(!std::is_same<Geo,PlaneGeometry>::value,
      "PlaneGeometry not implemented yet.");

    CpuTree(const std::shared_ptr<Coords<Geo>> crds,
      const TreeDepthControl& ctl, const Int n_ctl, const bool shrink=false);

    virtual ~CpuTree() {}

    int depth;
    Index n_nodes;
    std::unique_ptr<NodeType> root;

#ifdef LPM_USE_VTK
    void write_vtk(const std::string& ofname) const;
#endif

    std::string info_string(const int tab_level = 0) const;

  protected:
    CpuTree() : depth(0), n_nodes(0) {}

    TreeDepthControl control;
    Int control_n;
    bool do_shrink;

    void generate_tree_max_coords_per_node(NodeType* node,
      const Index max_coords_per_node);

    void generate_tree_max_depth(NodeType* node, int max_depth);

    void shrink_box(NodeType* node);

    void divide_node(NodeType* node);

    std::string node_info_string(NodeType* node) const;

#ifdef LPM_USE_VTK
    void insert_vtk_cell_points(
      vtkSmartPointer<vtkPoints> pts,
      vtkSmartPointer<vtkUnstructuredGrid> ugrid,
      vtkSmartPointer<vtkIntArray> levels, const NodeType* node) const;

    std::shared_ptr<Coords<Geo>> _crds;
};
#endif

} // namespace tree
} // namespace Lpm

#endif
