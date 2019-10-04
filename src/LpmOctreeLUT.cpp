#include "LpmOctreeLUT.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {
namespace Octree {

template Int table_val(const Int& i, const Int& j, const ko::View<ParentLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<ChildLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborsAtVertexLUT>& tableview);
}}
