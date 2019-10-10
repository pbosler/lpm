#include "LpmOctreeLUT.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <sstream>
#include <typeinfo>

namespace Lpm {
namespace Octree {

template Int table_val(const Int& i, const Int& j, const ko::View<ParentLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<ChildLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborsAtVertexLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborsAtEdgeLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborEdgeComplementLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<EdgeVerticesLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborsAtFaceLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<NeighborFaceComplementLUT>& tableview);
template Int table_val(const Int& i, const Int& j, const ko::View<FaceEdgesLUT>& tableview);
}}
