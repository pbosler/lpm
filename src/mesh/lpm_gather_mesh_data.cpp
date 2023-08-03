#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"

namespace Lpm {

/// ETI
template class GatherMeshData<QuadRectSeed>;
template class GatherMeshData<TriHexSeed>;
template class GatherMeshData<IcosTriSphereSeed>;
template class GatherMeshData<CubedSphereSeed>;
}
