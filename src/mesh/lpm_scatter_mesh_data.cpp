#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"

namespace Lpm {

// ETI
template class ScatterMeshData<QuadRectSeed>;
template class ScatterMeshData<TriHexSeed>;
template class ScatterMeshData<IcosTriSphereSeed>;
template class ScatterMeshData<CubedSphereSeed>;

}
