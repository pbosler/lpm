#include "dfs/lpm_compadre_dfs_remesh.hpp"
#include "dfs/lpm_compadre_dfs_remesh_impl.hpp"

namespace Lpm {
namespace DFS {

template <> struct CompadreDfsRemesh<CubedSphereSeed>;
template <> struct CompadreDfsRemesh<IcosTriSphereSeed>;

}
}
