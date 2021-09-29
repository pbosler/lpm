#include "tree/lpm_node_array_internal.hpp"
#include "tree/lpm_gpu_octree_kernels.hpp"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"

namespace Lpm {
namespace tree {

template <typename LowerLevelType>
void NodeArrayInternal::init_from_lower(LowerLevelType& lower) {
  if (level == 0) {
    build_root(lower.node_parents, lower.node_idx_count);
  }
  else {
    LPM_ASSERT(lower.node_keys.extent(0)%8 == 0);

    const auto nparents = lower.nnodes/8;
    Kokkos::View<key_type*> parent_keys("parent_keys", nparents);
    Kokkos::View<Index*> parent_idx0("parent_idx0", nparents);
    Kokkos::View<id_type*> parent_idxn("parent_idxn", nparents);

    Kokkos::parallel_for(nparents,
      ParentsFunctor(parent_keys, parent_idx0, parent_idxn,
        lower.node_keys, lower.node_idx_start, lower.node_idx_count,
        level, max_depth));

    Kokkos::View<id_type*> node_address("node_address", nparents);
    Kokkos::parallel_for("mark unique parents", nparents,
      MarkUniqueParentFunctor(node_address, parent_keys, level, max_depth));

    Kokkos::parallel_scan("scan unique parents", nparents,
      KOKKOS_LAMBDA (const Index i, id_type& update, const bool is_final) {
        const auto address_i = node_address(i);
        update += address_i;
        if (is_final) {
          node_address(i) = update;
        }
      });

    const auto nnview = Kokkos::subview(node_address, nparents-1);
    auto h_nnview = Kokkos::create_mirror_view(nnview);
    Kokkos::deep_copy(h_nnview, nnview);
    nnodes = h_nnview();

    node_keys = Kokkos::View<key_type*>("node_keys", nnodes);
    node_idx_start = Kokkos::View<Index*>("node_idx_start", nnodes);
    node_idx_count = Kokkos::View<id_type*>("node_idx_count", nnodes);
    node_parents = Kokkos::View<id_type*>("node_parents", nnodes);
    node_kids = Kokkos::View<id_type*[8]>("node_kids", nnodes);

    Kokkos::parallel_for(nparents,
      NodeArrayInternalFunctor(node_keys, node_idx_start, node_idx_count,
        node_parents, node_kids, lower.node_parents, node_address,
        parent_keys, parent_idx0, parent_idxn, lower.node_keys,
        level, max_depth));
  }
}

} // namespace tree
} // namespace Lpm
