#ifndef LPM_REFINEMENT_HPP
#define LPM_REFINEMENT_HPP

#include "LpmConfig.h"
#include "mesh/lpm_polymesh2d.hpp"

namespace Lpm {

/**

  A refinement iteration is:
  1. For each flag, flag any faces that meet criteria
  2. Count flag statistics
  3. divide flagged faces
 ---- not this class's responsibility ---
  4. set data on new vertices/faces
*/

template <typename SeedType> struct Refinement {
  Kokkos::View<bool*> flags;
  std::vector<Index> count;
  PolyMesh2d<SeedType>& mesh;

  Refinement(PolyMesh2d<SeedType>& mesh) :
    flags("refinement_flags", mesh.faces.area.extent(0)),
    mesh(mesh) {}

  template <typename FlagType>
  void iterate(const Index start_idx, const Index end_idx, FlagType& flagger) {
    Kokkos::deep_copy(flags, false);
    count = std::vector<Index>(1,0);
    const auto policy = Kokkos::RangePolicy<>(start_idx, end_idx);
    Kokkos::parallel_for(policy, flagger);

    const auto ff = flags;
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[0]);
  }

  template <typename FlagType1, typename FlagType2>
  void iterate(const Index start_idx, const Index end_idx, FlagType1& flag1, FlagType2& flag2) {
    Kokkos::deep_copy(flags, false);
    count = std::vector<Index>(2,0);
    const auto policy = Kokkos::RangePolicy<>(start_idx, end_idx);

    Kokkos::parallel_for(policy, flag1);
    const auto ff = flags;
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[0]);

    Kokkos::parallel_for(policy, flag2);
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[1]);
    count[1] -= count[0];
  }

  template <typename FlagType1, typename FlagType2, typename FlagType3>
  void iterate(const Index start_idx, const Index end_idx, FlagType1& flag1, FlagType2& flag2, FlagType3& flag3) {
    Kokkos::deep_copy(flags, false);
    count = std::vector<Index>(3,0);
    const auto policy = Kokkos::RangePolicy<>(start_idx, end_idx);

    Kokkos::parallel_for(policy, flag1);
    const auto ff = flags;
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[0]);

    Kokkos::parallel_for(policy, flag2);
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[1]);
    count[1] -= count[0];

    Kokkos::parallel_for(policy, flag3);
    Kokkos::parallel_reduce(policy,
      KOKKOS_LAMBDA (const Index i, Index& ct) {
        ct += (ff(i) ? 1 : 0);
      }, count[2]);
    count[2] -= (count[0] + count[1]);
  }

};

} // namespace Lpm

#endif
