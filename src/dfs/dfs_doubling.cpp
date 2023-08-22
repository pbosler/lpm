#include "dfs_doubling.hpp"

namespace SpherePoisson {

void dfs_doubling(view_2d<Real> u_tilde, view_1d<Real> u,  const GridType grid_type) {
  
  int ncols = u_tilde.extent(0);
  const int w = ncols/2;
  const bool is_shifted = grid_type == GridType::Shifted;
  int nrows = is_shifted ? (u_tilde.extent(0)/2):u_tilde.extent(0)/2+1;	
  
  const auto md_policy_i = Kokkos::MDRangePolicy<GlideReflector::ILoopTag,
    Kokkos::Rank<2>>( {0, 0}, {(is_shifted ? nrows : nrows-1), w});

  Kokkos::parallel_for("dfs_doubling_i_loop", md_policy_i,
     GlideReflector(u, u_tilde, nrows, ncols, grid_type));

  const auto md_policy_n = Kokkos::MDRangePolicy<GlideReflector::NLoopTag,
    Kokkos::Rank<2>>( {0, 0}, {(is_shifted ? nrows: nrows-1), ncols});

  Kokkos::parallel_for("dfs_doubling_n_loop", md_policy_n,
    GlideReflector(u, u_tilde, nrows, ncols, grid_type));

}

}
