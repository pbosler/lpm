#ifndef DFS_DOUBLING_HPP
#define DFS_DOUBLING_HPP

#include "dfs_config.hpp"

namespace SpherePoisson {

enum class GridType {Shifted=0, Unshifted=1};

struct GlideReflector {
  view_1d<Real> u;
  view_2d<Real> u_tilde;
  Int nrows;
  Int ncols;
  GridType grid_type;
  Int w;

  struct ILoopTag {};
  struct NLoopTag {};

  GlideReflector(view_1d<Real> uu, view_2d<Real> ut, const int mm, const int nn, const GridType gt) :
    u(uu), u_tilde(ut), nrows(mm), ncols(nn), grid_type(gt), w(ncols/2) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (ILoopTag, const Int i, const Int j) const {
    if (grid_type == GridType::Shifted) {
      u_tilde(nrows-1-i, j  ) = u(i*ncols + j+w);
      u_tilde(nrows-1-i, j+w) = u(i*ncols + j  );
    }
    else {
      u_tilde(nrows-2-i, j  ) = u((i+1)*ncols + j+w);
      u_tilde(nrows-2-i, j+w) = u((i+1)*ncols + j  );
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (NLoopTag, const Int i, const Int j) const {
    if (grid_type == GridType::Shifted) {
      u_tilde(nrows+i, j) = u(i*ncols + j);
    }
    else {
      u_tilde(nrows-1+i, j) = u(i*ncols + j);
    }
  }
};

// This code relates a function u defined on a latitude-longitude grid [0,pi] X [0, 2pi] to
// a  'doubled-up' version as u_tilde by applying the DFS method:
// References:
//Merilees, P. (1974). Numerical experiments with the pseudospectral method in spherical coordinates.
// Atmosphere.
// (1) Townsend, A., Wilber, H., and Wright, G. (2016), Computing with functions in
//spherical and polar geometries I. The sphere. SIAM Journal on Scientific Computing

void dfs_doubling(view_2d<Real> u_tilde, view_1d<Real> u, const GridType grid_type);

}
#endif
