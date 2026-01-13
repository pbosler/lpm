#ifndef LPM_PLANAR_GRID_HPP
#define LPM_PLANAR_GRID_HPP

#include "LpmConfig.h"

namespace Lpm {

struct PlanarGrid {
  Kokkos::View<Real* [2]> pts;
  Kokkos::View<Real*> wts;
  typename Kokkos::View<Real* [2]>::HostMirror h_pts;
  typename Kokkos::View<Real*>::HostMirror h_wts;
  Real xmin;
  Real xmax;
  Real ymin;
  Real ymax;

  PlanarGrid(const int n, const Real maxr)
      : pts("pts", n * n),
        wts("wts", n * n),
        xmin(-maxr),
        xmax(maxr),
        ymin(-maxr),
        ymax(maxr) {
    h_pts = Kokkos::create_mirror_view(pts);
    h_wts = Kokkos::create_mirror_view(wts);

    const Real dx = (xmax - xmin) / (n - 1);
    const Real dy = (ymax - ymin) / (n - 1);
    for (int i = 0; i < n * n; ++i) {
      const Int ii = i / n;
      const Int jj = i % n;
      h_pts(i, 0)  = xmin + ii * dx;
      h_pts(i, 1)  = ymin + jj * dy;
      h_wts(i)     = dx * dy;
    }
    Kokkos::deep_copy(pts, h_pts);
    Kokkos::deep_copy(wts, h_wts);
  }

  inline int size() const { return pts.extent(0); }

  inline int n() const { return int(sqrt(pts.extent(0))); }

  inline int nx() const { return n(); }

  inline int ny() const { return n(); }
};

}  // namespace Lpm

#endif
