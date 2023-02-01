#ifndef LPM_LAT_LON_PTS_HPP
#define LPM_LAT_LON_PTS_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "util/lpm_matlab_io.hpp"

namespace Lpm {

struct LatLonPts {
  Kokkos::View<Real*[3]> pts;
  Kokkos::View<Real*> wts;
  typename Kokkos::View<Real*[3]>::HostMirror h_pts;
  typename Kokkos::View<Real*>::HostMirror h_wts;

  Int nlon;
  Int nlat;
  Real dlambda;
  Real dtheta;

  explicit LatLonPts(const int n_unif, const Real r=1);

  LatLonPts(const int n_lat, const int n_lon);

  KOKKOS_INLINE_FUNCTION
  Int n() const {return nlat*nlon;}

  KOKKOS_INLINE_FUNCTION
  Int lat_idx(const Index pt_idx) const {return pt_idx/nlon;}

  KOKKOS_INLINE_FUNCTION
  Int lon_idx(const Index pt_idx) const {return pt_idx%nlon;}

  void write_grid_matlab(std::ostream& os, const std::string& name) const;

  template <typename HVT>
  void write_scalar_field_matlab(std::ostream& os, const std::string& name,
    const HVT vals) const {
    Kokkos::View<Real**, HostMemory> grid_vals("grid_vals", nlat, nlon);
    for (Int i=0; i<nlat; ++i) {
      for (Int j=0; j<nlon; ++j) {
        const Index k = i*nlon + j;
        grid_vals(i,j) = vals(k);
      }
    }
    write_array_matlab(os, name, grid_vals);
  }

  KOKKOS_INLINE_FUNCTION
  Int nx() const {return nlon;}

  KOKKOS_INLINE_FUNCTION
  Int ny() const {return nlat;}

  KOKKOS_INLINE_FUNCTION
  Index size() const {return pts.extent(0);}

  void init();
};

} // namespace Lpm

#endif
