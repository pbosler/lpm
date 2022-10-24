#include "lpm_lat_lon_pts.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

LatLonPts::LatLonPts(const int n_lat, const int n_lon) :
  nlat(n_lat), nlon(n_lon),
  dlambda(2*constants::PI/n_lon),
  dtheta(constants::PI/(n_lat-1)) {
    const Index npts = nlat*nlon;
    pts = Kokkos::View<Real*[3]>("lat_lon_pts", npts);
    wts = Kokkos::View<Real*>("lat_lon_wts", npts);

    h_pts = Kokkos::create_mirror_view(pts);
    h_wts = Kokkos::create_mirror_view(wts);
    const Real sin_half_dtheta = sin(0.5*dtheta);
    for (int i=0; i<nlat; ++i) {
      const Real lat = -0.5*constants::PI + i*dtheta;
      const Real coslat = cos(lat);
      const Real z = sin(lat);
      const Real w = 2*dlambda*sin_half_dtheta*coslat;
      for (int j=0; j<nlon; ++j) {
        const Real lon = j*dlambda;
        const Real x = cos(lon)*coslat;
        const Real y = sin(lon)*coslat;
        h_pts(i*nlon + j, 0) = x;
        h_pts(i*nlon + j, 1) = y;
        h_pts(i*nlon + j, 2) = z;
        h_wts(i*nlon + j) = w;
      }
    }
    Kokkos::deep_copy(pts, h_pts);
    Kokkos::deep_copy(wts, h_wts);
  }


void LatLonPts::write_grid_matlab(std::ostream& os, const std::string& name) const {
  Kokkos::View<Real**,HostMem> lats("lats", nlat, nlon);
  Kokkos::View<Real**,HostMem> lons("lons", nlat, nlon);
  Kokkos::View<Real**,HostMem> weights("weights", nlat, nlon);
  for (int i=0; i<nlat; ++i) {
    for (int j=0; j<nlon; ++j) {
      const Index k = i*nlon + j;
      lats(i,j) = SphereGeometry::latitude(Kokkos::subview(h_pts, k, Kokkos::ALL));
      lons(i,j) = SphereGeometry::longitude(Kokkos::subview(h_pts, k, Kokkos::ALL));
      weights(i,j) = h_wts(k);
    }
  }
  write_array_matlab(os, name+"_lats", lats);
  write_array_matlab(os, name+"_lons", lons);
  write_array_matlab(os, name+"_wts", weights);
}



}
