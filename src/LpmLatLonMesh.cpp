#include "LpmLatLonMesh.hpp"
#include "Kokkos_Core.hpp"
#include "LpmMatlabIO.hpp"
#include <cmath>

namespace Lpm {

LatLonMesh::LatLonMesh(const Int n_lat, const Int n_lon) : nlat(n_lat), nlon(n_lon), 
    dlam(2*PI/n_lon), dthe(PI/(n_lat-1)) {
    
    const Index npts = nlat*nlon;
    pts = ko::View<Real*[3]>("ll_mesh", npts);
    pts_host = ko::create_mirror_view(pts);
    wts = ko::View<Real*>("weights", npts);
    wts_host = ko::create_mirror_view(wts);
    
    ko::parallel_for(nlat, KOKKOS_LAMBDA (const Index i) {
        const Index start_ind = i*nlon;
        const Real lat = -0.5*PI + i*dthe;
        const Real coslat = std::cos(lat);
        const Real z = std::sin(lat);
        const Real w = 2*dlam*std::sin(0.5*dthe)*coslat;
        for (Int j=0; j<nlon; ++j) {
            const Index start_ind = i*nlon;
            const Real lon = j*dlam;
            const Real x = std::cos(lon)*coslat;
            const Real y = std::sin(lon)*coslat;
            pts(start_ind+j,0) = x;
            pts(start_ind+j,1) = y;
            pts(start_ind+j,2) = z;
            wts(start_ind+j) = w;
        }
    });
    
    ko::deep_copy(pts_host, pts);
    ko::deep_copy(wts_host, wts);
}

void LatLonMesh::writeLatLonMeshgrid(std::ostream& os) const {
    ko::View<Real**,HostMem> lats("lats", nlat, nlon);
    ko::View<Real**,HostMem> lons("lons", nlat, nlon);
    ko::View<Real**,HostMem> weights("weights", nlat, nlon);
    for (Int i=0; i<nlat; ++i) {
        for (Int j=0; j<nlon; ++j) {
            const Index k = i*nlon + j;
            lats(i,j) = SphereGeometry::latitude(ko::subview(pts_host, k, ko::ALL()));
            lons(i,j) = SphereGeometry::longitude(ko::subview(pts_host, k, ko::ALL()));
            weights(i,j) = wts_host(k);
        }
    }
    writeArrayMatlab(os, "lats", lats);
    writeArrayMatlab(os, "lons", lons);
    writeArrayMatlab(os, "weights", weights);
}


// ErrNorms LatLonMesh::compute_error(ko::View<Real*> e, 
//     ko::View<const Real*> computed, ko::View<const Real*> exact) const {
// 
//     ko::parallel_for(e.extent(0), KOKKOS_LAMBDA (const Index i) {
//         e(i) = abs(computed(i) - exact(i));
//     });
//     
//     ko::Tuple<Real,2> infpieces;
//     ko::parallel_reduce("MaxReduce", computed.extent(0), KOKKOS_LAMBDA (const Index i, ko::Tuple<Real,2>& err) {
//         if (e(i) > err[0]) err[0] = e(i);
//         if (abs(exact(i)) > err[1]) err[1] = abs(exact(i));
//     }, ko::Max<Real>(linf));
//     
//     ko::Tuple<Real,4> epieces;
//     ko::parallel_reduce(computed.extent(0), KOKKOS_LAMBDA (const Index i, ko::Tuple<Real,4>& e12) {
//         const Real coslat = std::cos(-0.5*PI + lat_index(i)*dthe);
//         e12[0] += e(i)*coslat;
//         e12[1] += abs(exact(i))*coslat;
//         e12[2] += e(i)*e(i)*coslat;
//         e12[3] += exact(i)*exact(i)*coslat;
//     }, epieces);
//     return ErrNorms(epieces[0]/epieces[1], std::sqrt(epieces[2]/epieces[3]), linf);
// }

}
