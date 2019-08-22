#include "LpmLatLonMesh.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {

LatLonMesh::LatLonMesh(const Int n_lat, const Int n_lon) : nlat(n_lat), nlon(n_lon), 
    dlam(2*PI/n_lon), dthe(PI/(n_lat-1)) {
    
    const Index npts = nlat*nlon;
    pts = ko::View<Real*[3]>("ll_mesh", npts);
    pts_host = ko::create_mirror_view(pts);
    
    ko::parallel_for(nlat, KOKKOS_LAMBDA (const Index i) {
        const Index start_ind = i*nlon;
        const Real lat = -0.5*PI + i*dthe;
        const Real coslat = std::cos(lat);
        const Real z = std::sin(lat);
        for (Int j=0; j<nlon; ++j) {
            const Index start_ind = i*nlon;
            const Real lon = j*dlam;
            const Real x = std::cos(lon)*coslat;
            const Real y = std::sin(lon)*coslat;
            pts(start_ind+j,0) = x;
            pts(start_ind+j,1) = y;
            pts(start_ind+j,2) = z;
        }
    });
    
    ko::deep_copy(pts_host, pts);
}

ErrNorms LatLonMesh::compute_error(ko::View<Real*> e, 
    ko::View<const Real*> computed, ko::View<const Real*> exact) const {

    ko::parallel_for(e.extent(0), KOKKOS_LAMBDA (const Index i) {
        e(i) = abs(computed(i) - exact(i));
    });
    
    Real linf;
    ko::parallel_reduce("MaxReduce", computed.extent(0), KOKKOS_LAMBDA (const Index i, Real& err) {
        if (e(i) > err) err = e(i);
    }, ko::Max<Real>(linf));
    
    ko::Tuple<Real,4> epieces;
    ko::parallel_reduce(computed.extent(0), KOKKOS_LAMBDA (const Index i, ko::Tuple<Real,4>& e12) {
        const Real coslat = std::cos(-0.5*PI + lat_index(i)*dthe);
        e12[0] += e(i)*coslat;
        e12[1] += exact(i)*coslat;
        e12[2] += e(i)*e(i)*coslat;
        e12[3] += exact(i)*exact(i)*coslat;
    }, epieces);
    return ErrNorms(epieces[0]/epieces[1], std::sqrt(epieces[2]/epieces[3]), linf);
}

}
