#include "LpmLatLonMesh.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"
#include "LpmMatlabIO.hpp"
#include <cmath>
#include <sstream>

namespace Lpm {

std::string ErrNorms::infoString(const std::string& label, const int tab_level) const {
    std::ostringstream ss;
    std::string tabstr;
    for (int i=0; i<tab_level; ++i) tabstr += "\t";
    ss << tabstr << label << " ErrNorms: l1 = " << l1 << " l2 = " << l2 << " linf = " << linf << "\n";
    return ss.str();
}

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

void LatLonMesh::writeLatLonMeshgrid(std::ostream& os, const std::string& name) const {
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
    writeArrayMatlab(os, name+"lats", lats);
    writeArrayMatlab(os, name+"lons", lons);
    writeArrayMatlab(os, name+"weights", weights);
}

void LatLonMesh::writeLatLonScalar(std::ostream& os, const std::string& field_name,
    const ko::View<Real*,HostMem> vals_host) const {
    ko::View<Real**,HostMem> vals("meshvals", nlat, nlon);
    for (Int i=0; i<nlat; ++i) {
        for (Int j=0; j<nlon; ++j) {
            const Index k = i*nlon + j;
            vals(i,j) = vals_host(k);
        }
    }
    writeArrayMatlab(os, field_name, vals);
}

void LatLonMesh::computeScalarError(ko::View<Real*> error, const ko::View<const Real*> computed, const ko::View<const Real*> exact) const {
    ko::parallel_for(error.extent(0), KOKKOS_LAMBDA (const Int& i) {
        error(i) = std::abs(computed(i)-exact(i));
    });
}

ErrNorms LatLonMesh::scalarErrorNorms(const ko::View<const Real*> error, const ko::View<const Real*> exact) const {

    Real l1num;
    Real l1denom;
    Real l2num;
    Real l2denom;
    Real linfnum;
    Real linfdenom;

    ko::parallel_reduce(error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& e) {
        e += error(i)*wts(i);
    }, l1num);
    
    ko::parallel_reduce(error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& v) {
        v += std::abs(exact(i))*wts(i);
    }, l1denom);
    
    ko::parallel_reduce(error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& e) {
        e += square(error(i))*wts(i);
    }, l2num);
    
    ko::parallel_reduce(error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& v) {
        v += square(exact(i))*wts(i);
    }, l2denom);    
    
    ko::parallel_reduce("MaxReduce", error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& e) { 
        if (error(i) > e) e = error(i);
    }, ko::Max<Real>(linfnum));

    ko::parallel_reduce("MaxReduce", error.extent(0), KOKKOS_LAMBDA (const Int& i, Real& v) {
        if (exact(i) > v) v = exact(i);
    }, ko::Max<Real>(linfdenom));

    return ErrNorms(l1num/l1denom, std::sqrt(l2num/l2denom), linfnum/linfdenom);
}



/// ETI


}
