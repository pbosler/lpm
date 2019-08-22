#ifndef LPM_LAT_LON_MESH_HPP
#define LPM_LAT_LON_MESH_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;
    
    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}
};

struct LatLonMesh {
    ko::View<Real*[3]> pts;
    typename ko::View<Real*[3]>::HostMirror pts_host;
    Int nlon;
    Int nlat;
    Real dlam;
    Real dthe;
    
    LatLonMesh(const Int n_lat, const Int n_lon);
    
    KOKKOS_INLINE_FUNCTION
    Int lat_index(const Index pt_index) const {return pt_index/nlon;}
    
    KOKKOS_INLINE_FUNCTION
    Int lon_index(const Index pt_index) const {return pt_index%nlon;}
    
    ErrNorms compute_error(ko::View<const Real*> computed, ko::View<const Real*> exact) const;
    
    ErrNorms compute_error(ko::View<Real*> e, 
        ko::View<const Real*> computed, ko::View<Real*> exact) const;
};



}
#endif