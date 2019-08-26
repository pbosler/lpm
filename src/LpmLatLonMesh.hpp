#ifndef LPM_LAT_LON_MESH_HPP
#define LPM_LAT_LON_MESH_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <string>

namespace Lpm {

struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;
    
    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}
    
    std::string infoString(const std::string& label="", const int tab_level=0) const;
};

struct LatLonMesh {
    ko::View<Real*[3]> pts;
    ko::View<Real*> wts;
    typename ko::View<Real*[3]>::HostMirror pts_host;
    typename ko::View<Real*>::HostMirror wts_host;
    Int nlon;
    Int nlat;
    Real dlam;
    Real dthe;
    
    LatLonMesh(const Int n_lat, const Int n_lon);
    
    KOKKOS_INLINE_FUNCTION
    Int lat_index(const Index pt_index) const {return pt_index/nlon;}
    
    KOKKOS_INLINE_FUNCTION
    Int lon_index(const Index pt_index) const {return pt_index%nlon;}
    
    void writeLatLonMeshgrid(std::ostream& os, const std::string& name="") const;
    
    void writeLatLonScalar(std::ostream& os, const std::string& field_name, 
        const ko::View<Real*,HostMem> vals_host) const;
    
    void computeScalarError(ko::View<Real*> error, const ko::View<const Real*> computed, const ko::View<const Real*> exact) const;
    
    ErrNorms scalarErrorNorms(const ko::View<const Real*> error, const ko::View<const Real*> exact) const;

};

}
#endif