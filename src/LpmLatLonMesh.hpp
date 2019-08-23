#ifndef LPM_LAT_LON_MESH_HPP
#define LPM_LAT_LON_MESH_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Parallel_Reduce.hpp"
#include <iostream>
#include <cfloat>

namespace Lpm {

struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;
    
    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}
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
    
    void writeLatLonMeshgrid(std::ostream& os) const;

};

template <typename Space>
struct LinfNormReducer {
    typedef LinfNormReducer reducer;
    typedef ko::Tuple<Real,2> value_type;
    typedef ko::View<Real[2], Space, ko::MemoryUnmanaged> result_view_type;
    
    private:
        value_type& val;
    
    public:
    
    KOKKOS_INLINE_FUNCTION
    LinfNormReducer(value_type& v) : val(v) {}
    
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
        if (dst[0] < src[0]) dst[0] = src[0];
        if (dst[1] < src[1]) dst[1] = src[1];
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        if (dst[0] < src[0]) dst[0] = src[0];
        if (dst[1] < src[1]) dst[1] = src[1];
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val[0] = ko::reduction_identity<Real>::max();
        val[1] = ko::reduction_identity<Real>::max();
    }
    
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return val;}
    
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return result_view_type(&val);}
    
    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const {return true;}
};

template <typename Space>
struct LPNormReducer {
    typedef LPNormReducer reducer;
    typedef ko::Tuple<Real,2> value_type;
    typedef ko::View<Real[2], Space, ko::MemoryUnmanaged> result_view_type;
    
    private:
        value_type& val;
    
    public:
    
    KOKKOS_INLINE_FUNCTION
    LPNormReducer(value_type& v) : val(v) {}
    
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
        dst += src;
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        dst += src;
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& v) const {
        v[0] = ko::reduction_identity<Real>::sum();
        v[1] = ko::reduction_identity<Real>::sum();
    }
    
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return val;}
    
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return result_view_type(&val);}
    
    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const {return true;}
};

}
#endif