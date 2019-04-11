#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/**
    All initialization is done on host.
*/
template <typename Geo> class Coords {
    public:
        typedef ko::View<Real*[Geo::ndim]> crd_view_type;
        typedef ko::View<Real[Geo::ndim]> vec_type;
#ifdef HAVE_CUDA
        typedef ko::View<Real*, ko::LayoutStride,
            typename crd_view_type::device_type, ko::MemoryTraits<ko::Unmanaged>> slice_type;
        typedef ko::View<const Real*, ko::LayoutStride,
            typename crd_view_type::device_type, ko::MemoryTraits<ko::Unmanaged>> const_slice_type;
        typedef ko::View<Real*, ko::LayoutStride, typename crd_view_type::host_mirror_space, 
            ko::MemoryTraits<ko::Unmanaged>> host_slice_type;
        typedef ko::View<const Real*, ko::LayoutStride, typename crd_view_type::host_mirror_space,
            ko::MemoryTraits<ko::Unmanaged>> const_host_slice;
#else
        typedef typename crd_view_type::value_type* slice_type;
        typedef typename crd_view_type::const_value_type* const_slice_type;
        typedef slice_type host_slice_type;
        typedef const_slice_type const_host_slice;
#endif        
    
        Coords(const Index nmax) : _crds("crds", nmax), _nmax(nmax), _n("n") {
            _host_crds = ko::create_mirror_view(_crds);
            _nh = ko::create_mirror_view(_n);
            _nh(0) = 0;
        };
        
        /// Host function
        Index nMax() const { return _crds.extent(0);} //return _nmax;}
        
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n(0);}
        
        /// Host function
        Index nh() const {return _nh(0);}
        
        KOKKOS_INLINE_FUNCTION
        slice_type getSlice(const Index ia) {return slice(_crds, ia);}
        
        KOKKOS_INLINE_FUNCTION
        const_slice_type getConstSlice(const Index ia) const {return const_slice(_crds, ia);}
        
        host_slice_type getSliceHost(const Index ia) {return slice(_host_crds, ia);}
        
        const_host_slice getConstSliceHost(const Index ia) const {return const_slice(_host_crds, ia);}

        void updateDevice() {
            ko::deep_copy(_crds, _host_crds);
            ko::deep_copy(_n, _nh);
        }
        
        void updateHost() {
            ko::deep_copy(_host_crds, _crds);
            ko::deep_copy(_nh, _n);
        }
        
        /// Host function
        template <typename CV> void insertHost(const CV v) {
            for (int i=0; i<Geo::ndim; ++i) {
                _host_crds(_nh(0), i) = v[i];
            }
            _nh(0) += 1;
        }
        
        /// Host function
        void printcrds(const std::string& label) const;
        
        /// Host function
        void initRandom(const Real max_range=1.0, const Int ss=0);
                
    protected:
        crd_view_type _crds;
        typename crd_view_type::HostMirror _host_crds;
        Index _nmax;
        Kokkos::View<Index> _n;
        Kokkos::View<Index>::HostMirror _nh;
};


}
#endif
