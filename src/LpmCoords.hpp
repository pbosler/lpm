#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/**
    All initialization is done on host.
*/
template <typename Geo> class Coords {
    public:
        typedef typename Geo::crd_view_type crd_view_type;
              
        crd_view_type crds;
        
        Coords(const Index nmax) : crds("crds", nmax), _nmax(nmax), _n("n") {
            _hostcrds = ko::create_mirror_view(crds);
            _nh = ko::create_mirror_view(_n);
            _nh(0) = 0;
        };
        
        /// Host function
        Index nMax() const { return crds.extent(0);} //return _nmax;}
        
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n(0);}
        
        void updateDevice() const {
            ko::deep_copy(crds, _hostcrds);
            ko::deep_copy(_n, _nh);
        }
        
        void updateHost() const {
            ko::deep_copy(_hostcrds, crds);
            ko::deep_copy(_nh, _n);
        }
        
/*/////  HOST FUNCTIONS ONLY BELOW THIS LINE         
    
        todo: make them protected, not public    
        
*/        
        /// Host function
        Index nh() const {return _nh(0);}
        
        inline Real getCrdComponentHost(const Index ind, const Int dim) const {return _hostcrds(ind, dim);}
        
        /// Host function
        template <typename CV> 
        void insertHost(const CV v) {
            LPM_THROW_IF(_nmax < _nh(0) + 1, "Coords::insert error: not enough memory.");
            for (int i=0; i<Geo::ndim; ++i) {
                _hostcrds(_nh(0), i) = v[i];
            }
            _nh(0) += 1;
        }

        /// Host function
        void relocateHost(const Index ind, const ko::View<Real[Geo::ndim], Host> v) {
            LPM_THROW_IF(ind >= _nh(0), "Coords::relocateHost error: index out of range.");
            for (int i=0; i<Geo::ndim; ++i) {
                _hostcrds(ind, i) = v(i);
            }
        }
        
        /// Host function
        std::string infoString(const std::string& label) const;
        
        /// Host function
        void initRandom(const Real max_range=1.0, const Int ss=0);
        
        /// Host function
        template <typename SeedType>
        void initBoundaryCrdsFromSeed(const MeshSeed<SeedType>& seed);
        
        /// Host function
        template <typename SeedType>
        void initInteriorCrdsFromSeed(const MeshSeed<SeedType>& seed);
        
        /// Host function
        void writeMatlab(std::ostream& os, const std::string& name) const;
        
        /// Host function
        ko::View<Real[Geo::ndim],Host> crdVecHost(const Index ind) {return ko::subview(_hostcrds, ind, ko::ALL());}
        
        ko::View<const Real[Geo::ndim],Host> crdVecHostConst(const Index ind) const {
            return ko::subview(_hostcrds, ind, ko::ALL());
        }
    
        
        
    protected:
        typename crd_view_type::HostMirror _hostcrds;
        Index _nmax;
        n_view_type _n;
        typename n_view_type::HostMirror _nh;
        
};


}
#endif
