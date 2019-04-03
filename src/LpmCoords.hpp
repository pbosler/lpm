#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmRealVector.hpp"

namespace Lpm {

template <int ndim, GeometryType geom> class Coords {
    public:
        typedef Kokkos::View<Real*[ndim]> view_type;
    
        Coords(const GeometryType& gm, const Index nmax) : _crds("crds", nmax), _geom(gm), _nmax(nmax), _n(0) {};
        
        inline GeometryType geometry() const {return _geom;}
        inline Index nMax() const {return _nmax;}
        inline Index n() const {return _n;}
        
        inline RealVec<ndim> getVec(const Index ind) const {
            RealVec<ndim> result;
            for (int j=0; j<ndim; ++j) {
                result(j) = _crds(ind, j);
            }
            return result;
        }
        
        void insert(const RealVec<ndim>& vec);
        
        Real distance(const Index inda, const Index indb) const {
            const RealVec<ndim> veca = this->getVec(inda);
            const RealVec<ndim> vecb = this->getVec(indb);
            return dist(veca, vecb);
        }
        
        RealVec<ndim> midpoint(const Index inda, const Index indb) const {
            const RealVec<ndim> veca = this->getVec(inda);
            const RealVec<ndim> vecb = this->getVec(indb);
            return midpoint(veca, vecb);
        }
        
    protected:
        view_type _crds;
        Index _nmax;
        Index _n;
        
        
};

Real Coords<3, SPHERICAL_SURFACE_GEOMETRY>::distance(const Index inda, const Index indb) const {
    const RealVec<ndim> veca = this->getVec(inda);
    const RealVec<ndim> vecb = this->getVec(indb);
    return sphereDist(veca, vecb);
}

Real Coords<3, SPHERICAL_SURFACE_GEOMETRY>::midpoint(const Index inda, const Index indb) const {
    const RealVec<ndim> veca = this->getVec(inda);
    const RealVec<ndim> vecb = this->getVec(indb);
    return sphereMidpoint(veca, vecb);
}

}
#endif
