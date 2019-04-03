#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmRealVector.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"
#include "Kokkos_Random.hpp"

namespace Lpm {

typedef Kokkos::Random_XorShift64_Pool<> pool_type;
typedef typename pool_type::generator_type generator_type;

struct rand_crd_gen;
struct rand_crd_gen_sphere;

template <int ndim, GeometryType geom> class Coords {
    public:
        typedef Kokkos::View<Real*[ndim]> view_type;
    
        Coords(const Index nmax) : _crds("crds", nmax), _nmax(nmax), _n(0) {};
        
        inline GeometryType geometry() const {return geom;}
        
        KOKKOS_INLINE_FUNCTION
        Index nMax() const { return _crds.extent(0);} //return _nmax;}
        
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n;}
        
        KOKKOS_INLINE_FUNCTION
        RealVec<ndim> getVec(const Index ind) const {
            RealVec<ndim> result;
            for (int j=0; j<ndim; ++j) {
                result[j] = _crds(ind, j);
            }
            return result;
        }
        
        void insert(const RealVec<ndim>& vec);
        
        KOKKOS_FUNCTION
        Real distance(const Index inda, const Index indb) const {
            const RealVec<ndim> veca = this->getVec(inda);
            const RealVec<ndim> vecb = this->getVec(indb);
            return veca.dist(vecb);
        }
        
        KOKKOS_FUNCTION
        RealVec<ndim> midpoint(const Index inda, const Index indb) const {
            const RealVec<ndim> veca = this->getVec(inda);
            const RealVec<ndim> vecb = this->getVec(indb);
            return veca.midpoint(vecb);
        }
        
        KOKKOS_FUNCTION
        RealVec<ndim> barycenter(IndexArray inds) const {
            view_type pts("bc_pts", inds.extent(0));
            for (int i=0; i<inds.extent(0); ++i) {
                for (int j=0; j<ndim; ++j) {
                    pts(i,j) = _crds(inds(i), j);
                }
            }
            return Lpm::barycenter<ndim>(pts);
        }
        
        KOKKOS_FUNCTION
        Real triArea(const Index inda, const Index indb, const Index indc) const {
            const RealVec<ndim> veca = this->getVec(inda);
            const RealVec<ndim> vecb = this->getVec(indb);
            const RealVec<ndim> vecc = this->getVec(indc);
            return Lpm::triArea(veca, vecb, vecc);
        }
        
        KOKKOS_FUNCTION 
        void initRandom(const Real max_range, const Int ss=0) {
            Kokkos::parallel_for(this->_nmax, rand_crd_gen(_crds, max_range, ss));
            _n = _nmax;
        }
        
    protected:
        view_type _crds;
        Index _nmax;
        Index _n;
};

struct rand_crd_gen { 
    Kokkos::View<Real**> vals;
    Real maxR;
    pool_type pool;
    
    KOKKOS_FUNCTION
    rand_crd_gen(Kokkos::View<Real**> v, const Real max_range, const Int seed_start) : vals(v), maxR(max_range), pool(seed_start) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (Index i) const {
        generator_type rgen = pool.get_state();
        for (int j=0; j<vals.extent(1); ++j) {
            vals(i,j) = rgen.drand(maxR);
        }
        pool.free_state(rgen);
    }
};


/**
    
    Specializations for the sphere

*/
template <> KOKKOS_INLINE_FUNCTION
Real Coords<3,SPHERICAL_SURFACE_GEOMETRY>::distance(const Index inda, const Index indb) const {
    const RealVec<3> veca = this->getVec(inda);
    const RealVec<3> vecb = this->getVec(indb);
    return veca.sphereDist(vecb);
}

template <> KOKKOS_INLINE_FUNCTION
RealVec<3> Coords<3,SPHERICAL_SURFACE_GEOMETRY>::midpoint(const Index inda, const Index indb) const {
    const RealVec<3> veca = this->getVec(inda);
    const RealVec<3> vecb = this->getVec(indb);
    return veca.sphereMidpoint(vecb);
}

template <> KOKKOS_FUNCTION
RealVec<3> Coords<3,SPHERICAL_SURFACE_GEOMETRY>::barycenter(IndexArray inds) const {
    view_type pts("bc_pts", inds.extent(0));
    for (int i=0; i<inds.extent(0); ++i) {
        for (int j=0; j<3; ++j) {
            pts(i,j) = _crds(inds(i), j);
        }
    }
    return sphereBarycenter(pts);
}

template <> KOKKOS_FUNCTION
Real Coords<3,SPHERICAL_SURFACE_GEOMETRY>::triArea(const Index inda, const Index indb, const Index indc) const {
    const RealVec<3> veca = this->getVec(inda);
    const RealVec<3> vecb = this->getVec(indb);
    const RealVec<3> vecc = this->getVec(indc);
    return sphereTriArea(veca, vecb, vecc);
}

struct rand_crd_gen_sphere {
    Kokkos::View<Real*[3]> vals;
    Real radius;
    pool_type pool;
    
    KOKKOS_FUNCTION
    rand_crd_gen_sphere(Kokkos::View<Real*[3]> v, const Real r, const Int seed_start) : vals(v), radius(r), pool(seed_start) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (Index i) const {
        generator_type rgen = pool.get_state();
        Real u = rgen.drand(-1.0, 1.0);
        Real v = rgen.drand(-1.0, 1.0);
        while (u*u + v*v > 1.0) {
            u = rgen.drand(-1.0, 1.0);
            v = rgen.drand(-1.0, 1.0);
        }
        vals(i,0) = radius * (2.0 * u * std::sqrt(1 - u*u - v*v));
        vals(i,1) = radius * (2.0 * v * std::sqrt(1 - u*u - v*v));
        vals(i,2) = radius * (1.0 - 2.0*(u*u + v*v));
        pool.free_state(rgen);
    }
};


template <> KOKKOS_FUNCTION
void Coords<3, SPHERICAL_SURFACE_GEOMETRY>::initRandom(const Real r, const Int ss) {
    Kokkos::parallel_for(this->_nmax, rand_crd_gen_sphere(this->_crds, r, ss));
    this->_n = this->_nmax;
}



}
#endif
