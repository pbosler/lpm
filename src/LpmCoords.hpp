#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"
#include "Kokkos_Random.hpp"

namespace Lpm {

typedef Kokkos::Random_XorShift64_Pool<> pool_type;
typedef typename pool_type::generator_type generator_type;

struct rand_crd_gen;
struct rand_crd_gen_sphere;

template <typename Geo> class Coords {
    public:
        typedef Kokkos::View<Real*[Geo::ndim]> crd_view_type;
        typedef Kokkos::View<Real[Geo::ndim]> vec_type;
    
        Coords(const Index nmax) : _crds("crds", nmax), _nmax(nmax), _n(0) {
            _host_crds = ko::create_mirror_view(_crds);
        };
        
        KOKKOS_INLINE_FUNCTION
        Index nMax() const { return _crds.extent(0);} //return _nmax;}
        
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n;}
        
        KOKKOS_INLINE_FUNCTION
        vec_type getVec(const Index ia) const {
            vec_type result;
            for (int j=0; j<Geo::ndim; ++j) 
                result(j) = _crds(ia, j);
            return result;
        }
        
        KOKKOS_INLINE_FUNCTION
        Real dist(const Index ia, const Index ib) const {
            vec_type a = this->getVec(ia);
            vec_type b = this->getVec(ib);
            return Geo::distance(a, b);
        }
        
        void updateDevice() {
            ko::deep_copy(_crds, _host_crds);
        }
        
        void updateHost() {
            ko::deep_copy(_host_crds, _crds);
        }
        
        template <typename CV>
        void insertHost(const CV v) {
        // never insert on device
            for (int i=0; i<Geo::ndim; ++i) 
                _host_crds(_n, i) = v[i];
            _n += 1;
        }
        
                
    protected:
        crd_view_type _crds;
        typename crd_view_type::HostMirror _host_crds;
        Index _nmax;
        Index _n;
};

struct rand_crd_gen { 
    Kokkos::View<Real**> vals;
    Real maxR;
    pool_type pool;
    
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

struct rand_crd_gen_sphere {
    Kokkos::View<Real*[3]> vals;
    Real radius;
    pool_type pool;
    
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


}
#endif
