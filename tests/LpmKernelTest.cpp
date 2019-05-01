#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmGeometry.hpp"
#include <iostream>
#include <sstream>

using namespace Lpm;

typedef ko::View<Real*[3],Dev> view_type;
typedef ko::TeamPolicy<>::member_type member_type;

void init(view_type v1, view_type v2, view_type v3);

namespace Kokkos {
    template <int ndim> struct RVec {
        Lpm::Real v[ndim];
        KOKKOS_FORCEINLINE_FUNCTION RVec () {for (int i=0; i<ndim; ++i) v[i] = 0;}
        KOKKOS_FORCEINLINE_FUNCTION RVec operator += (const RVec<ndim>& o) const {
            RVec<ndim> result;
            for (int i=0; i<ndim; ++i) {
                result.v[i] = v[i] + o.v[i];
            }
            return result;
        }
        KOKKOS_FORCEINLINE_FUNCTION Real& operator [] (const Int i) {return v[i];}
        KOKKOS_FORCEINLINE_FUNCTION const Real& operator [] (const Int i) const {return v[i];}
    };
    
    template <> template <int ndim> struct reduction_identity<RVec<ndim>> {
        KOKKOS_FORCEINLINE_FUNCTION static RVec<ndim> sum() {return RVec<ndim>();}
    };
}

struct VecReducer {
    typedef ko::RVec<3> value_type;
    typedef Index size_type;
    
    view_type x;
    view_type xx;
    Index i;
    
    KOKKOS_INLINE_FUNCTION
    VecReducer(view_type srcx, view_type tgtx, const Index ii) : x(srcx), xx(tgtx), i(ii) {}
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& v) const {
        for (int j=0; j<3; ++j) {
            v[j] = 0;
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        for (int j=0; j<3; ++j) {
            dst.v[j] += src.v[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index &j, value_type& v) const {
        auto xs = ko::subview(x, i, ko::ALL());
        auto xt = ko::subview(xx,j, ko::ALL());
        ko::RVec<3> cp;
        SphereGeometry::cross(cp.v, xs, xt);
        for (int k=0; k<3; ++k) {
            v[k] += cp[k];
        }
    }
};

struct VecComputer {
    view_type x;
    view_type xx;
    view_type u;
    Index src_size;
    
    VecComputer(view_type srcx, view_type tgtx, view_type tgtv, const Index ns) : x(srcx), xx(tgtx), u(tgtv), src_size(ns) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        ko::RVec<3> vec;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr, src_size), VecReducer(x, xx, i), vec);
        for (int j=0; j<3; ++j) {
            u(i,j) = vec[j];
        }
    }
};


int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const Int nsrc = 10;
    const Int ntgt = 8;
    view_type srclocs("src_x", nsrc);
    view_type tgtlocs("tgt_x", ntgt);
    view_type tgtvel("tgt_u", ntgt);
    
    init(srclocs, tgtlocs, tgtvel);
    
    auto policy = ko::TeamPolicy<>(ntgt, ko::AUTO());
    
    ko::parallel_for(policy, VecComputer(srclocs, tgtlocs, tgtvel, nsrc));
    
    auto hvel = ko::create_mirror_view(tgtvel);
    ko::deep_copy(hvel, tgtvel);
    for (Int i=0; i<ntgt; ++i) {
        std::cout << "(" << hvel(i,0) << ", " << hvel(i,1) << ", " << hvel(i,2) << ")" << std::endl;
    }
}
ko::finalize();
return 0;
}

void init(view_type v1, view_type v2, view_type v3) {
    auto h1 = ko::create_mirror_view(v1);
    auto h2 = ko::create_mirror_view(v2);
    auto h3 = ko::create_mirror_view(v3);
    
    for (int i=0; i<v1.extent(0); ++i) {
        std::cout << "h1(i,:) = (";
        for (int j=0; j<3; ++j) {
            h1(i,j) = (i%2==0 ? 1 : -1)+i*j;
            std::cout << h1(i,j) << " ";
        }
        std::cout << ")" << std::endl;
    }

    for (int i=0; i<v2.extent(0); ++i) {
        std::cout << "h2(i,:) = (";
        for (int j=0; j<3; ++j) {
            h2(i,j) = (i%2==0 ? 2 : -1) + i -j;
            std::cout << h2(i,j) << " ";
        }
        std::cout << ")" << std::endl;
    }    
    
    for (int i=0; i<v3.extent(0); ++i) {
        for (int j=0; j<3; ++j) {
            h3(i,j) = 0;
        }
    }
    ko::deep_copy(v1, h1);
    ko::deep_copy(v2, h2);
    ko::deep_copy(v3, h3);
}
