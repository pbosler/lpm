#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmGeometry.hpp"
#include <iostream>
#include <sstream>
#include <cstdio>
using namespace Lpm;

typedef ko::View<Real*[3],Dev> view_type;
typedef ko::TeamPolicy<>::member_type member_type;

void init(view_type v1, view_type v2, view_type v3);

namespace Kokkos {
    template <int ndim> struct RVec {
        Lpm::Real v[ndim];
        KOKKOS_FORCEINLINE_FUNCTION RVec () {for (int i=0; i<ndim; ++i) v[i] = 0;}
        KOKKOS_FORCEINLINE_FUNCTION RVec operator += (const RVec<ndim>& o) {
            for (int i=0; i<ndim; ++i) {
                v[i] += o.v[i];
            }
            return *this;
        }
        KOKKOS_FORCEINLINE_FUNCTION Real& operator [] (const Int i) {return v[i];}
        KOKKOS_FORCEINLINE_FUNCTION const Real& operator [] (const Int i) const {return v[i];}
    };
    
    template <> struct reduction_identity<RVec<3>> {
        KOKKOS_FORCEINLINE_FUNCTION static RVec<3> sum() {return RVec<3>();}
    };
}

template <typename CV> KOKKOS_INLINE_FUNCTION
static void cross(ko::RVec<3>& c, const CV a, const CV b) {
        c.v[0] = a[1]*b[2] - a[2]*b[1];
        c.v[1] = a[2]*b[0] - a[0]*b[2];
        c.v[2] = a[0]*b[1] - a[1]*b[0];
}

template <typename CV> KOKKOS_INLINE_FUNCTION
static ko::RVec<3> cross(const CV a, const CV b) {
    ko::RVec<3> c;
    c.v[0] = a[1]*b[2] - a[2]*b[1];
    c.v[1] = a[2]*b[0] - a[0]*b[2];
    c.v[2] = a[0]*b[1] - a[1]*b[0];
    return c;
}

struct VecReducer {
    typedef ko::RVec<3> value_type;
    typedef Index size_type;
    
    view_type x;
    view_type xx;
    Index i;
    
    KOKKOS_INLINE_FUNCTION
    VecReducer(view_type srcx, view_type tgtx, const Index ii) : x(tgtx), xx(srcx), i(ii) {}
    
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
//         printf("\ti=%d, j=%d\n", i,j);
        auto xs = ko::subview(xx, j, ko::ALL());
        auto xt = ko::subview(x, i, ko::ALL());
//         printf("xt = %f, %f, %f\n", xt(0), xt(1), xt(2));
        const ko::RVec<3> cp = cross(xs, xt);
        for (int k=0; k<3; ++k) {
            v.v[k] += cp.v[k];
        }
//         printf("cp3 = %f\n", cp.v[2]);
    }
};

struct VecComputer {
    view_type x;
    view_type xx;
    view_type u;
    Index src_size;
    
    VecComputer(view_type srcx, view_type tgtx, view_type tgtv, const Index ns) : xx(srcx), x(tgtx), u(tgtv), src_size(ns) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        ko::RVec<3> vec;
        const Index i = mbr.league_rank();
//         printf("league_rank = %d, team_rank = %d, team_size = %d\n", mbr.league_rank(), mbr.team_rank(), mbr.team_size());
        ko::parallel_reduce(ko::TeamThreadRange(mbr, src_size), VecReducer(xx, x, i), vec);
        for (int j=0; j<3; ++j) {
            u(i,j) = vec.v[j];
        }
    }
};


int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const Int nsrc = 30;
    const Int ntgt = 42;
    view_type srclocs("src_x", nsrc);
    view_type tgtlocs("tgt_x", ntgt);
    view_type tgtvel("tgt_u", ntgt);
    
    init(srclocs, tgtlocs, tgtvel);
    std::cout << "data initialized." << std::endl;
    
    std::cout << "sums = (";
    for (int i=0; i<ntgt; ++i) {
        Real sum=0;
        for (int j=0; j<nsrc; ++j) {
            const Real cp = i*j;
            sum += cp;
        }
        std::cout << sum << (i<ntgt-1 ? ", " : "");
    }    
    std::cout << ")" << std::endl;
    auto policy = ko::TeamPolicy<>(ntgt, ko::AUTO());
    
    ko::parallel_for(policy, VecComputer(srclocs, tgtlocs, tgtvel, nsrc));
    
    std::cout << "kernels returned." << std::endl;
    
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
//         std::cout << "h1(" << i << ",:) = (";
        for (int j=0; j<3; ++j) {
            h1(i,j) = (j == 0 ? i : (j==1 ? -i : 0));
//             std::cout << h1(i,j) << " ";
        }
//         std::cout << ")" << std::endl;
    }

    for (int i=0; i<v2.extent(0); ++i) {
//         std::cout << "h2(" << i << ",:) = (";
        for (int j=0; j<3; ++j) {
            h2(i,j) = (j==1 ? i : 0);
//             std::cout << h2(i,j) << " ";
        }
//         std::cout << ")" << std::endl;
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
