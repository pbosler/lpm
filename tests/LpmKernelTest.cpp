#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
using namespace Lpm;

typedef ko::View<Real*[3],Dev> view_type;
typedef ko::TeamPolicy<>::member_type member_type;

void init(view_type v1, view_type v2, view_type v3);

struct VecReducer {
    typedef ko::Tuple<Real,3> value_type;
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
            dst.data[j] += src.data[j];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index &j, value_type& v) const {
        auto xs = ko::subview(xx, j, ko::ALL());
        auto xt = ko::subview(x, i, ko::ALL());
        const value_type cp = SphereGeometry::cross(xs, xt);
        for (int k=0; k<3; ++k) {
            v.data[k] += cp.data[k];
        }
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
        ko::Tuple<Real,3> vec;
        const Index i = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr, src_size), VecReducer(xx, x, i), vec);
        for (int j=0; j<3; ++j) {
            u(i,j) = vec.data[j];
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
    
    std::vector<Real> sums(ntgt);
    for (int i=0; i<ntgt; ++i) {
        Real sum=0;
        for (int j=0; j<nsrc; ++j) {
            const Real cp = i*j;
            sum += cp;
        }
        sums[i] = sum;
    }    
    
    auto policy = ko::TeamPolicy<>(ntgt, ko::AUTO());
    ko::parallel_for(policy, VecComputer(srclocs, tgtlocs, tgtvel, nsrc));
    std::cout << "kernels returned." << std::endl;
    
    auto hvel = ko::create_mirror_view(tgtvel);
    ko::deep_copy(hvel, tgtvel);
    for (Int i=0; i<ntgt; ++i) {
        LPM_THROW_IF(hvel(i,2) != sums[i], "Incorrect sum.");
    }
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}

void init(view_type v1, view_type v2, view_type v3) {
    auto h1 = ko::create_mirror_view(v1);
    auto h2 = ko::create_mirror_view(v2);
    auto h3 = ko::create_mirror_view(v3);
    
    for (int i=0; i<v1.extent(0); ++i) {
        for (int j=0; j<3; ++j) {
            h1(i,j) = (j == 0 ? i : (j==1 ? -i : 0));
        }
    }

    for (int i=0; i<v2.extent(0); ++i) {
        for (int j=0; j<3; ++j) {
            h2(i,j) = (j==1 ? i : 0);
       }
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
