#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Sort.hpp"
#include <iostream>
#include <iomanip>
#include <string>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    ko::View<Real*[3]> pts("pts",4);
    typename ko::View<Real*[3]>::HostMirror host_pts = ko::create_mirror_view(pts);
    for (int i=0; i<4; ++i) {
        host_pts(i,0) = (i%2==0 ? 2.0*i : -i);
        host_pts(i,1) = (i%2==1 ? 2-i : 1 + i);
        host_pts(i,2) = (i%2==0 ? i : -i);
    }
    ko::deep_copy(pts, host_pts);
    
    Octree::BBox box = Octree::get_bbox(pts);
    
    for (int i=0; i<4; ++i) {
        std::cout << "pt(" << i << ") = (";
        for (int j=0; j<3; ++j) {
            std::cout << std::setw(5) << host_pts(i,j) << (j<2 ? " " : ")\n");
        }
    }
    std::cout << "----------------------\n";
    std::cout << "bbox = (";
    for (int i=0; i<6; ++i) {
        std::cout << std::setw(4) << box.bds[i] << (i<5 ? " " : ")\n");
    }
    
    const int tree_lev = 3;
    ko::View<uint_fast32_t*> keys("keys",4);
    ko::View<uint_fast64_t*> codes("codes",4);
    ko::parallel_for(keys.extent(0), KOKKOS_LAMBDA (const Int& i) {
        auto pos = ko::subview(pts, i, ko::ALL());
        keys(i) = Octree::compute_key(pos, tree_lev);
        codes(i) = Octree::encode(keys(i), i);
    });
    auto host_keys = ko::create_mirror_view(keys);
    ko::deep_copy(host_keys, keys);
    std::cout << "presort: keys = (";
    for (int i=0; i<keys.extent(0); ++i) 
        std::cout << host_keys(i) << (i<3 ? " " : ")\n");
    
    ko::sort(keys);
    ko::deep_copy(host_keys, keys);
    std::cout << "postsort: keys = (";
    for (int i=0; i<4; ++i) {
        std::cout << host_keys(i) << (i<3 ? " " : ")\n");
    }
    
    auto host_codes = ko::create_mirror_view(codes);
    ko::deep_copy(host_codes, codes);
    std::cout << "presort: codes = (";
    for (int i=0; i<4; ++i) {
        std::cout << host_codes(i) << (i<3 ? " " : ")\n");
    }
    
    ko::sort(codes);
    ko::deep_copy(host_codes, codes);
    std::cout << "postsort: codes = (";
    for (int i=0; i<4; ++i) {
        std::cout << host_codes(i) << (i<3 ? " " : ")\n");
    }
    
    ko::View<uint_fast32_t*> decoded_keys("decoded_keys",4);
    ko::parallel_for(keys.extent(0), KOKKOS_LAMBDA (const Int& i) {
        decoded_keys(i) = Octree::decode_id(codes(i));
    });
    auto host_ids = ko::create_mirror_view(decoded_keys);
    ko::deep_copy(host_ids, decoded_keys);
    std::cout << "decoded ids = (";
    for (int i=0; i<4; ++i) {
        std::cout << host_ids(i) << (i<3 ? " " : ")\n");
    }
    
}
ko::finalize(); 
return 0;
}
