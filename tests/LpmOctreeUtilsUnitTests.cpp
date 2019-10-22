#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Sort.hpp"
#include "LpmOctreeKernels.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <exception>
#include <bitset>

using namespace Lpm;
using namespace Lpm::Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    Int nerr = 0;
    const int npts = 7;
    const int max_depth = 3;
    const int tree_lev = 3;
    ko::View<Real*[3]> pts("pts",npts);
    typename ko::View<Real*[3]>::HostMirror host_pts = ko::create_mirror_view(pts);
    for (int i=0; i<4; ++i) {
        host_pts(i,0) = (i%2==0 ? 2.0*i : -i);
        host_pts(i,1) = (i%2==1 ? 2-i : 1 + i);
        host_pts(i,2) = (i%2==0 ? i : -i);
    }
    host_pts(4,0) = 0.01;
    host_pts(4,1) = 0.9;
    host_pts(4,2) = 0.01;
    host_pts(5,0) = 0.011;
    host_pts(5,1) = 0.91;
    host_pts(5,2) = 0.015;
    host_pts(6,0) = 0.0111;
    host_pts(6,1) = 0.911;
    host_pts(6,2) = 0.0151;
    ko::deep_copy(pts, host_pts);
    
    BBox box;
    ko::parallel_reduce(pts.extent(0), BoxFunctor(pts), BBoxReducer<Host>(box));
    std::cout << "bbox = " << box;
    std::cout << "\taspect ratio = " << boxAspectRatio(box) << "\n";
    BBox boxsol(-3,4,-1,3,-3,2);
    if (box != boxsol) {
        throw std::runtime_error("bbox computation failed.\n");
    }
    else {
        std::cout << "BBox test passed.\n";
    }
    box2cube(box);
    std::cout << "after box2cube, bbox = " << box;
    std::cout << "\taspect ratio = " << boxAspectRatio(box) << "\n";
    for (int i=0; i<npts; ++i) {
        std::cout << "pt(" << i << ") = (";
        for (int j=0; j<3; ++j) {
            std::cout << std::setw(5) << host_pts(i,j) << (j<2 ? " " : ")\n");
        }
        for (int j=0; j<=tree_lev; ++j) {
        	const key_type k = compute_key_for_point(ko::subview(host_pts, i, ko::ALL()), max_depth, box);
        	const key_type p = parent_key(k, j, max_depth);
        	const key_type lk = local_key(k, j, max_depth);
        	std::cout << std::setw(20) << "key at lev " << j << " = " << std::setw(6) << k << " : " <<  std::bitset<12>(k) 
        		<<  " parent = " << std::setw(6) <<  p << " : " << std::bitset<12>(p) 
        		<< " local = " << std::setw(4) << lk << " : " << std::bitset<12>(lk) << "\n";
        }
    }
    std::cout << "----------------------\n";
    
    
    
    ko::View<BBox[8]> boxkids("boxkids");
    n_view_type inkid("inkid");
    ko::parallel_for(1, KOKKOS_LAMBDA (const int& i) {
        BBox b(-1,1,-1,1,-1,1);
        bisectBoxAllDims(boxkids, b);
        Real qp[3];
        qp[0] = 0.5;
        qp[1] = 0.25;
        qp[2] = -0.75;
        inkid() = child_index(b,qp);
        printf("(%f,%f,%f) is in kid = %d\n",qp[0],qp[1],qp[2],inkid());
    });
    auto host_kids = ko::create_mirror_view(boxkids);
    ko::deep_copy(host_kids, boxkids);
    for (int i=0; i<8; ++i) {
        std::cout << "child[" << i << "] = " << *(host_kids.data()+i);
    }
    auto host_kid_id = ko::create_mirror_view(inkid);
    ko::deep_copy(host_kid_id, inkid);
    if (host_kid_id() != 6) {
        throw std::runtime_error("Local child id test failed.\n");
    }
    else {
        std::cout << "Local child id test passed.\n";
    }
    
    ko::View<BBox> boxview("box");
    auto hb = ko::create_mirror_view(boxview);
    hb() = box;
    ko::deep_copy(boxview, hb);
    ko::View<key_type*> keys("keys",npts);
    ko::View<code_type*> codes("codes",npts);
    ko::parallel_for(keys.extent(0), KOKKOS_LAMBDA (const Int& i) {
        auto pos = ko::subview(pts, i, ko::ALL());
        keys(i) = compute_key_for_point(pos, max_depth, boxview());
        codes(i) = encode(keys(i), i);
    });
    auto host_keys = ko::create_mirror_view(keys);
    ko::deep_copy(host_keys, keys);
    std::cout << "presort: keys = (";
    for (int i=0; i<keys.extent(0); ++i) 
        std::cout << host_keys(i) << (i<npts-1 ? " " : ")\n");
    
    ko::sort(keys);
    ko::deep_copy(host_keys, keys);
    std::cout << "postsort: keys = (";
    for (int i=0; i<npts; ++i) {
        std::cout << host_keys(i) << (i<npts-1 ? " " : ")\n");
    }
    
    auto host_codes = ko::create_mirror_view(codes);
    ko::deep_copy(host_codes, codes);
    std::cout << "presort: codes = (";
    for (int i=0; i<npts; ++i) {
        std::cout << host_codes(i) << (i<npts-1 ? " " : ")\n");
    }
    
    ko::sort(codes);
    ko::deep_copy(host_codes, codes);
    std::cout << "postsort: codes = (";
    for (int i=0; i<npts; ++i) {
        std::cout << host_codes(i) << (i<npts-1 ? " " : ")\n");
    }
    
    ko::View<key_type*> decoded_ids("decoded_ids",npts);
    ko::parallel_for(keys.extent(0), KOKKOS_LAMBDA (const Int& i) {
        decoded_ids(i) = decode_id(codes(i));
    });
    auto host_ids = ko::create_mirror_view(decoded_ids);
    ko::deep_copy(host_ids, decoded_ids);
    std::cout << "decoded ids = (";
    for (int i=0; i<npts; ++i) {
        std::cout << host_ids(i) << (i<npts-1 ? " " : ")\n");
    }
    
    std::cout << "decoded keys = (";
    for (int i=0; i<npts; ++i) {
        std::cout << decode_key(host_codes(i)) << (i<npts-1 ? " " : ")\n");
    }
    
    ko::View<Index*> orig_id("original_pt_id", pts.extent(0));
    {
        ko::View<Real*[3]> temp_pts("temp_pts",pts.extent(0));
        ko::parallel_for(pts.extent(0), PermuteFunctor(temp_pts, orig_id, pts, codes));
        pts = temp_pts;
    }
    ko::deep_copy(host_pts, pts);
    std::cout << "sorted pts :\n";
    for (int i=0; i<npts; ++i) {
        std::cout << "pts(" << i << ") = (" ;
        for (int j=0; j<3; ++j) {
            std::cout << host_pts(i,j) << (j<2 ? " " : ")\n");
        }
    }
    
    {
        ko::View<Index*> flag_view("flags",npts);
        ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0,npts),
             MarkDuplicates(flag_view, codes));
        auto fhost = ko::create_mirror_view(flag_view);
        ko::deep_copy(fhost, flag_view);
        std::cout << "flags after marking: (";
        for (int i=0; i<npts; ++i) {
            std::cout << fhost(i) << (i<npts-1 ? " " : ")\n");
        }
        
        ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0,npts),
            MarkDuplicates(flag_view, codes));
        ko::deep_copy(fhost, flag_view);
        std::cout << "flags after scan   : (";
        for (int i=0; i<npts; ++i) {
            std::cout << fhost(i) << (i<npts-1 ? " " : ")\n");
        }
        
        n_view_type count_view = ko::subview(flag_view, npts-1);
        auto host_ct = ko::create_mirror_view(count_view);
        ko::deep_copy(host_ct, count_view);
        std::cout << "found " << host_ct()+1 << " unique keys.\n";
        ko::View<key_type*> ukeys("unique_keys", host_ct());
        ko::View<Index*[2]> pt_inds("pt_inds",host_ct());
        ko::parallel_for(npts, UniqueNodeFunctor(ukeys, pt_inds, flag_view, codes));
        auto uhost = ko::create_mirror_view(ukeys);
        ko::deep_copy(uhost, ukeys);
        auto ihost = ko::create_mirror_view(pt_inds);
        ko::deep_copy(ihost, pt_inds);
        
        std::cout << "unique keys = (";
        for (int i=0; i<ukeys.extent(0); ++i) {
            std::cout << uhost(i) << (i<ukeys.extent(0)-1 ? " " : ")\n");
        }
        std::cout << "pt start, count: \n";
        for (int i=0; i<ukeys.extent(0); ++i) {
        	std::cout << "key(" << uhost(i) << ") start = " << ihost(i,0) << " count = " 
        		<< ihost(i,1) << "\n";
        }
        ko::View<Index*> node_address("node_address", ukeys.extent(0));
        auto host_na = ko::create_mirror_view(node_address);
        ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0, ukeys.extent(0)), 
        	NodeSiblingCounter(node_address, ukeys, tree_lev, max_depth));
        ko::deep_copy(host_na, node_address);
        for (int i=0; i<ukeys.extent(0); ++i) {
            std::cout << "node_address(mark)(i) = " << host_na(i) << " pkey = " << std::bitset<32>(parent_key(uhost(i), tree_lev, max_depth)) << "\n";
        }
        ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0, ukeys.extent(0)),
        	NodeSiblingCounter(node_address, ukeys, tree_lev, max_depth));
        ko::deep_copy(host_na, node_address);
        for (int i=0; i<ukeys.extent(0); ++i) {
            std::cout << "node_address(scan)(i) = " << host_na(i) << " pkey = " << std::bitset<32>(parent_key(uhost(i), tree_lev, max_depth)) << "\n";
        }

        
    }
    if (nerr>0) {
        throw std::runtime_error("Error found in LpmOctreeUtilsUnitTests");
    }
    std::cout << "program complete" << std::endl;
}
ko::finalize(); 
return 0;
}
