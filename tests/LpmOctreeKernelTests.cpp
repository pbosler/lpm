#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmBox3d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeKernels.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmUtilities.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_Sort.hpp"

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
    typedef QuadFace facetype;
    typedef CubedSphereSeed seedtype;
    
    Int nerr = 0;
//=========================================================== 
//     typedef TriFace facetype;
//     typedef IcosTriSphereSeed seedtype;
    
    const int mesh_depth = 6; // must be >= 2
    const int octree_depth = 4;
    Index nmaxverts, nmaxedges, nmaxfaces;
    MeshSeed<seedtype> seed;
    seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, mesh_depth);
    PolyMesh2d<SphereGeometry,facetype> sphere(nmaxverts, nmaxedges, nmaxfaces);
    sphere.treeInit(mesh_depth, seed);
    sphere.updateDevice();
    ko::View<Real*[3]> src_crds = sourceCoords<SphereGeometry,facetype>(sphere);
    const Int npts = src_crds.extent(0);
    auto src_crds_host = ko::create_mirror_view(src_crds);
    ko::deep_copy(src_crds_host, src_crds);

    ko::View<BBox> root_box("root_box");
    ko::parallel_reduce(npts, BoxFunctor(src_crds), BBoxReducer<Dev>(root_box));
    auto rb_host = ko::create_mirror_view(root_box);
    ko::deep_copy(rb_host, root_box);
    std::cout << "root_box = " << rb_host();
//     for (Index i=0; i<src_crds.extent(0); ++i) {
//         std::cout << "sph_pt(" << i << ") = (";
//         Real nsq = 0;
//         for (int j=0; j<3; ++j) {
//             std::cout << src_crds_host(i,j) << (j!=2 ? " " : ") ");
//             nsq += square(src_crds(i,j));
//         }
//         std::cout << "has norm = " << nsq << "\n";
//     }
    const BBox spherebox(-1,1,-1,1,-1,1);
    if (rb_host() != spherebox) {
        ++nerr;
        throw std::runtime_error("Box reduction failed.");
    }
    else {
        std::cout << "Box reduction passed.\n";
    }
//===========================================================
    ko::View<code_type*> pt_codes("point_codes", npts);
    ko::parallel_for(npts, EncodeFunctor(pt_codes, src_crds, root_box, octree_depth));
    auto codes_host = ko::create_mirror_view(pt_codes);
    ko::deep_copy(codes_host, pt_codes);
    
    for (Int i=0; i<npts; ++i) {
//         std::cout << "point " << i << " has node key = " << decode_key(codes_host(i)) 
//             << " and id = " << decode_id(codes_host(i)) << "\n";
        if (i != decode_id(codes_host(i))) ++nerr; 
        const key_type ptkey = decode_key(codes_host(i));
        const auto pbox = box_from_key(ptkey, rb_host(), octree_depth, octree_depth);
        if (!boxContainsPoint(pbox, ko::subview(src_crds_host, i, ko::ALL()))) {
            ++nerr;
        }
    }
    if (nerr > 0) {
        throw std::runtime_error("presort: encode/decode id test failed.");
    }
    else {
        std::cout << "presort: pt encoding & node box test passes.\n";
    }
//===========================================================    
    ko::sort(pt_codes);
    ko::deep_copy(codes_host, pt_codes);
    
    ko::View<Real*[3]> sorted_pts("sorted_pts", npts);
    ko::View<Index*> original_ptids("orig_pt_ids", npts);
    ko::parallel_for(npts, PermuteFunctor(sorted_pts, original_ptids, src_crds, pt_codes));
    auto sort_host = ko::create_mirror_view(sorted_pts);
    auto orig_inds = ko::create_mirror_view(original_ptids);
    ko::deep_copy(sort_host, sorted_pts);
    ko::deep_copy(orig_inds, original_ptids);
    for (Index i=0; i<npts; ++i) {
//         std::cout << "sorted point " << i << " has key = " << decode_key(codes_host(i)) << " old ind = " 
//             << decode_id(codes_host(i)) << " == " << orig_inds(i) << "\n";
        if (i>0 && decode_key(codes_host(i)) < decode_key(codes_host(i-1))) ++nerr;
        if (decode_id(codes_host(i)) != orig_inds(i)) ++nerr;
    }
    if (nerr > 0) {throw std::runtime_error("postsort: encode/decode id test failed.");}
    else {std::cout << "postsort: encode/decode id test passes.\n";}
//===========================================================    
    ko::View<Real*[3]> unsorted_pts("unsorted_pts", npts);
    ko::parallel_for(npts, UnpermuteFunctor(unsorted_pts, sorted_pts, original_ptids));
    auto unsort_host = ko::create_mirror_view(unsorted_pts);
    ko::deep_copy(unsort_host, unsorted_pts);
    for (int i=0; i<npts; ++i) {
        const key_type ptkey = decode_key(codes_host(i));
        const auto pbox = box_from_key(ptkey, rb_host(), octree_depth, octree_depth);
        if (!boxContainsPoint(pbox, ko::subview(sort_host, i, ko::ALL()))) ++nerr;
        for (int j=0; j<3; ++j) {
            if (src_crds(i,j) != unsort_host(i,j)) {
                ++nerr;
                std::cout << "error: src_crds(" << i << "," << j << ") (" << 
                    src_crds_host(i,j) << " .NEQ. unsorted_pts(" << i << "," << j << ") (" <<
                    unsort_host(i,j) << ")\n";
            }
        }
    }
    if (nerr>0) {
        throw std::runtime_error("permute/unpermute test failed.");
    }
    else {
        std::cout << "permute/unpermute test passes.\n";
    }
//===========================================================    

    ko::View<Index*> unode_flags("unode_flags", npts);
    ko::View<Index*> marked_nodes("marked_nodes", npts);
    ko::View<key_type*> ukeys_test("ukeys_test", npts);
    ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0,npts),
        MarkDuplicates(unode_flags, pt_codes));
    ko::deep_copy(marked_nodes, unode_flags);
    
    ko::parallel_for(npts, KOKKOS_LAMBDA (const Index& i) {
        if (unode_flags(i) > 0) {
            ukeys_test(i) = decode_key(pt_codes(i));
        }
    });
    auto marked_nodes_host = ko::create_mirror_view(marked_nodes);
    ko::deep_copy(marked_nodes_host, marked_nodes);    
    
    ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0,npts),
        MarkDuplicates(unode_flags, pt_codes));
    
    n_view_type un_count = ko::subview(unode_flags, npts-1);
    auto un_count_host = ko::create_mirror_view(un_count);
    ko::deep_copy(un_count_host, un_count);
    const Index nunodes = un_count_host();
    std::cout << "counted " << nunodes << " nonempty nodes out of " << pintpow8(octree_depth) << " possible\n";
    
    auto uflag_host = ko::create_mirror_view(unode_flags);
    ko::deep_copy(uflag_host, unode_flags);

    
    ko::View<key_type*> ukeys("ukeys", nunodes);
    ko::View<Index*[2]> uinds("uinds", nunodes);
    ko::parallel_for(npts, UniqueNodeFunctor(ukeys, uinds, unode_flags, pt_codes));
    auto ukeys_test_host = ko::create_mirror_view(ukeys_test);
    auto ukeys_host = ko::create_mirror_view(ukeys);
    auto uinds_host = ko::create_mirror_view(uinds);
    ko::deep_copy(ukeys_host, ukeys);
    ko::deep_copy(uinds_host, uinds);
    ko::deep_copy(ukeys_test_host, ukeys_test);
    
    Index uniq_ind = 0;
    for (Index i=0; i<npts; ++i) {
        if (marked_nodes_host(i) > 0) {
            const key_type ptkey = ukeys_test_host(i);
            const auto pbox = box_from_key(ptkey, rb_host(), octree_depth, octree_depth);
            if (!boxContainsPoint(pbox, ko::subview(sort_host, i, ko::ALL()))) ++nerr;
            if (ukeys_host(uniq_ind) != ukeys_test_host(i)) {
                std::cout << "error: ukeys(" << uniq_ind << ") = " << ukeys_host(uniq_ind) << " .NEQ. ukeys_test("
                << i << ")\n";
                ++nerr;
            }
            ++uniq_ind;
        }
    }
    Index test_npts = 0;
    ko::parallel_reduce(nunodes, KOKKOS_LAMBDA (const Index& i, Index& ct) {
        ct += uinds(i,1);
    }, test_npts);
    if (npts != test_npts) ++nerr;
    if (nerr > 0) {
        throw std::runtime_error("Unique Node tests failed.");
    }
    else {
        std::cout << "unique node tests pass.\n";
    }
//===========================================================    
    ko::View<key_type*> pkeys("pkeys", nunodes);
    ko::parallel_for(nunodes, KOKKOS_LAMBDA (const Index& i) {
        pkeys(i) = parent_key(ukeys(i), octree_depth, octree_depth);
    });
    auto pkeys_host = ko::create_mirror_view(pkeys);
    ko::deep_copy(pkeys_host, pkeys);
    std::cout << "ukeys = (";
    for (Index i=0; i<nunodes; ++i) {
        std::cout << ukeys_host(i) << (i<nunodes-1 ? " " : ")\n");
    }
    std::cout << "pkeys = (";
    for (Index i=0; i<nunodes; ++i) {
        std::cout << pkeys_host(i) << (i<nunodes-1 ? " " : ")\n");
    }
    ko::View<Index*> nsiblings("nsiblings", nunodes);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0,nunodes), 
        NodeSiblingCounter(nsiblings, ukeys, octree_depth, octree_depth));
    auto na_host = ko::create_mirror_view(nsiblings);
    ko::deep_copy(na_host, nsiblings);
    std::cout << "nsiblings = (";
    for (Index i=0; i<nunodes; ++i) {
        std::cout << na_host(i) << (i<nunodes-1 ? " " : ")\n");
    }
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0,nunodes),
        NodeSiblingCounter(nsiblings, ukeys, octree_depth, octree_depth));
    ko::deep_copy(na_host, nsiblings);
    std::cout << "nsiblings = (";
    for (Index i=0; i<nunodes; ++i) {
        std::cout << na_host(i) << (i<nunodes-1 ? " " : ")\n");
    }
    n_view_type nnodesview = ko::subview(nsiblings, nunodes-1);
    auto nnodes = ko::create_mirror_view(nnodesview);
    ko::deep_copy(nnodes, nnodesview);
    std::cout << "including all siblings makes " << nnodes() << " nodes out of " << pintpow8(octree_depth) << " possible.\n";
    
    ko::View<key_type*> node_keys("node_keys", nnodes());
    ko::View<Index*[2]> node_pt_inds("node_pt_inds", nnodes());
    ko::parallel_for(nunodes, KOKKOS_LAMBDA (const Index& i) {
        bool new_parent = true;
        if (i>0) new_parent = (nsiblings(i) > nsiblings(i-1));
        if (new_parent) {
            const Index kid0_address = nsiblings(i) - 8;
            const key_type pkey = parent_key(ukeys(i), octree_depth, octree_depth);
            for (int j=0; j<8; ++j) {
                const Index node_ind = kid0_address +j;
                const key_type new_key = pkey+j;
                node_keys(node_ind) = new_key;
                const Index found_key = binarySearchKeys(new_key, ukeys, true);
//                 printf("new_key = %u found_key = %d ukeys(found_key) = %u\n", new_key, found_key, ukeys(found_key));
                if (found_key != NULL_IND) {
                    node_pt_inds(node_ind,0) = uinds(found_key,0);
                    node_pt_inds(node_ind,1) = uinds(found_key,1);    
                }
                else {
                    node_pt_inds(node_ind,0) = 0;
                    node_pt_inds(node_ind,1) = 0;
                }
            }
        }
    });
    auto nkeys_host = ko::create_mirror_view(node_keys);
    auto node_pts_host = ko::create_mirror_view(node_pt_inds);
    ko::deep_copy(nkeys_host, node_keys);
    ko::deep_copy(node_pts_host, node_pt_inds);

    for (Index i=0; i<nnodes(); ++i) {
//         std::cout << "node " << i << " has key " << nkeys_host(i) << " and contains " << node_pts_host(i,1) <<
//             " points starting at pt address " << node_pts_host(i,0) << "\n";
        const BBox nbox = box_from_key(nkeys_host(i), rb_host(), octree_depth, octree_depth);
        for (Index j=0; j<node_pts_host(i,1); ++j) {
            auto pt = ko::subview(sort_host, node_pts_host(i,0) + j, ko::ALL());
            if (!boxContainsPoint(nbox, pt)) {
                std::cout << "\tpt(" << node_pts_host(i,0) + j << ") = (" << pt(0) << " " << pt(1) << " " << pt(2)
                    << ") is not contained in node box " << nbox;
            
                ++nerr;
            }
        }
    }
    if (nerr>0) throw std::runtime_error("error: node box/point relationship test failed.");
    else {
        std::cout << "node box/point relationship tests pass.\n";
    }
//===========================================================    
}
ko::finalize();
return 0;
};
