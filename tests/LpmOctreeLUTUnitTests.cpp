#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <exception>

using namespace Lpm;
using namespace Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    std::ostringstream ss;
    ss << "LUT Unit tests:\n";
    Int nerr=0;
    
    ko::View<ParentLUT> ptable("ParentLUT");
    ko::View<ChildLUT> ctable("ChildLUT");
    ko::View<NeighborsAtVertexLUT> nvtable("NeighborsAtVertexLUT");
    ko::View<NeighborsAtEdgeLUT> netable("NeighborsAtEdgeLUT");
    
    // identity tests
    ko::View<Int[8]> parentI("parent_identity");
    ko::View<Int[8]> childI("child_identity");
    ko::View<Int[8]> nvI("neighbor_vertex_identity");
    ko::parallel_for(8, KOKKOS_LAMBDA (const Int& i) {
        parentI(i) = table_val(i,13, ptable);
        childI(i) = table_val(i,13, ctable);
        nvI(i) = table_val(i,7-i, nvtable);
    });
    auto pi_host = ko::create_mirror_view(parentI);
    auto ci_host = ko::create_mirror_view(childI);
    auto nv_host = ko::create_mirror_view(nvI);
    ko::deep_copy(pi_host, parentI);
    ko::deep_copy(ci_host, childI);
    ko::deep_copy(nv_host, nvI);
    for (Int i=0; i<8; ++i) {
        if (pi_host(i) != 13) {
            nerr += 1;
            ss << "\tidentity error: table_val(" << i << ",13, ParentLUT) = " << pi_host(i) << " (should be 13)\n";
        }
        if (ci_host(i) != i) {
            nerr += 1;
            ss << "\tidentity error: table_val(" << i << ",13, ChildLUT) = " << ci_host(i) << " (should be " << i << ")\n";
        }
        if (nv_host(i) != 13) {
            nerr += 1;
            ss << "\tidentity error: table_val(" << i << "," << 7-i << ", NeighborsAtVertexLUT) = " << nv_host(i)
               << " (should be 13)\n";
        }
    }
    
    // bad value tests
    ko::View<bool[8][27]> bad_parent_nbrs("bad_parent_nbrs");
    ko::View<bool[8][27]> bad_child_inds("bad_child_inds");
    ko::View<bool[12][4]> bad_edge_neighbors("bad_edge_neighbors");
    ko::parallel_for(12, KOKKOS_LAMBDA (const Int& i) {
        const Int bad_vals[8] = {0, 2, 6, 8, 18, 20, 24, 26};
        for (Int j=0; j<4; ++j) {
            bad_edge_neighbors(i,j) = false;
            const Int val = table_val(i,j,netable);
            for (Int k=0; k<8; ++k) {
                if (val == bad_vals[k]) bad_edge_neighbors(i,j) = true;
            }
        }
    });
    ko::parallel_for(27, KOKKOS_LAMBDA (const Int& i) {
        for (Int j=0; j<27; ++j) {
            bad_parent_nbrs(i,j) = false;
            bad_child_inds(i,j) = false;
            const Int pval = table_val(i,j, ptable);
            const Int cval = table_val(i,j, ctable);
            if (pval < 0 || pval > 26) bad_parent_nbrs(i,j) = true;
            if (cval < 0 || cval > 7) bad_child_inds(i,j) = true;
        }
    });
    auto bv_en_host = ko::create_mirror_view(bad_edge_neighbors);
    auto bv_pn_host = ko::create_mirror_view(bad_parent_nbrs);
    auto bv_ci_host = ko::create_mirror_view(bad_child_inds);
    ko::deep_copy(bv_en_host, bad_edge_neighbors);
    ko::deep_copy(bv_pn_host, bad_parent_nbrs);
    ko::deep_copy(bv_ci_host, bad_child_inds);
    for (Int i=0; i<12; ++i) {
        for (Int j=0; j<4; ++j) {
            if (bv_en_host(i,j)) {
                nerr += 1;
                ss << "\tbad value error: table_val(" << i << "," << j << ", NeighborsAtEdgeLUT) is wrong.\n";
            }
        }
    }
    for (Int i=0; i<8; ++i) {
        for (Int j=0; j<27; ++j) {
            if (bv_pn_host(i,j)) {
                nerr += 1;
                ss << "\tbad value error: table_val(" << i << "," << j << ", ParentLUT) is wrong.\n";
            }
            if (bv_ci_host(i,j)) {
                nerr += 1;
                ss << "\tbad value error: table_val(" << i << "," << j << ", ChildLUT) is wrong.\n";
            }
        }
    }
    
    
    if (nerr == 0) {
        ss << "\tall tests pass.\n";
        std::cout << ss.str();
    }
    else {
        throw std::runtime_error(ss.str());
    }
}
ko::finalize();
return 0;
}
