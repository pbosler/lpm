#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Sort.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <exception>
#include <bitset>
#include <vector>

using namespace Lpm;
using namespace Lpm::Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    Int nerr = 0;
    const BBox sphereBox(-1,1,-1,1,-1,1);
    {// Max tree depth = 1
        std::vector<Int> leaf_keys(8);
        for (int k=0; k<8; ++k) {
            leaf_keys[k] = k;
            std::cout << "key " << k << " = " << std::bitset<3>(k) <<  " (k&4) = " << (k&4) 
                << " (k&2) = " << (k&2) << " (k&1) = " << (k&1) << "\n";
        }
//         std::cout << "sphereBox = " << sphereBox;
        std::vector<BBox> kidboxes(8);
        ko::View<BBox[8],Host> kidboxes1("kidboxes");
        bisectBoxAllDims(kidboxes1, sphereBox);
        for (int k=0; k<8; ++k) {
//             std::cout << "local_key(k,1,1) = " << local_key(k,1,1) << "\n";
            kidboxes[k] = box_from_key(leaf_keys[k], sphereBox, 1,1);
//             std::cout << "child box from key    " << k << " = " << kidboxes[k];
//             std::cout << "child box from bisect " << k << " = " << kidboxes1(k);
            if (kidboxes[k] != kidboxes1(k)) ++nerr;
        }
        if (nerr>0) {throw std::runtime_error("Box division test failed.");}
        else {std::cout << "box division test passed.\n";}
    }
    {// Max tree depth = 2
        const int max_depth = 2;
        std::vector<key_type> ldkeys(pintpow8(max_depth));
        ko::View<Real[64][3],Host> l2centroids("cntds");
        for (key_type k=0; k<64; ++k) {
            Real cx=0;
            Real cy=0;
            Real cz=0;
            Real half_len = 1;
            for (int l = 1; l<=2; ++l) {
                half_len *= 0.5;
                const key_type lkey = local_key(k, l, 2);
                if ((lkey&1) > 0) {
                    cz += half_len;
                }
                else {
                    cz -= half_len;
                }
                if ((lkey&2) > 0) {
                    cy += half_len;
                }
                else {
                    cy -= half_len;
                }
                if ((lkey&4) > 0) {
                    cx += half_len;
                }
                else {
                    cx -= half_len;
                }
            }
            l2centroids(k,0) = cx;
            l2centroids(k,1) = cy;
            l2centroids(k,2) = cz;
        }

        for (int i=0; i<64; ++i) {
            const key_type k = compute_key_for_point(ko::subview(l2centroids, i, ko::ALL()), 2, sphereBox);
            const code_type c = encode(k,i);
            const key_type k1 = decode_key(c);
            const int i1 = decode_id(c);
            if (k!=k1) ++nerr;
            if (k!=i) ++nerr;
            if (i!=i1) ++nerr;
            const BBox nbox = box_from_key(k, sphereBox, 2,2);
            auto pt = ko::subview(l2centroids, i, ko::ALL());
            if (!boxContainsPoint(nbox, pt)) ++nerr;
            Real cx, cy, cz;
            boxCentroid(cx,cy,cz, nbox);
            if (cx != l2centroids(i,0)) ++nerr;
            if (cy != l2centroids(i,1)) ++nerr;
            if (cz != l2centroids(i,2)) ++nerr;
        }
        if (nerr>0) {
            throw std::runtime_error("error in computed keys/boxes test.");
        }
        else {
            std::cout << "computed keys/boxes tests pass.\n";
        }
        
//         for (key_type k=0; k<pintpow8(max_depth); ++k) {
//             ldkeys[k] = k;
//             const key_type pkey = parent_key(k, max_depth, max_depth);
//             std::cout << "ldkeys[" << k << "] = " << k << " " << std::bitset<12>(k) << " parent = " 
//                 << pkey << " " << std::bitset<12>(pkey) 
//                 << " box = " << box_from_key(k, sphereBox, max_depth, max_depth);
//         }
    }
}
ko::finalize();
return 0;
}