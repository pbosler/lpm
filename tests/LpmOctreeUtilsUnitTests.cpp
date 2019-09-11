#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "Kokkos_Core.hpp"
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
    
    Octree::BBox box = Octree::get_bbox<Host>(pts);
    
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
}
return 0;
ko::finalize();    
}
