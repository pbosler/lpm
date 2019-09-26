#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBox3d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"

using namespace Lpm;
using namespace Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{

    const int npts = 6;
    const int max_depth = 4;
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
    ko::deep_copy(pts, host_pts);
    
    NodeArrayD leaves(pts, tree_lev, max_depth);
    
    std::cout << "NodeArrayD:\n";
    std::cout << leaves.infoString();
    
    NodeArrayInternal nextlev(leaves, tree_lev-1, max_depth);
    std::cout << nextlev.infoString();
    std::cout << leaves.infoString();
    
    std::cout << "program complete." << std::endl;
}
ko::finalize();
return 0;
}