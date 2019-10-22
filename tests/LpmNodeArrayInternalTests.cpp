#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBox3d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"
#include "LpmPolyMesh2d.hpp"

#include "Kokkos_Core.hpp"

#include <fstream>
#include <exception>

using namespace Lpm;
using namespace Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const int mesh_depth = 4;
    const int octree_depth = 3;
    Index nmaxverts, nmaxedges, nmaxfaces;
    MeshSeed<IcosTriSphereSeed> icseed;
    icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, mesh_depth);
    PolyMesh2d<SphereGeometry,TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.treeInit(mesh_depth, icseed);
    trisphere.updateDevice();
    ko::View<Real*[3]> src_crds = sourceCoords<SphereGeometry,TriFace>(trisphere);

    NodeArrayD leaves(src_crds, octree_depth);
    std::cout << "NodeArrayD constructor returned.\n";
    std::cout << leaves.infoString();
    
    NodeArrayInternal level2(leaves);
//     std::cout << level2.infoString();
}
ko::finalize();
return 0;
}

