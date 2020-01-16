#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmOctree.hpp"
#include "LpmBox3d.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmCompadre.hpp"
#include "LpmLatLonMesh.hpp"

#include "Kokkos_Core.hpp"

using namespace Lpm;
using namespace Octree;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{


    /**
        Build source mesh
    */
    typedef QuadFace facetype;
    typedef CubedSphereSeed seedtype;

//     typedef TriFace facetype;
//     typedef IcosTriSphereSeed seedtype;

    const int mesh_depth = 6; // must be >= 2 for IcosTriSphere or box test will fail
    const int octree_depth = 4;
    Index nmaxverts, nmaxedges, nmaxfaces;
    MeshSeed<seedtype> seed;
    seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, mesh_depth);
    PolyMesh2d<seedtype> sphere(nmaxverts, nmaxedges, nmaxfaces);
    sphere.treeInit(mesh_depth, seed);
    sphere.updateDevice();
    ko::View<Real*[3]> src_crds = sourceCoords<seedtype>(sphere);
    const Int npts = src_crds.extent(0);
    auto src_crds_host = ko::create_mirror_view(src_crds);
    ko::deep_copy(src_crds_host, src_crds);

    /**
        Build octree
    */
	Tree(src_crds, octree_depth);

}
ko::finalize();
return 0;
}
