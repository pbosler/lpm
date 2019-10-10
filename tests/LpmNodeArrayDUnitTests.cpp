#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBox3d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"
#include "LpmPolyMesh2d.hpp"
#include <fstream>
using namespace Lpm;
using namespace Octree;

template <typename G, typename F>
ko::View<Real*[3]> sourceCoords(const PolyMesh2d<G,F>& pm) {
    const Index nv = pm.nvertsHost();
    const Index nl = pm.faces.nLeavesHost();
    std::cout << "nv = " << nv << " nleaf_faces = " << nl << "\n";
    ko::View<Real*[3]> result("source_coords", nv + nl);
    std::cout << "srcCrds result allocated.\n";
    ko::parallel_for(nv, KOKKOS_LAMBDA (int i) {
        for (int j=0; j<3; ++j) {
            result(i,j) = pm.physVerts.crds(i,j);
        }
    });
    std::cout << "vertices copied to srcCrds.\n";
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
        Int offset = nv;
        for (int j=0; j<pm.nfaces(); ++j) {
            if (!pm.faces.mask(j)) {
                result(offset,0) = pm.physFaces.crds(j,0);
                result(offset,1) = pm.physFaces.crds(j,1);
                result(offset++,2) = pm.physFaces.crds(j,2);
            }
        }
    });
    std::cout << "faces copied to srcCrds.\n";
    return result;
}

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
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
    
    std::cout << "points ready.\n";
    
    NodeArrayD leaves(pts, tree_lev, max_depth);
    
    NodeArrayInternal nextlev(leaves);
    
    NodeArrayInternal topLevel(nextlev);
    
    NodeArrayInternal root(topLevel);
    std::cout << root.infoString();
    std::cout << topLevel.infoString();
    std::cout << nextlev.infoString();
    std::cout << leaves.infoString();
    
    }
    
    {
    /**
        Build source mesh 
    */
    const int mesh_depth = 4;
    const int octree_depth = 4;
    Index nmaxverts, nmaxedges, nmaxfaces;
    MeshSeed<IcosTriSphereSeed> icseed;
    icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, mesh_depth);
    PolyMesh2d<SphereGeometry,TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.treeInit(mesh_depth, icseed);
    trisphere.updateDevice();
    ko::View<Real*[3]> src_crds = sourceCoords<SphereGeometry,TriFace>(trisphere);
    
    NodeArrayD leaves(src_crds, octree_depth, octree_depth);
    NodeArrayInternal nextlev(leaves);
    std::ofstream of("node_array_d_test_output.txt");
    of << leaves.infoString();
    of << nextlev.infoString();
    of.close();
    }
    std::cout << "program complete." << std::endl;
}
ko::finalize();
return 0;
}