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

struct Input {
    int max_pmesh_depth;
    std::string mfilename;
    std::vector<Int> nlons;
    std::vector<Int> nlats;
    Input(int argc, char* argv[]);
};

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
    Input input(argc, argv);
    
    /**
        Build source mesh 
    */
    Index nmaxverts, nmaxedges, nmaxfaces;
    MeshSeed<IcosTriSphereSeed> icseed;
    icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, input.max_pmesh_depth);
    PolyMesh2d<SphereGeometry,TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.treeInit(input.max_pmesh_depth, icseed);
    trisphere.updateDevice();
    ko::View<Real*[3]> src_crds = sourceCoords<SphereGeometry,TriFace>(trisphere);
    
    /**
        Build octree
    */    
    Tree octree(src_crds, input.max_pmesh_depth);
    std::cout << octree.infoString();
    
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
    max_pmesh_depth = 4;
    mfilename = "octree_tests.m";
    nlats = {91};//, 181, 361, 721};
    nlons = {180};//, 360, 720, 1440};
    for (int i=1; i<argc; ++i) {
        const std::string& token = argv[i];
        if (token == "-d" || token == "-tree") {
            max_pmesh_depth = std::stoi(argv[++i]);
        }
        else if (token == "-m") {
            mfilename = argv[++i];
        }
    }
}
