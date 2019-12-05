#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"

using namespace Lpm;

#define MEM_HEADER std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20) \
        << "nedges" << std::setw(20) << "nfaces" << std::endl

#define MEM_LINE(lev, nv, ne, nf) std::setw(20) << lev << std::setw(20) << nv << std::setw(20) \
            << ne << std::setw(20) << nf << std::endl

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const Int maxlev = 9;
    Index nmax_verts;
    Index nmax_faces;
    Index nmax_edges;

    MeshSeed<QuadRectSeed> qrseed;
    std::cout << qrseed.infoString();
    std::cout << qrseed.idString() << " memory requirements" << std::endl;
    std::cout << MEM_HEADER;
    for (int i=0; i<maxlev; ++i) {
        qrseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << MEM_LINE(i, nmax_verts, nmax_edges, nmax_faces);
    }

    MeshSeed<TriHexSeed> thseed;
    std::cout << thseed.infoString();
    std::cout << thseed.idString() << " memory requirements" << std::endl;
    std::cout << MEM_HEADER;
    for (int i=0; i<maxlev; ++i) {
        thseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << MEM_LINE(i, nmax_verts, nmax_edges, nmax_faces);
    }


    MeshSeed<CubedSphereSeed> csseed;
    std::cout << csseed.infoString();
    std::cout << csseed.idString() << " memory requirements" << std::endl;
    std::cout << MEM_HEADER;
    for (int i=0; i<maxlev; ++i) {
        csseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << MEM_LINE(i, nmax_verts, nmax_edges, nmax_faces);
    }


    MeshSeed<IcosTriSphereSeed> icseed;
    std::cout << icseed.infoString();
    std::cout << icseed.idString() << " memory requirements" << std::endl;
    std::cout << MEM_HEADER;
    for (int i=0; i<maxlev; ++i) {
        icseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << MEM_LINE(i, nmax_verts, nmax_edges, nmax_faces);
    }

    MeshSeed<IcosTriDualSeed> icdseed;
    std::cout << icdseed.infoString();
    std::cout << icdseed.idString() << " memory requirements\n";
    std::cout << MEM_HEADER;
    for (int i=0; i<maxlev; ++i) {
        icdseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << MEM_LINE(i, nmax_verts, nmax_edges, nmax_faces);
    }

}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
