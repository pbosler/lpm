#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmSeedReader.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const Int maxlev = 9;
    Index nmax_verts;
    Index nmax_faces;
    Index nmax_edges;

    SeedReader<QuadRectSeed> qrseed;
    std::cout << qrseed.infoString();
    std::cout << qrseed.idString() << " memory requirements" << std::endl;
    std::cout << std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20)
        << "nedges" << std::setw(20) << "nfaces" << std::endl;
    for (int i=0; i<maxlev; ++i) {
        qrseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << std::setw(20) << i << std::setw(20) << nmax_verts << std::setw(20) 
            << nmax_edges << std::setw(20) << nmax_faces << std::endl;
    }
    
    SeedReader<TriHexSeed> thseed;
    std::cout << thseed.infoString();
    std::cout << thseed.idString() << " memory requirements" << std::endl;
    std::cout << std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20)
        << "nedges" << std::setw(20) << "nfaces" << std::endl;
    for (int i=0; i<maxlev; ++i) {
        thseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << std::setw(20) << i << std::setw(20) << nmax_verts << std::setw(20) 
            << nmax_edges << std::setw(20) << nmax_faces << std::endl;
    }

    
    SeedReader<CubedSphereSeed> csseed;
    std::cout << csseed.infoString();
    std::cout << csseed.idString() << " memory requirements" << std::endl;
    std::cout << std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20)
        << "nedges" << std::setw(20) << "nfaces" << std::endl;
    for (int i=0; i<maxlev; ++i) {
        csseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << std::setw(20) << i << std::setw(20) << nmax_verts << std::setw(20) 
            << nmax_edges << std::setw(20) << nmax_faces << std::endl;
    }

    
    SeedReader<IcosTriSphereSeed> icseed;
    std::cout << icseed.infoString();
    std::cout << icseed.idString() << " memory requirements" << std::endl;
    std::cout << std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20)
        << "nedges" << std::setw(20) << "nfaces" << std::endl;
    for (int i=0; i<maxlev; ++i) {
        icseed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, i);
        std::cout << std::setw(20) << i << std::setw(20) << nmax_verts << std::setw(20) 
            << nmax_edges << std::setw(20) << nmax_faces << std::endl;
    }

    
}
ko::finalize();
return 0;
}
