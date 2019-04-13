#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmMeshSeed.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    Faces<TriFace> planeTri(11);
    Faces<TriFace> sphereTri(30);
    //std::cout << planeTri.infoString("init");
    
    
    {
    const MeshSeed<TriHexSeed> thseed;
    Index nmaxverts;
    Index nmaxfaces;
    Index nmaxedges;
    thseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 1);
    std::cout << "memory allocations " << nmaxverts << " vertices, "
              << nmaxedges << " edges, " << nmaxfaces << " faces" << std::endl;
    Coords<PlaneGeometry> thcb(11);
    Coords<PlaneGeometry> thcbl(11);
    Coords<PlaneGeometry> thci(11);
    Coords<PlaneGeometry> thcil(11);
    Edges the(24);
        
    thcb.initBoundaryCrdsFromSeed(thseed);
    thcbl.initBoundaryCrdsFromSeed(thseed);
    thcb.writeMatlab(std::cout, "bcrds1");
    std::cout << thcb.infoString("bc init.");
    thci.initInteriorCrdsFromSeed(thseed);
    std::cout << thci.infoString("ic init.");
    thcil.initInteriorCrdsFromSeed(thseed);
    the.initFromSeed(thseed);
    std::cout << the.infoString("the");
    planeTri.initFromSeed(thseed);
    std::cout << planeTri.infoString("seed");
    
    typedef FaceDivider<PlaneGeometry,TriFace> thdiv;
    thdiv::divide(0, planeTri, the, thci, thcil, thcb, thcbl);
    std::cout << planeTri.infoString("divide face 0: faces");
//     std::cout << the.infoString("divide face 0: edges");
//     std::cout << thcb.infoString("divide face 0: verts");
//     std::cout << thci.infoString("divide face 0: facecrds");

    thcb.writeMatlab(std::cout, "bcrds2");
    thci.writeMatlab(std::cout, "icrds2");
    }
    
    
}
ko::finalize();
return 0;
}
