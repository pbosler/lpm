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
    Faces<TriFace> planeTri(20);
    Faces<TriFace> sphereTri(30);
    std::cout << planeTri.infoString("init");
    {
    MeshSeed<TriHexSeed> thseed;
    planeTri.initFromSeed(thseed);
    std::cout << planeTri.infoString("seed");
    }
    
    
}
ko::finalize();
return 0;
}
