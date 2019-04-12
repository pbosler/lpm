#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.cpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    Faces<TriFace> planeTri(10);
    Faces<TriFace> sphereTri(30);
    
    

}
ko::finalize();
return 0;
}
