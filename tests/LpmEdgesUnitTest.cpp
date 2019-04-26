#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"

using namespace Lpm;

typedef ko::DefaultExecutionSpace ExeSpace;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    Coords<SphereGeometry> sc4(6);
    const Real p0[3] = {0.57735026918962584,  -0.57735026918962584,  0.57735026918962584};
    const Real p1[3] = {0.57735026918962584,  -0.57735026918962584,  -0.57735026918962584};
    const Real p2[3] = {0.57735026918962584,  0.57735026918962584,  -0.57735026918962584};
    const Real p3[3] = {0.57735026918962584,  0.57735026918962584, 0.57735026918962584};
    sc4.insertHost(p0);
    sc4.insertHost(p1);
    sc4.insertHost(p2);
    sc4.insertHost(p3);
    sc4.updateDevice();
    
    std::cout << sc4.infoString("sc4 init");
    
    Coords<SphereGeometry> sc4lag(6);
    sc4lag.insertHost(p0);
    sc4lag.insertHost(p1);
    sc4lag.insertHost(p2);
    sc4lag.insertHost(p3);
    sc4lag.updateDevice();
    
    Edges edges(6);
    const Index e0[4] = {0,1,0,3};
    const Index e1[4] = {1,2,0,5};
    const Index e2[4] = {2,3,0,1};
    std::cout << "Edges: edges.nh() = " << edges.nh() << ", edges.nmax() = " << edges.nmax() << std::endl;
    
    edges.insertHost(e0[0], e0[1], e0[2], e0[3]);
    edges.insertHost(e1[0], e1[1], e1[2], e1[3]);
    edges.insertHost(e2[0], e2[1], e2[2], e2[3]);   
    std::cout << "Edges: edges.nh() = " << edges.nh() << ", edges.nmax() = " << edges.nmax() << std::endl;
    std::cout << edges.infoString("edges init");
    
    std::cout << "calling divide." << std::endl;
    edges.divide<SphereGeometry>(0, sc4, sc4lag);
    edges.updateDevice();
    sc4.updateDevice();
    sc4lag.updateDevice();
    std::cout << "Edges: edges.nh() = " << edges.nh() << ", edges.nmax() = " << edges.nmax() << std::endl;
    std::cout << sc4.infoString("sc4.divide(0)");
    std::cout << edges.infoString("edges after divide");

    std::cout << "edge 0 is divided = " << (edges.hasKidsHost(0) ? "true" : "false") << std::endl;
    LPM_THROW_IF(!edges.hasKidsHost(0), "edge divide error.");
    std::cout << "edge 1 is divided = " << (edges.hasKidsHost(1) ? "true" : "false") << std::endl;
    LPM_THROW_IF(edges.hasKidsHost(1), "the impossible happened.");
    std::cout << "edge 5 is divided = " << (edges.hasKidsHost(5) ? "true" : "false") << std::endl;
    LPM_THROW_IF(edges.hasKidsHost(5), "the impossible happened.");

    Edges sedges(14);
    const MeshSeed<QuadRectSeed> seed;
    sedges.initFromSeed(seed);
    std::cout << sedges.infoString("QuadRectSeed");
    
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
