#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"

using namespace Lpm;

typedef ko::DefaultExecutionSpace ExeSpace;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
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
  std::cout << "edge 4 is divided = " << (edges.hasKidsHost(4) ? "true" : "false") << std::endl;
  LPM_THROW_IF(edges.hasKidsHost(4), "the impossible happened.");

  Edges sedges(14);
  const MeshSeed<QuadRectSeed> seed;
  sedges.initFromSeed(seed);
  std::cout << sedges.infoString("QuadRectSeed");
  }
  {
    const Real p0[2] = {0,0.5};
    const Real p1[2] = {-0.5,0};
    const Real p6[2] = {-1,0};

    Coords<CircularPlaneGeometry> udcrds(8);
    udcrds.insertHost(p0);
    udcrds.insertHost(p1);
    udcrds.insertHost(p6);
    Coords<CircularPlaneGeometry> udlagcrds(8);
    udlagcrds.insertHost(p0);
    udlagcrds.insertHost(p1);
    udlagcrds.insertHost(p6);
    Edges edges(6);
    const Index e0[4] = {0,1,0,1};
    const Index e11[4] = {2,1,1,2};
    edges.insertHost(e0[0], e0[1], e0[2], e0[3]);
    edges.insertHost(e11[0], e11[1], e11[2], e11[3]);
    edges.divide<CircularPlaneGeometry>(0, udcrds, udlagcrds);
    std::cout << edges.infoString("edges after radial divide", 0, true);
    Real rmidpt[2];
    for (int i=0; i<2; ++i) rmidpt[i] = udcrds.getCrdComponentHost(3,i);
    LPM_THROW_IF(!fp_equiv(CircularPlaneGeometry::mag(rmidpt), 0.5),
      "error: radial midpoint did not preserve radius.");
    edges.divide<CircularPlaneGeometry>(1, udcrds, udlagcrds);
    std::cout << edges.infoString("edges after axial divide", 0, true);
    std::cout << udcrds.infoString("crds after 2 divides", 0, true);

  }

}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
