#include "LpmConfig.h"
#include "LpmDefs.hpp"

#include "Kokkos_Core.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <exception>

using namespace Lpm;
using namespace Voronoi;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    std::ostringstream ss;
    ss << "Sphere Voronoi Primitives Unit tests:\n";
    Int nerr=0;

    const Real xyz[3] = {1.0, 0.0, 0.0};
    const Real latlon[2] = {0.0, 0.0};
    Cell cell(xyz,  0);
    std::cout << cell.infoString();

    Vertex vertex(xyz, 0);
    std::cout << vertex.infoString();

    WingedEdge edge(xyz, 0, 1, 0, 1, 1, 2, 3, 4);
    std::cout << edge.infoString();

    if (nerr == 0) {
        ss << "\tall tests pass.\n";
        std::cout << ss.str();
    }
    else {
        throw std::runtime_error(ss.str());
    }
}
ko::finalize();
return 0;
}
