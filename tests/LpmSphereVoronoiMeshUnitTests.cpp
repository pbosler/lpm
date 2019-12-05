#include "LpmConfig.h"
#include "LpmDefs.hpp"

#include "Kokkos_Core.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include "LpmSphereVoronoiMesh.hpp"
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

    MeshSeed<IcosTriDualSeed> seed;

    VoronoiMesh<IcosTriDualSeed> vmesh0(seed, 0);
    std::vector<Index> vinds;
    std::vector<Index> einds0;
    std::vector<Index> einds1;
    std::vector<Index> finds;

    std::cout << vmesh0.infoString(true);

    for (Short i=0; i<vmesh0.nverts(); ++i) {
        vmesh0.getEdgesAndCellsAtVertex(einds0, finds, i);
    }

    for (Short i=0; i<vmesh0.ncells(); ++i) {
        vmesh0.getEdgesAndVerticesInCell(einds1, vinds, i);
    }

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
