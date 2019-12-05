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

    VoronoiMesh<IcosTriDualSeed> vmesh(seed, 0);

    std::cout << vmesh.infoString(true);

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
