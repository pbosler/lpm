#include "LpmConfig.h"
#include "LpmDefs.hpp"

#include "Kokkos_Core.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include "LpmSphereVoronoiMesh.hpp"
#include "LpmVtkIO.hpp"
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
    ss << "Sphere Voronoi Mesh Unit tests:\n";
    Int nerr=0;

    MeshSeed<IcosTriDualSeed> seed;

    VoronoiMesh<IcosTriDualSeed> vmesh0(seed, 0);
    std::vector<Index> vinds;
    std::vector<Index> einds0;
    std::vector<Index> einds1;
    std::vector<Index> finds;

//     std::cout << vmesh0.infoString(true);

    Voronoi::VtkInterface<IcosTriDualSeed> vtk;
    auto pd = vtk.toVtkPolyData(vmesh0);
    vtk.writePolyData("vmesh0.vtk", pd);

    for (Short i=0; i<vmesh0.nverts(); ++i) {
        vmesh0.getEdgesAndCellsAtVertex(einds0, finds, i);
    }

    for (Short i=0; i<vmesh0.ncells(); ++i) {
        vmesh0.getEdgesAndVerticesInCell(einds1, vinds, i);
    }

    for (Short i=0; i<vmesh0.ncells(); ++i) {
        const Real qpt[3] = {vmesh0.cells[i].xyz[0], vmesh0.cells[i].xyz[1], vmesh0.cells[i].xyz[2]};
        const auto loc = vmesh0.cellContainingPoint(qpt, 0);
        if (loc != i) {
            ++nerr;
        }
    }

    for (Int i=6; i<9; ++i) {
        ss.str(std::string());
        VoronoiMesh<IcosTriDualSeed> vmesh(seed,i);
        std::cout << vmesh.infoString(false);
        pd = vtk.toVtkPolyData(vmesh);
        ss << "vmesh" << i << ".vtk";
        vtk.writePolyData(ss.str(),pd);
    }



    if (nerr>0) {
        throw std::runtime_error("point location identity test failed.");
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
