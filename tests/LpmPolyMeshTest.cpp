#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2d.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    Index nmaxverts;
    Index nmaxedges;
    Index nmaxfaces;

    MeshSeed<TriHexSeed> thseed;
    thseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    PolyMesh2d<TriHexSeed> triplane(nmaxverts, nmaxedges, nmaxfaces);
    triplane.treeInit(3, thseed);
    triplane.outputVtk("triplane_test.vtk");
    triplane.updateDevice();

    MeshSeed<QuadRectSeed> qrseed;
    qrseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    PolyMesh2d<QuadRectSeed> quadplane(nmaxverts, nmaxedges, nmaxfaces);
    quadplane.treeInit(3, qrseed);
    quadplane.outputVtk("quadplane_test.vtk");
    quadplane.updateDevice();

    MeshSeed<IcosTriSphereSeed> icseed;
    icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    PolyMesh2d<IcosTriSphereSeed> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.treeInit(3, icseed);
    trisphere.outputVtk("trisphere_test.vtk");
    trisphere.updateDevice();

    MeshSeed<CubedSphereSeed> csseed;
    csseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    PolyMesh2d<CubedSphereSeed> quadsphere(nmaxverts, nmaxedges, nmaxfaces);
    quadsphere.treeInit(3, csseed);
    quadsphere.outputVtk("quadsphere_test.vtk");
    quadsphere.updateDevice();

    MeshSeed<UnitDiskSeed> udseed;
    udseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 4);
    PolyMesh2d<UnitDiskSeed> udisk(nmaxverts, nmaxedges, nmaxfaces);
    udisk.treeInit(3,udseed);
    udisk.outputVtk("unitdisk_test.vtk");
    udisk.updateDevice();
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
