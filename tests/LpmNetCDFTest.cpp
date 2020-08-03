#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmNetCDF.hpp"
#include "LpmNetCDF_Impl.hpp"

#include "Kokkos_Core.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{

  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;
  MeshSeed<TriHexSeed> thseed;
  thseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
  auto triplane =
    std::shared_ptr<PolyMesh2d<TriHexSeed>>(new
      PolyMesh2d<TriHexSeed>(nmaxverts, nmaxedges, nmaxfaces));
  triplane->treeInit(3, thseed);
  triplane->updateDevice();

  NcWriter tri_hex_writer("tri_hex.nc");
  tri_hex_writer.writePolymesh(triplane);


//   MeshSeed<QuadRectSeed> qrseed;
//   qrseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
//   PolyMesh2d<QuadRectSeed> quadplane(nmaxverts, nmaxedges, nmaxfaces);
//   quadplane.treeInit(3, qrseed);
//   quadplane.updateDevice();
//
//   NcWriter("quadrect.nc");
//
//   MeshSeed<IcosTriSphereSeed> icseed;
//   icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
//   PolyMesh2d<IcosTriSphereSeed> trisphere(nmaxverts, nmaxedges, nmaxfaces);
//   trisphere.treeInit(3, icseed);
//   trisphere.updateDevice();
//
//   NcWriter("icostrisphere.nc");
//
//   MeshSeed<CubedSphereSeed> csseed;
//   csseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
//   PolyMesh2d<CubedSphereSeed> quadsphere(nmaxverts, nmaxedges, nmaxfaces);
//   quadsphere.treeInit(3, csseed);
//   quadsphere.updateDevice();
//
//   NcWriter("cubedsphere.nc");
}
ko::finalize();
return 0;
}
