#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmNetCDF.hpp"
#include "LpmNetCDF_Impl.hpp"

#include "Kokkos_Core.hpp"

#include <typeinfo>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
  const std::string f1 = "test_filename1.nc";
  const std::string f2 = "test_filename2.vtp";
  std::cout << f1
            << " ends in .nc = " << std::boolalpha << has_nc_file_extension(f1) << "\n";
  std::cout << f2
            << " ends in .nc = " << std::boolalpha << has_nc_file_extension(f2) << "\n";

  std::cout << "Lpm(Real): " << typeid(Real).name() << " nc_real_type: "
    << typeid(nc_real_type).name() << "\n";
  std::cout << "Lpm(Index): " << typeid(Index).name() << " nc_index_type: "
    << typeid(nc_index_type).name() << "\n";


  Index nmaxverts;
  Index nmaxedges;
  Index nmaxfaces;

  {
    MeshSeed<TriHexSeed> thseed;
    thseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    auto triplane =
      std::shared_ptr<PolyMesh2d<TriHexSeed>>(new
        PolyMesh2d<TriHexSeed>(nmaxverts, nmaxedges, nmaxfaces));
    triplane->treeInit(3, thseed);
    triplane->updateDevice();

    std::cout << triplane->infoString("triplane base");

    scalar_view_type ones("ones", triplane->nvertsHost());
    ko::parallel_for("ones fill", triplane->nvertsHost(),
      KOKKOS_LAMBDA (const Index& i) {
        ones(i) = 1.0;
      });


    NcWriter tri_hex_writer("tri_hex.nc");
    tri_hex_writer.writePolymesh(triplane);
    tri_hex_writer.writeScalarField(ones, VertexField);

    PolyMeshReader tri_hex_reader("tri_hex.nc");
    const auto physcrds = Coords<PlaneGeometry>(tri_hex_reader.getVertPhysCrdView());
    const auto facecrds = Coords<PlaneGeometry>(tri_hex_reader.getFaceLagCrdView());
    std::cout << physcrds.infoString("physcrds after reading");
    std::cout << facecrds.infoString("facecrds after reading");
    Edges tri_hex_edges(tri_hex_reader);
    std::cout << tri_hex_edges.infoString("edges from netcdf");
    Faces<TriFace> tri_hex_faces(tri_hex_reader);
    std::cout << tri_hex_faces.infoString("faces from netcdf");
  }
  {
    MeshSeed<QuadRectSeed> qrseed;
    qrseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    auto quadplane =
      std::shared_ptr<PolyMesh2d<QuadRectSeed>>(new
        PolyMesh2d<QuadRectSeed>(nmaxverts, nmaxedges, nmaxfaces));
    quadplane->treeInit(3, qrseed);
    quadplane->updateDevice();
    quadplane->outputVtk("qp_seed.vtk");

    ko::View<Real*[2]> twos("twos", quadplane->nfacesHost());
    ko::parallel_for("twos fill", quadplane->nfacesHost(),
      KOKKOS_LAMBDA (const Index& i) {
        twos(i,0) = 2.0;
        twos(i,1) = 2.0;
    });

    NcWriter quadrect_writer("quad_rect.nc");
    quadrect_writer.writePolymesh(quadplane);
    quadrect_writer.writeVectorField(twos, FaceField);

    PolyMeshReader quad_rect_reader("quad_rect.nc");
    auto quad_from_nc=std::shared_ptr<PolyMesh2d<QuadRectSeed>>(
      new PolyMesh2d<QuadRectSeed>(quad_rect_reader));
    quad_from_nc->outputVtk("qp_netcdf.vtk");
  }
  {
    MeshSeed<IcosTriSphereSeed> icseed;
    icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    auto trisphere = std::shared_ptr<PolyMesh2d<IcosTriSphereSeed>>(new
      PolyMesh2d<IcosTriSphereSeed>(nmaxverts, nmaxedges, nmaxfaces));
    trisphere->treeInit(3, icseed);
    trisphere->updateDevice();

    NcWriter icostri_writer("icostri_sphere.nc");
    icostri_writer.writePolymesh(trisphere);
  }
  {
    MeshSeed<CubedSphereSeed> csseed;
    csseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 3);
    auto quadsphere = std::shared_ptr<PolyMesh2d<CubedSphereSeed>>(new
      PolyMesh2d<CubedSphereSeed>(nmaxverts, nmaxedges, nmaxfaces));
    quadsphere->treeInit(3, csseed);
    quadsphere->updateDevice();

    NcWriter cs_writer("cubed_sphere.nc");
    cs_writer.writePolymesh(quadsphere);
  }
}
ko::finalize();
return 0;
}
