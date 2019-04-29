#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmVtkIO.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    {
    Faces<TriFace> planeTri(11);
    const MeshSeed<TriHexSeed> thseed;
    Index nmaxverts;
    Index nmaxfaces;
    Index nmaxedges;
    thseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, 1);
    std::cout << "memory allocations " << nmaxverts << " vertices, "
              << nmaxedges << " edges, " << nmaxfaces << " faces" << std::endl;
    Coords<PlaneGeometry> thcb(11);
    Coords<PlaneGeometry> thcbl(11);
    Coords<PlaneGeometry> thci(11);
    Coords<PlaneGeometry> thcil(11);
    Edges the(24);
        
    thcb.initBoundaryCrdsFromSeed(thseed);
    thcbl.initBoundaryCrdsFromSeed(thseed);
    thcb.writeMatlab(std::cout, "bcrds1");
//     std::cout << thcb.infoString("bc init.");
    thci.initInteriorCrdsFromSeed(thseed);
//     std::cout << thci.infoString("ic init.");
    thcil.initInteriorCrdsFromSeed(thseed);
    the.initFromSeed(thseed);
//     std::cout << the.infoString("the");
    planeTri.initFromSeed(thseed);
//     std::cout << planeTri.infoString("seed");
    
    typedef FaceDivider<PlaneGeometry,TriFace> thdiv;
    thdiv::divide(0, thcb, thcbl, the, planeTri, thci, thcil);
    std::cout << planeTri.infoString("divide face 0: faces");
//     std::cout << the.infoString("divide face 0: edges");
//     std::cout << thcb.infoString("divide face 0: verts");
//     std::cout << thci.infoString("divide face 0: facecrds");

    thcb.writeMatlab(std::cout, "bcrds2");
    thci.writeMatlab(std::cout, "icrds2");
    }
    
    {
        typedef MeshSeed<CubedSphereSeed> seed_type;
        const seed_type seed;
        typedef SphereGeometry geo;
        typedef QuadFace face_type;
        Index nmaxverts;
        Index nmaxfaces;
        Index nmaxedges;
        const int maxlev = 6;
        seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, maxlev);
        std::cout << "allocating " << nmaxverts << " vertices, " << nmaxedges << " edges, and " << nmaxfaces << " faces for cubed sphere." <<std::endl;
        Coords<geo> csverts(nmaxverts);
        Coords<geo> cslagverts(nmaxverts);
        Coords<geo> csfacecrds(nmaxfaces);
        Coords<geo> cslagfacecrds(nmaxfaces);
        Faces<QuadFace> csfaces(nmaxfaces);
        Edges csedges(nmaxedges);
        csverts.initBoundaryCrdsFromSeed(seed);
        cslagverts.initBoundaryCrdsFromSeed(seed);
        csfacecrds.initInteriorCrdsFromSeed(seed);
        cslagfacecrds.initInteriorCrdsFromSeed(seed);
        csfaces.initFromSeed(seed);
        csedges.initFromSeed(seed);
        typedef FaceDivider<geo,face_type> csdiv;
        for (int i=0; i<maxlev; ++i) {
            std::cout << "tree level " << i  << ": faces.nh = " << csfaces.nh() << ", edges.nh = " << csedges.nh() << ", verts.nh = " << csverts.nh() << "; faces.nmax = " << csfaces.nMax() << std::endl;
            Index startInd = 0;
            Index stopInd = csfaces.nh();
            for (Index j=startInd; j<stopInd; ++j) {
                if (!csfaces.hasKidsHost(j)) {
                    csdiv::divide(j, csverts, cslagverts, csedges, csfaces, csfacecrds, cslagfacecrds);
                }
            }
        }
        std::cout << "Sphere surface area = " << csfaces.surfAreaHost() << std::endl;
        
        VtkInterface<SphereGeometry, Faces<QuadFace>> vtk;
        vtkSmartPointer<vtkPolyData> pd = vtk.toVtkPolyData(csfaces, csedges, csfacecrds, csverts);
        std::cout << "vtk data conversion done." << std::endl;
        vtk.writePolyData("cs_test.vtk", pd);
    }
    
    {
        const MeshSeed<IcosTriSphereSeed> seed;
        Index nmaxverts;
        Index nmaxfaces;
        Index nmaxedges;
        const int maxlev = 5;
        seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, maxlev);
        
        Coords<SphereGeometry> icvertcrds(nmaxverts);
        Coords<SphereGeometry> icvertlagcrds(nmaxverts);
        Coords<SphereGeometry> icfacecrds(nmaxfaces);
        Coords<SphereGeometry> icfacelagcrds(nmaxfaces);
        Edges icedges(nmaxedges);
        Faces<TriFace> icfaces(nmaxfaces);
        icvertcrds.initBoundaryCrdsFromSeed(seed);
        icvertlagcrds.initBoundaryCrdsFromSeed(seed);
        icfacecrds.initInteriorCrdsFromSeed(seed);
        icfacelagcrds.initInteriorCrdsFromSeed(seed);
        icfaces.initFromSeed(seed);
        icedges.initFromSeed(seed);
        typedef FaceDivider<SphereGeometry,TriFace> icdiv;
        for (int i=0; i<maxlev; ++i) {
            const Index stopInd = icfaces.nh();
            for (Index j=0; j<stopInd; ++j) {
                if (!icfaces.hasKidsHost(j)) {
                    icdiv::divide(j, icvertcrds, icvertlagcrds, icedges, icfaces, icfacecrds, icfacelagcrds);
                }
            }
        }
        std::cout << "ICT sphere surf area = " << icfaces.surfAreaHost() << std::endl;
        VtkInterface<SphereGeometry, Faces<TriFace>> vtk;
        std::cout << "writing vtk output." << std::endl;
        vtkSmartPointer<vtkPolyData> pd = vtk.toVtkPolyData(icfaces, icedges, icfacecrds, icvertcrds);
        std::cout << "conversion to vtk polydata complete." << std::endl;
        vtk.writePolyData("ic_test.vtk", pd);
        std::cout << "tests pass." << std::endl;
    }
}
ko::finalize();
return 0;
}
