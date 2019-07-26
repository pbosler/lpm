#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSpherePoisson.hpp"
#include <iostream>
#include <sstream>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    for (int i=0; i<4; ++i) {
        int tree_depth = i;
        Index nmaxverts, nmaxedges, nmaxfaces;
        std::ostringstream ss;
        {    
        MeshSeed<IcosTriSphereSeed> triseed;
        triseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);
    
        SpherePoisson<TriFace> ic(nmaxverts, nmaxedges, nmaxfaces);
        ic.treeInit(tree_depth, triseed);
        ic.updateDevice();
    
        ic.init();
        std::cout << "icostri: ";
        ic.solve();
    
        ic.updateHost();
        ss << "poisson_ic_" << tree_depth << ".vtk";
        ic.outputVtk(ss.str());
        ss.str("");
        }
        {
        MeshSeed<CubedSphereSeed> quadseed;
        quadseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);
    
        SpherePoisson<QuadFace> cs(nmaxverts, nmaxedges, nmaxfaces);
        cs.treeInit(tree_depth, quadseed);
        cs.updateDevice();
    
        cs.init();
        std::cout << "cubedsphere: ";
        cs.solve();
    
        cs.updateHost();
        ss << "poisson_cs_" << tree_depth << ".vtk";
        cs.outputVtk(ss.str());
        ss.str("");
        }
    }
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
