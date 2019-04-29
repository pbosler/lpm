#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmBVESphere.hpp"
#include "Kokkos_Core.hpp"

using namespace Lpm;

template <typename FaceKind> struct InitSolidBody {
    struct VertTag {};
    struct FaceTag {};
    BVESphere<FaceKind> sphere;
    ko::View<Index*[4],Dev> faceTree;
    
    static constexpr Real OMEGA = 2*PI;
    
    InitSolidBody(const BVESphere<FaceKind> sph) : sphere(sph) {}
        
    void init() const {
        ko::parallel_for(ko::RangePolicy<VertTag>(0, sphere.physVerts.nh()), *this);
        ko::parallel_for(ko::RangePolicy<FaceTag>(0, sphere.faces.nh()), *this);
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const VertTag&, const Index i) const {
        sphere.relVortVerts(i) = 2*OMEGA*sphere.physVerts.crds(i,2);
        sphere.absVortVerts(i) = 2*OMEGA*sphere.physVerts.crds(i,2);
        sphere.streamFnVerts(i) = -OMEGA*sphere.physVerts.crds(i,2);
        sphere.velocityVerts(i,0) = -OMEGA*sphere.physVerts.crds(i,1);
        sphere.velocityVerts(i,1) = OMEGA*sphere.physVerts.crds(i,0);
        sphere.velocityVerts(i,2) = 0;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const FaceTag&, const Index i) const {
        if (!sphere.faces.hasKids(i)) {
            sphere.relVortFaces(i) = 2*OMEGA*sphere.physFaces.crds(i,2);
            sphere.absVortFaces(i) = 2*OMEGA*sphere.physFaces.crds(i,2);
            sphere.streamFnFaces(i)=  -OMEGA*sphere.physFaces.crds(i,2);
            sphere.velocityFaces(i,0) = -OMEGA*sphere.physFaces.crds(i,1);
            sphere.velocityFaces(i,1) = OMEGA*sphere.physFaces.crds(i,0);
            sphere.velocityFaces(i,2) = 0;
        }
    }
};


int main (int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    Index nmaxverts, nmaxedges, nmaxfaces;
    const Index tree_depth = 4;

    MeshSeed<IcosTriSphereSeed> triseed;
    triseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);
    BVESphere<TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
    trisphere.treeInit(tree_depth, triseed);
    trisphere.updateDevice();
    std::cout << "tree initialized.  starting problem initialization." << std::endl;
    trisphere.initProblem<InitSolidBody<TriFace>>();
    std::cout << "problem init." << std::endl;
    trisphere.updateHost();
    trisphere.outputVtk("solidBody_icostri.vtk");
    
    MeshSeed<CubedSphereSeed> quadseed;
    quadseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);
    BVESphere<QuadFace> quadsphere(nmaxverts, nmaxedges, nmaxfaces);
    quadsphere.treeInit(tree_depth, quadseed);
    quadsphere.updateDevice();
    quadsphere.initProblem<InitSolidBody<QuadFace>>();
    quadsphere.updateHost();
    quadsphere.outputVtk("solidBody_cubedsph.vtk");

}
std::cout << "tests pass" << std::endl;
ko::finalize();
return 0;
}