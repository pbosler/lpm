#include "LpmPolyMesh2d.hpp"

namespace Lpm {

template <typename FaceType, typename SeedType>
PolyMesh2d<FaceType,SeedType>::PolyMesh2d(const int initTreeDepth, const int maxTreeDepth )  : 
    nVerts("nverts"), nFaces("nfaces"), nEdges("nedges"), nLeafEdges("nleafedges"), nLeafFaces("nleaffaces")
{
    Index maxverts, maxedges, maxfaces;
    MeshSeed<SeedType> seed;
    seed.setMaxAllocations(maxverts, maxedges, maxfaces, maxTreeDepth);
    
    vertCrds = crd_view_type("vertcrds", maxverts);
    lagVertCrds = crd_view_type("lagvertcrds", maxverts);
    
    faceCrds = crd_view_type("facecrds", maxfaces);
    lagFaceCrds = crd_view_type("lagfacecrds", maxfaces);
    
    edges = edge_view_type("edges",maxedges);
    edgeTree = edge_tree_type("edge_tree",maxedges);
    
    faceVerts = face_view_type("faceverts", maxfaces);
    faceEdges = face_view_type("faceedges", maxfaces);
    faceTree = face_tree_type("face_tree", maxfaces); 
    
    ko::parallel_for(1,KOKKOS_LAMBDA (int i) {
        nVerts(0) = 0;
        nFaces(0) = 0;
        nEdges(0) = 0;
        nLeafEdges(0) = 0;
        nLeafFaces(0) = 0;
    });
}

/// ETI
template class PolyMesh2d<TriFace, TriHexSeed>;
template class PolyMesh2d<QuadFace, QuadRectSeed>;
template class PolyMesh2d<TriFace, IcosTriSphereSeed>;
template class PolyMesh2d<QuadFace, CubedSphereSeed>;
}
