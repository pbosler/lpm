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
    
    seedInit();
    
    ko::parallel_for(1,KOKKOS_LAMBDA (int i) {
        nVerts(0) = SeedType::nverts;
        nFaces(0) = SeedType::nfaces;
        nEdges(0) = SeedType::nedges;
        nLeafEdges(0) = SeedType::nEdges;
        nLeafFaces(0) = SeedType::nfaces;
    });
}

template <typename FaceType, typename SeedType>
void PolyMesh2d<FaceType,SeedType>::seedInit() {
    MeshSeed<SeedType> seed;
    typedef typename crd_view_type::HostMirror host_crd_view;
    typedef typename edge_view_type::HostMirror host_edge_view;
    typedef typename edge_tree_type::HostMirror host_edge_tree;
    typedef typename face_view_type::HostMirror host_face_view;
    typedef typename face_tree_type::HostMirror host_face_tree;

    host_crd_view hvertCrds = ko::create_mirror_view(vertCrds);
    host_crd_view hlagVertCrds = ko::create_mirror_view(lagVertCrds);
    host_crd_view hfaceCrds = ko::create_mirror_view(faceCrds);
    host_crd_view hlagFaceCrds = ko::create_mirror_view(lagFaceCrds);

    host_edge_view hedges = ko::create_mirror_view(edges);
    host_edge_tree hedgetree = ko::create_mirror_view(edgeTree);
    
    host_face_view hfaceverts = ko::create_mirror_view(faceVerts);
    host_face_view hfaceedges = ko::create_mirror_view(faceEdges);
    host_face_tree hfacetree = ko::create_mirror_view(faceTree);

    // initialize vertices
    for (int i=0; i<SeedType::nverts; ++i) {
        for (int j=0; j<geo::ndim; ++j) {
            hvertCrds(i,j) = seed.scrds(i,j);
            hlagVertCrds(i,j) = seed.scrds(i,j);
        }
    }
    // initialize edges
    for (int i=0; i<SeedType::nedges; ++i) {
        for (int j=0; j<4; ++j) {
            hedges(i,j) = seed.sedges(i,j);   
        }
        for (int j=0; j<3; ++j) {
            hedgetree(i,j) = NULL_IND;
        }
    }
    // initialize faces
    for (int i=0; i<SeedType::nfaces; ++i) {
        for (int j=0; j<geo::ndim; ++j) {
            hfaceCrds(i,j) = seed.scrds(SeedType::nverts+i,j);
            hlagFaceCrds(i,j) = seed.scrds(SeedType::nverts+i,j);
        }
        for (int j=0; j<SeedType::nfaceverts; ++j) {
            hfaceverts(i,j) = seed.sfaceverts(i,j);
            hfaceedges(i,j) = seed.sfaceedges(i,j);
        }
        for (int j=0; j<5; ++j) {
            hfacetree(i,j) = NULL_IND;
        }
    }
    // copy to device
    ko::deep_copy(vertCrds, hvertCrds);
    ko::deep_copy(lagVertCrds, hlagVertCrds);
    ko::deep_copy(faceCrds, hfaceCrds);
    ko::deep_copy(lagFaceCrds, hlagFaceCrds);
    ko::deep_copy(edges, hedges);
    ko::deep_copy(edgeTree, hedgetree);
    ko::deep_copy(faceVerts, hfaceverts);
    ko::deep_copy(faceEdges, hfaceedges);
    ko::deep_copy(faceTree, hfacetree);
}


/// ETI
template class PolyMesh2d<TriFace, TriHexSeed>;
template class PolyMesh2d<QuadFace, QuadRectSeed>;
template class PolyMesh2d<TriFace, IcosTriSphereSeed>;
template class PolyMesh2d<QuadFace, CubedSphereSeed>;
}
