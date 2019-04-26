#include "LpmPolyMesh2d.hpp"
#include "LpmPolyMesh2dRefiner.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

template <typename SeedType>
PolyMesh2d<SeedType>::PolyMesh2d(const int initTreeDepth, const int maxDepth )  : 
    nVerts("nverts"), nFaces("nfaces"), nEdges("nedges"), nLeafEdges("nleafedges"), nLeafFaces("nleaffaces"),
    baseTreeDepth(initTreeDepth), maxTreeDepth(maxDepth)
{
    Index maxverts, maxedges, maxfaces;
    MeshSeed<SeedType> seed;
    seed.setMaxAllocations(maxverts, maxedges, maxfaces, maxTreeDepth);
    
    vertCrds = crd_view_type("vertcrds", maxverts);
    lagVertCrds = crd_view_type("lagvertcrds", maxverts);

    edges = edge_view_type("edges",maxedges);
    edgeTree = edge_tree_type("edge_tree",maxedges);

    faceCrds = crd_view_type("facecrds", maxfaces);
    lagFaceCrds = crd_view_type("lagfacecrds", maxfaces);    
    faceVerts = face_view_type("faceverts", maxfaces);
    faceEdges = face_view_type("faceedges", maxfaces);
    faceTree = face_tree_type("face_tree", maxfaces); 
    faceArea = scalar_view_type("face_area",maxfaces);
    
    seedInit();
}

template <typename SeedType>
void PolyMesh2d<SeedType>::seedInit() {
    typedef typename crd_view_type::HostMirror host_crd_view;
    typedef typename edge_view_type::HostMirror host_edge_view;
    typedef typename edge_tree_type::HostMirror host_edge_tree;
    typedef typename face_view_type::HostMirror host_face_view;
    typedef typename face_tree_type::HostMirror host_face_tree;
    typedef typename n_view_type::HostMirror host_n;
    typedef typename scalar_view_type::HostMirror host_scalar;

    host_crd_view hvertCrds = ko::create_mirror_view(vertCrds);
    host_crd_view hlagVertCrds = ko::create_mirror_view(lagVertCrds);
    host_crd_view hfaceCrds = ko::create_mirror_view(faceCrds);
    host_crd_view hlagFaceCrds = ko::create_mirror_view(lagFaceCrds);
    host_n nverts = ko::create_mirror_view(nVerts);

    host_edge_view hedges = ko::create_mirror_view(edges);
    host_edge_tree hedgetree = ko::create_mirror_view(edgeTree);
    host_n nedges = ko::create_mirror_view(nEdges);
    host_n nleafedges = ko::create_mirror_view(nLeafEdges);
    
    host_face_view hfaceverts = ko::create_mirror_view(faceVerts);
    host_face_view hfaceedges = ko::create_mirror_view(faceEdges);
    host_face_tree hfacetree = ko::create_mirror_view(faceTree);
    host_scalar hfacearea = ko::create_mirror_view(faceArea);
    host_n nfaces = ko::create_mirror_view(nFaces);
    host_n nleaffaces = ko::create_mirror_view(nLeafFaces);
    
    MeshSeed<SeedType> seed;
    seed.setMaxAllocations(nverts(0), nedges(0), nfaces(0), baseTreeDepth);
    nleaffaces(0) = SeedType::nFacesAtTreeLevel(baseTreeDepth);
    nleafedges(0) = SeedType::nEdgesAtTreeLevel(nverts(0), nleaffaces(0));

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
        hfacearea(i) = seed.faceArea(i);
    }
    // copy seed data to device
    ko::deep_copy(vertCrds, hvertCrds);
    ko::deep_copy(lagVertCrds, hlagVertCrds);
    ko::deep_copy(faceCrds, hfaceCrds);
    ko::deep_copy(lagFaceCrds, hlagFaceCrds);
    ko::deep_copy(faceArea, hfacearea);
    ko::deep_copy(edges, hedges);
    ko::deep_copy(edgeTree, hedgetree);
    ko::deep_copy(faceVerts, hfaceverts);
    ko::deep_copy(faceEdges, hfaceedges);
    ko::deep_copy(faceTree, hfacetree);
    ko::deep_copy(nVerts, nverts);
    ko::deep_copy(nEdges, nedges);
    ko::deep_copy(nLeafEdges, nleafedges);
    ko::deep_copy(nFaces, nfaces);
    ko::deep_copy(nLeafFaces, nleaffaces);

    
    // uniform refinement to base tree depth
    for (int i=1; i<baseTreeDepth; ++i) {
        Index nv_before; // number of vertices in tree memory at current level
        Index nf_before; // number of faces in tree memory at current level
        Index ne_before; // number of edges in tree memory at current level
        seed.setMaxAllocations(nv_before, ne_before, nf_before, i-1);
        
        Index ne_after = ne_before + 2*ne_before;
        ko::parallel_for(ne_before, UniformEdgeDivider<SeedType>(vertCrds, lagVertCrds, edges, edgeTree,
            ne_before, ne_after, nv_before));
            
        ne_before = ne_after;
        ne_after = ne_before + SeedType::nfaceverts*ne_before;
        const Index nf_after = nf_before + 4*nf_before;
        ko::parallel_for(nf_before, UniformFaceDivider<SeedType>(vertCrds, lagVertCrds, nv_before, edges, edgeTree, 
            ne_before, faceCrds, lagFaceCrds, faceVerts, faceEdges, faceTree, nf_before, nf_after));
    }
}


/// ETI
template class PolyMesh2d<TriHexSeed>;
template class PolyMesh2d<QuadRectSeed>;
template class PolyMesh2d<IcosTriSphereSeed>;
template class PolyMesh2d<CubedSphereSeed>;
}
