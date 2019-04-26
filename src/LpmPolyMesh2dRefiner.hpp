#ifndef LPM_POLYMESH2D_REFINER_HPP
#define LPM_POLYMESH2D_REFINER_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmPolyMesh2d.hpp"

namespace Lpm {

template <typename SeedType> struct UniformEdgeDivider {
    typedef typename SeedType::geo geo;
    typedef typename geo::crd_view_type crd_view;
    typedef typename PolyMesh2d<SeedType>::edge_view_type edge_view;
    typedef typename PolyMesh2d<SeedType>::edge_tree_type edge_tree;
    crd_view physCrds;
    crd_view lagCrds;
    edge_view edges;
    edge_tree edgeTree;
    Index nEdgesBefore;
    Index nEdgesAfter;
    Index nVertsBefore;
    
    UniformEdgeDivider(crd_view pc, crd_view lc, edge_view es, edge_tree et, 
        const Index nb, const Index na, const Index nv) : 
      physCrds(pc), lagCrds(lc), edges(es), edgeTree(et), nEdgesBefore(nb), nEdgesAfter(na), nVertsBefore(nv) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
        const Index parentIndex = nEdgesBefore + i;
        const Index midptInd = nVertsBefore + parentIndex;
        const Index kid0Ind = nEdgesAfter + 2*i;
        
        /// compute midpoint of parent edge
        auto orig = ko::subview(physCrds, edges(i,0), ko::ALL());
        auto dest = ko::subview(physCrds, edges(i,1), ko::ALL());
        auto lagorig = ko::subview(lagCrds, edges(i,0), ko::ALL());
        auto lagdest = ko::subview(lagCrds, edges(i,1), ko::ALL());
        
        auto midpt = ko::subview(physCrds, midptInd, ko::ALL());
        auto lagmidpt = ko::subview(lagCrds, midptInd, ko::ALL());
        
        geo::midpoint(midpt, orig, dest);
        geo::midpoint(lagmidpt, lagorig, lagdest);
        
        /// construct & initialize children
        edges(kid0Ind,0) = edges(i,0);  // kid 0: parent origin to new midpoint.
        edges(kid0Ind,1) = midptInd;
        edges(kid0Ind+1,0) = midptInd;  // kid 1: new midpoint to parent dest
        edges(kid0Ind+1,1) = edges(i,1);
        
        for (int j=0; j<2; ++j) { // loop over kids
            edges(kid0Ind+j,2) = edges(i,2);    // kids have same left face as parent
            edges(kid0Ind+j,3) = edges(i,3);    // kids have same right face as parent
            edgeTree(kid0Ind+j,0) = i;  // kids have same parent
            for (int k=1; k<3; ++k) { // loop over kids' kids (initialize)
                edgeTree(kid0Ind+j, k) = NULL_IND;
            }    
        }
    }
};

template <typename SeedType> struct UniformFaceDivider {
    typedef typename SeedType::geo geo;
    typedef typename geo::crd_view_type crd_view;
    typedef typename PolyMesh2d<SeedType>::edge_view_type edge_view;
    typedef typename PolyMesh2d<SeedType>::edge_tree_type edge_tree;
    typedef typename PolyMesh2d<SeedType>::face_view_type face_view;
    typedef typename PolyMesh2d<SeedType>::face_tree_type face_tree;
    
    crd_view physVerts;
    crd_view lagVerts;
    Index nVertsBefore;
    
    edge_view edges;
    edge_tree edgeTree;
    Index nEdgesBefore;
    
    crd_view physFaces;
    crd_view lagFaces;
    face_view faceVerts;
    face_view faceEdges;
    face_tree faceTree;
    scalar_view_type faceArea;
    Index nFacesBefore;
    Index nFacesAfter;
    
    UniformFaceDivider(crd_view pv, crd_view lv, const Index nv, edge_view es, edge_tree et, const Index ne, 
                       crd_view pf, crd_view lf, face_view fv, face_view fe, face_tree ft, scalar_view_type fa, 
                       const Index nfb, const Index nfa) :
            physVerts(pv), lagVerts(lv), nVertsBefore(nv), edges(es), edgeTree(et), nEdgesBefore(ne),
            physFaces(pf), lagFaces(lf), faceVerts(fv), faceEdges(fe), faceTree(ft), faceArea(fa),
            nFacesBefore(nfb), nFacesAfter(nfa) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
        const Index parentIndex = nFacesBefore + i;
        const Index kid0Ind = nFacesAfter+4*i;
        switch (SeedType::nfaceverts) {
            case (3) : {
                Index newFaceVerts[4][3];
                Index newFaceEdges[4][3];
                Index parentVertInds[3];
                Index parentEdgeInds[3];
                Index newFaceKids[4];
                // initialize to invalid state
                for (int j=0; j<4; ++j) {
                    for (int k=0; k<3; ++k) {
                        newFaceVerts[j][k] = NULL_IND;
                        newFaceEdges[j][k] = NULL_IND;
                    }
                }
                
                /// pull data from parent
                for (int j=0; j<3; ++j) {
                    parentVertInds[j] = faceVerts(i,j);
                    parentEdgeInds[j] = faceEdges(i,j);
                    newFaceVerts[j][j] = faceVerts(i,j); 
                }
                for (int j=0; j<4; ++j) {
                    newFaceKids[j] = kid0Ind + j;
                }
                
                /// loop over parent edges
                for (int j=0; j<3; ++j) {
                    const Index parentEdge = parentEdgeInds[j];
                    const Index edgeKid0 = edgeTree(parentEdge,1);
                    const Index edgeKid1 = edgeTree(parentEdge,2);
                    const Index kid0dest = edges(edgeKid0,1);
                    
                    if (edgeIsPositive(parentIndex, j)) {
                        newFaceEdges[j][j] = edgeKid0;
                        edges(edgeKid0,2) = newFaceKids[j];
                        
                        newFaceEdges[(j+1)%3][j] = edgeKid1;
                        edges(edgeKid1,2) = newFaceKids[(j+1)%3];
                    }
                    else {
                        newFaceEdges[j][j] = edgeKid1;
                        edges(edgeKid1,3) = newFaceKids[j];
                        
                        newFaceEdges[(j+1)%3][j] = edgeKid0;
                        edges(edgeKid0,3) = newFaceKids[(j+1)%3];
                    }
                    switch (j) {
                        case (0) : {
                            newFaceVerts[0][1] = kid0dest;
                            newFaceVerts[1][0] = kid0dest;
                            newFaceVerts[3][2] = kid0dest;
                            break;
                        }
                        case (1) : {
                            newFaceVerts[1][2] = kid0dest;
                            newFaceVerts[2][1] = kid0dest;
                            newFaceVerts[3][0] = kid0dest;
                            break;
                        }
                        case (2) : {
                            newFaceVerts[2][0] = kid0dest;
                            newFaceVerts[0][2] = kid0dest;
                            newFaceVerts[3][1] = kid0dest;
                            break;
                        }
                    }
                }
                
                /// create new interior edges
                for (int j=0; j<3; ++j) {
                    newFaceEdges[3][j] = nEdgesBefore + j;
                }
                newFaceEdges[0][1] = nEdgesBefore + 1;
                newFaceEdges[1][2] = nEdgesBefore + 2;
                newFaceEdges[2][0] = nEdgesBefore;
                initEdge(nEdgesBefore, newFaceVerts[2][1], newFaceVerts[2][0], newFaceKids[3], newFaceKids[2]);
                initEdge(nEdgesBefore, newFaceVerts[0][2], newFaceVerts[0][1], newFaceKids[3], newFaceKids[0]);
                initEdge(nEdgesBefore, newFaceVerts[1][0], newFaceVerts[1][2], newFaceKids[3], newFaceKids[1]);
                
                /// create new face coordinates
                Real vert_crds[3][geo::ndim];
                Real lag_vert_crds[3][geo::ndim];
                for (int j=0; j<4; ++j) { // loop over child faces
                    auto ctr = ko::subview(physFaces, newFaceKids[j], ko::ALL());
                    auto lagctr = ko::subview(lagFaces, newFaceKids[j], ko::ALL());
                    for (int k=0; k<3; ++k) { // loop over vertices
                        for (int l=0; l<geo::ndim; ++l) { // loop over components
                            vert_crds[k][l] = physVerts(newFaceVerts[j][k], l);
                            lag_vert_crds[k][l] = lagVerts(newFaceVerts[j][k],l);
                        }
                    }
                    geo::barycenter(ctr, vert_crds, 3);
                    geo::barycenter(lagctr, lag_vert_crds, 3);
                    faceArea(newFaceKids[j]) = geo::polygonArea(ctr, vert_crds, 3);
                }
                /// initialize child faces
                for (int j=0; j<4; ++j) {
                    for (int k=0; k<3; ++k) {
                        faceVerts(newFaceKids[j],k) = newFaceVerts[j][k];
                        faceEdges(newFaceKids[j],k) = newFaceEdges[j][k];
                    }
                    faceTree(newFaceKids[j],0) = parentIndex;
                    faceTree(parentIndex,1+j) = newFaceKids[j];
                }
                faceArea(parentIndex) = 0.0;
                
                break;
            }
            case (4) : {
                Index newFaceEdges[4][4];
                Index newFaceVerts[4][4];
                Index newFaceKids[4];
                auto parentVerts = ko::subview(faceVerts, parentIndex, ko::ALL());
                auto parentEdges = ko::subview(faceEdges, parentIndex, ko::ALL());
                for (int j=0; j<4; ++j) {
                    newFaceKids[j] = kid0Ind + j;
                    for (int k=0; k<4; ++k) {
                        newFaceEdges[j][k] = NULL_IND;
                        newFaceVerts[j][k] = NULL_IND;
                    }
                }
                for (int j=0; j<4; ++j) { // loop over parent edges
                    newFaceVerts[j][j] = parentVerts(j);
                    
                    const Index parentEdge = parentEdges[j];
                    const Index edgeKid0 = edgeTree(parentEdge,1);
                    const Index edgeKid1 = edgeTree(parentEdge,2);
                    if (edgeIsPositive(parentIndex, j)) {
                        newFaceEdges[j][j] = edgeKid0;
                        edges(edgeKid0,2) = newFaceKids[j];
                        
                        newFaceEdges[(j+1)%4][j] = edgeKid1;
                        edges(edgeKid1,2) = newFaceKids[(j+1)%4];
                    }
                    else {
                        newFaceEdges[j][j] = edgeKid1;
                        edges(edgeKid1,3) = newFaceKids[j];
                        
                        newFaceEdges[(j+1)%4][j] = edgeKid0;
                        edges(edgeKid0,3) = newFaceKids[(j+1)%4];
                    }
                    
                    const Index kid0dest = edges(edgeKid0,1);
                    newFaceVerts[j][(j+1)%4] = kid0dest;
                    newFaceVerts[(j+1)%4][j] = kid0dest;
                }
                
                /// special case for quads: parent center becomes a child vertex.
                const Index vertexCenterInd = nVertsBefore + i;
                for (int j=0; j<geo::ndim; ++j) {
                    physVerts(vertexCenterInd,j) = physFaces(parentIndex,j);
                    lagVerts(vertexCenterInd,j) = lagFaces(parentIndex,j);
                }
                for (int j=0; j<4; ++j) {
                    newFaceVerts[j][(j+2)%4] = vertexCenterInd;
                }
                
                /// create new edges
                const Index edgeInsertPoint = nEdgesBefore + 4*i;
                initEdge(edgeInsertPoint, newFaceVerts[0][1], newFaceVerts[0][2], newFaceKids[0], newFaceKids[1]);
                newFaceEdges[0][1] = edgeInsertPoint;
                newFaceEdges[1][3] = edgeInsertPoint;
                initEdge(edgeInsertPoint, newFaceVerts[2][0], newFaceVerts[2][3], newFaceKids[3], newFaceKids[2]);
                newFaceEdges[2][3] = edgeInsertPoint+1;
                newFaceEdges[3][1] = edgeInsertPoint+1;
                initEdge(edgeInsertPoint, newFaceVerts[2][1], newFaceVerts[2][0], newFaceKids[1], newFaceKids[2]);
                newFaceEdges[1][2] = edgeInsertPoint;
                newFaceEdges[2][0] = edgeInsertPoint;
                initEdge(edgeInsertPoint, newFaceVerts[3][1], newFaceVerts[3][0], newFaceKids[0], newFaceKids[3]);
                newFaceEdges[0][2] = edgeInsertPoint;
                newFaceEdges[3][0] = edgeInsertPoint;
                
                /// create new faces
                Real vert_crds[4][geo::ndim];
                Real lag_vert_crds[4][geo::ndim];
                for (int j=0; j<4; ++j) {
                    auto ctr = ko::subview(physFaces, newFaceKids[j], ko::ALL());
                    auto lagctr = ko::subview(lagFaces, newFaceKids[j], ko::ALL());
                    for (int k=0; k<4; ++k) {
                        for (int l=0; l<geo::ndim; ++l) {
                            vert_crds[k][l] = physVerts(newFaceVerts[j][k],l);
                            lag_vert_crds[k][l] = lagVerts(newFaceVerts[j][k],l);
                        }
                        faceVerts(newFaceKids[j],k) = newFaceVerts[j][k];
                        faceEdges(newFaceKids[j],k) = newFaceEdges[j][k];
                    }
                    geo::barycenter(ctr, vert_crds, 4);
                    geo::barycenter(lagctr, lag_vert_crds,4);
                    faceArea(newFaceKids[j]) = geo::polygonArea(ctr, vert_crds, 4);
                    faceTree(parentIndex,1+j) = newFaceKids[j];
                    faceTree(newFaceKids[j],0) = parentIndex;
                }
                faceArea(parentIndex) = 0.0;
                
                break;
            }
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void initEdge(const Index ind, const Index o, const Index d, const Index l, const Index r) const {
        edges(ind,0) = o;
        edges(ind,1) = d;
        edges(ind,2) = l;
        edges(ind,3) = r;
    }
    
    /// return true if parent face is the left face of its edge
    KOKKOS_INLINE_FUNCTION
    bool edgeIsPositive(const Index parentFace, const Int relEdge) const {
        return parentFace == edges(faceEdges(parentFace, relEdge), 2);
    }
    
};

}
#endif
