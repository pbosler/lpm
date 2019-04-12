#include "LpmFaces.hpp"

namespace Lpm {

template <typename FaceKind>
void Faces<FaceKind>::insertHost(const Index ctr_ind, ko::View<Index*,Host> vertinds, ko::View<Index*,Host> edgeinds, const Index prt, const Real ar) {
    LPM_THROW_IF(_nh(0)+1 > _nmax, "Faces::insert error: not enough memory.");
    const Index ins = _nh(0);
    for (int i=0; i<FaceKind::nverts; ++i) {
        _hostverts(ins, i) = vertinds(i);
        _hostedges(ins, i) = edgeinds(i);
    }
    for (int i=0; i<4; ++i) {
        _hostkids(ins, i) = NULL_IND;
    }
    _hostcenters(ins) = ctr_ind;
    _hostparent(ins) = prt;
    _hostarea(ins) = ar;
    _nh(0) += 1;
    _hnActive(0) += 1;
}

template <typename FaceKind>
void Faces<FaceKind>::setKids(const Index parent, const Index* kids) {
    for (int i=0; i<4; ++i) {
        _hostkids(parent, i) = kids[i];
    }
}

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd, Faces<TriFace>& faces, Edges& edges, Coords<Geo>& intr_crds, Coords<Geo>& intr_lagcrds, Coords<Geo>& bndry_crds, Coords<Geo>& bndry_lagcrds) {
    LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
    LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");
    
    ko::View<Index[4][3], Host> newFaceEdgeInds("newFaceEdges");  // (child face index, edge index)
    ko::View<Index[4][3], Host> newFaceVertInds("newFaceVerts");  // (child face index, vertex index)
    
    // for debugging, set to invalid value
    for (int i=0; i<4; ++i) {
        for (int j=0; j<3; ++j) {
            newFaceVertInds(i,j) = NULL_IND;
            newFaceEdgeInds(i,j) = NULL_IND;
        }
    }
    
    const Index face_insert_pt = faces.nh();
    ko::View<Index[4], Host> newFaceKids("newFaceKids");
    for (int i=0; i<4; ++i) {
        newFaceKids(i) = face_insert_pt+i;
    }

    /// connect parent vertices to child faces
    ko::View<Index[3], Host> parentVertInds("parentVerts");
    for (int i=0; i<3; ++i) {
        newFaceVertInds(i,i) = parentVertInds(i);
    }
    
    /// loop over parent edges, replace with child edges
    ko::View<Index[3], Host> parentEdgeInds("parentEdges");
    for (int i=0; i<3; ++i) {
        const Index parentEdge = parentEdgeInds(i);
        ko::View<Index[2], Host> edge_kids;
        if (edges.hasKidsHost(parentEdge)) { // edge already divided
            edges.getKidsHost(edge_kids, parentEdge);
        }
        else { // divide edge
            edge_kids(0) = edges.nh();
            edge_kids(1) = edges.nh() + 1;
            
            edges.divide<Geo>(parentEdge, bndry_crds, bndry_lagcrds);
        }
        
        // connect child edges to child faces
        if (faces.edgeIsPositive(faceInd, i, edges)) { // edge has positive orientation
            newFaceEdgeInds(i,i) = edge_kids(0);
            edges.setLeft(edge_kids(0), newFaceKids(i));
            
            newFaceEdgeInds((i+1)%3, i) = edge_kids(1);
            edges.setLeft(edge_kids(1), newFaceKids((i+1)%3));
        }
        else { // edge has negative orientation
            newFaceEdgeInds(i,i) = edge_kids(1);
            edges.setRight(edge_kids(1), newFaceKids(i));
            
            newFaceEdgeInds((i+1)%3, i) = edge_kids(0);
            edges.setRight(edge_kids(0), newFaceKids((i+1)%3));
        }
        
        const Index c1dest = edges.getDestHost(edge_kids(1));
        if (i==0) {
            newFaceVertInds(0,1) = c1dest;
            newFaceVertInds(1,0) = c1dest;
            newFaceVertInds(3,2) = c1dest;
        }
        else if (i==1) {
            newFaceVertInds(1,2) = c1dest;
            newFaceVertInds(2,1) = c1dest;
            newFaceVertInds(3,0) = c1dest;
        }
        else {
            newFaceVertInds(2,0) = c1dest;
            newFaceVertInds(0,2) = c1dest;
            newFaceVertInds(3,1) = c1dest;
        }
    }
    
    // debug: check vertex connectivity
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            LPM_THROW_IF(newFaceVertInds(i,j) == NULL_IND, "TriFace::divide error: vertex connectivity");
        }
    }
    
    /// create new interior edges
    const Index edge_ins_pt = edges.nh();
    for (int i=0; i<3; ++i) {
        newFaceEdgeInds(3,i) = edge_ins_pt+i;
    }
    newFaceEdgeInds(0,1) = edge_ins_pt+1;
    newFaceEdgeInds(1,2) = edge_ins_pt+2;
    newFaceEdgeInds(2,0) = edge_ins_pt;
    edges.insertHost(newFaceVertInds(2,1), newFaceVertInds(2,0), newFaceKids(3), newFaceKids(2));
    edges.insertHost(newFaceVertInds(0,2), newFaceVertInds(0,1), newFaceKids(3), newFaceKids(0));
    edges.insertHost(newFaceVertInds(1,0), newFaceVertInds(1,2), newFaceKids(3), newFaceKids(1));
    
    /// create new center coordinates
    ko::View<Real[3][Geo::ndim], Host> vertCrds;
    ko::View<Real[3][Geo::ndim], Host> vertLagCrds;
    ko::View<Real[4][Geo::ndim], Host> faceCrds;
    ko::View<Real[4][Geo::ndim], Host> faceLagCrds;
    ko::View<Real[4], Host> faceArea;
    for (int i=0; i<4; ++i) { // loop over child Faces
        auto ctr = ko::subview(faceCrds, i, ko::ALL());
        auto lagctr = ko::subview(faceLagCrds, i, ko::ALL());
        faceArea(i) = 0.0;
        for (int j=0; j<3; ++j) { // loop over vertices
            for (int k=0; k<Geo::ndim; ++k) { // loop over components
                vertCrds(j,k) = bndry_crds.getCrdComponentHost(newFaceVertInds(i,j),k);
            }
        }
        Geo::barycenter(ctr, vertCrds, 3);
        Geo::barycenter(lagctr, vertLagCrds, 3);
        faceArea(i) = Geo::polygonArea(ctr, vertCrds, 3);
    }
    
    /// create new child Faces
    const Index crd_ins_pt = intr_crds.nh();
    for (int i=0; i<3; ++i) {
        intr_crds.insertHost(slice(faceCrds,i));
        intr_lagcrds.insertHost(slice(faceLagCrds,i));
        faces.insertHost(crd_ins_pt+i, ko::subview(newFaceVertInds,i, ko::ALL()), ko::subview(newFaceEdgeInds, i, ko::ALL()), faceInd, faceArea(i));
    }
    /// special case for child 3: reuse center particle memory
    const Index ctr_ind = faces.getCenterIndHost(faceInd);
    intr_crds.relocateHost(ctr_ind, ko::subview(faceCrds,3, ko::ALL()));
    intr_lagcrds.relocateHost(ctr_ind, ko::subview(faceLagCrds,3, ko::ALL()));
    faces.insertHost(ctr_ind, ko::subview(newFaceVertInds, 3, ko::ALL()), ko::subview(newFaceEdgeInds,3,ko::ALL()), faceInd, faceArea(3));
    
    /// Remove parent from leaf computations
    faces.setKidsHost(faceInd, newFaceKids);
    faces.setAreaHost(faceInd, 0.0);
    faces.decrementNActive();
}



/// ETI
template class Faces<TriFace>;
template class Faces<QuadFace>;

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;

}
