#include "LpmFaces.hpp"

namespace Lpm {

template <typename FaceKind>
void Faces<FaceKind>::insertHost(const Index* vertinds, const Index* edgeinds, const Index prt, const Real ar) {
    LPM_THROW_IF(_nh(0)+1 > _nmax, "Faces::insert error: not enough memory.");
    const Index ins = _nh(0);
    for (int i=0; i<FaceKind::nverts; ++i) {
        _hostverts(ins, i) = vertinds[i];
        _hostedges(ins, i) = edgeinds[i];
    }
    for (int i=0; i<4; ++i) {
        _hostkids(ins, i) = NULL_IND;
    }
    _hostparent(ins) = prt;
    _hostarea(ins) = ar;
    _nh(0) += 1;
}

template <typename FaceKind>
void Faces<FaceKind>::setKids(const Index parent, const Index* kids) {
    for (int i=0; i<4; ++i) {
        _hostkids(parent, i) = kids[i];
    }
}

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd, Faces<TriFace>& faces, Edges& edges, Coords<Geo>& crds, Coords<Geo>& lagcrds) const {
    LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
    LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");
    
    Index newFaceEdgeInds[4][3];  // (child face index, edge index)
    Index newFaceVertInds[4][3];  // (child face index, vertex index)
    
    const Index face_insert_pt = faces.nh();
    const Index newFaceKids[4] = {face_insert_pt, face_insert_pt+1, face_insert_pt+2, face_insert_pt+3};
    
    Faces<TriFace>::const_host_ind_slice parentVertInds = faces.getConstVertsHost(faceInd);
    Faces<TriFace>::const_host_ind_slice parentEdgeInds = faces.getConstEdgesHost(faceInd);
    for (int i=0; i<3; ++i) {
        newFaceVertInds[i][i] = parentVertInds[i];
    }
    
    /// loop over parent edges, replace with child edges
    for (int i=0; i<3; ++i) {
        const Index parentEdge = parentEdgeInds[i];
        Index edge_kids[2];
        if (edges.hasKidsHost(parentEdge)) {
            edges.getKidsHost(edge_kids, parentEdge);
        }
        else {
            edge_kids[0] = edges.nh();
            edge_kids[1] = edges.nh() + 1;
            
            edges.divide<Geo>(parentEdge, crds, lagcrds);
        }
        
        if (faces.edgeHasPositiveOrientation(faceInd, i, edges)) {
            newFaceEdgeInds[i][i] = edge_kids[0];
            newFaceEdgeInds[(i+1)%3][i] = edge_kids[1];
            
            edges.setLeft(edge_kids[0], newFaceKids[i]);
            edges.setLeft(edge_kids[1], newFaceKids[(i+1)%3]);
        }
        else {
            newFaceEdgeInds[i][i] = edge_kids[1];
            newFaceEdgeInds[(i+1)%3][i] = edge_kids[0];
            
            edges.setRight(edge_kids[1], newFaceKids[i]);
            edges.setRight(edge_kids[0], newFaceKids[(i+1)%3]);
        }
        
        const Index midVertInd = edges.getDestHost(edge_kids[1]);
        if (i==0) {
            newFaceVertInds[0][1] = midVertInd;
            newFaceVertInds[1][0] = midVertInd;
            newFaceVertInds[3][2] = midVertInd;
        }
        else if (i==1) {
            newFaceVertInds[1][2] = midVertInd;
            newFaceVertInds[2][1] = midVertInd;
            newFaceVertInds[3][0] = midVertInd;
        }
        else {
            newFaceVertInds[2][0] = midVertInd;
            newFaceVertInds[0][2] = midVertInd;
            newFaceVertInds[3][1] = midVertInd;
        }
    }
    
    /// add new interior Edges
    const Index edge_insert_pt = edges.nh();
    for (int i=0; i<3; ++i) {
        newFaceEdgeInds[3][i] = edge_insert_pt + i;
    }
    newFaceEdgeInds[0][1] = edge_insert_pt+1;
    newFaceEdgeInds[1][2] = edge_insert_pt+2;
    newFaceEdgeInds[2][0] = edge_insert_pt;
    edges.insertHost(newFaceVertInds[2][1], newFaceVertInds[2][0], newFaceKids[3], newFaceKids[2]);
    edges.insertHost(newFaceVertInds[0][2], newFaceVertInds[0][1], newFaceKids[3], newFaceKids[0]);
    edges.insertHost(newFaceVertInds[1][0], newFaceVertInds[1][2], newFaceKids[3], newFaceKids[1]);
    
    /// add new child faces
    for (int i=0; i<4; ++i) {
        Real vert_crds[3][Geo::ndim];
        for (int j=0; j<3; ++j) {
            typename Coords<Geo>::const_host_slice vec = crds.getConstSlice(newFaceVertInds[i][j]);
            for (int k=0; k<Geo::ndim; ++k) {
                vert_crds[j][k] = vec[k];
            }
        }
        Real ctr[Geo::ndim]; 
        Geo::barycenter<Real*, Real**>(ctr, vert_crds, 3);
        const Real area = Geo::polygonArea(ctr, vert_crds, 3);
        faces.insertHost(newFaceVertInds[i], newFaceVertInds[i], faceInd, area);
    }
    faces.setKids(faceInd, newFaceKids);
    faces.setAreaHost(faceInd, 0.0);
}



/// ETI
template class Faces<TriFace>;
template class Faces<QuadFace>;

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;

}
