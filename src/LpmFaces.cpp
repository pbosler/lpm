#include "LpmFaces.hpp"

namespace Lpm {

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd, Faces<TriFace>& faces, Edges& edges, Coords<Geo>& crds, Coords<Geo>& lagcrds) const {
    LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
    LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");
    
    Index newFaceEdgeInds[4][3];
    Index newFaceVertInds[4][3];
    
    const Index face_insert_pt = faces.nh();
    const Index newFaceKids[4] = {face_insert_pt, face_insert_pt+1, face_insert_pt+2, face_insert_pt+3};
    
    Faces<TriFace>::const_host_ind_slice parentVertInds = faces.getConstVertsHost(faceInd);
    Faces<TriFace>::const_host_ind_slice parentEdgeInds = faces.getConstEdgesHost(faceInd);
    for (int i=0; i<3; ++i) {
        newFaceVertInds[i][i] = parentVertInds[i];
    }
    
    /// loop over parent edges
    for (int i=0; i<3; ++i) {
        const Index parentEdge = parentEdgeInds[i];
        Index edge_kids[2];
        if (edges.hasKidsHost(parentEdge)) {
            edges.getKidsHost(edge_kids, parentEdge);
        }
        else {
            edge_kids[0] = edges.nh();
            edge_kids[1] = edges.nh() + 1;
            
            edges.divide(parentEdge, crds, lagcrds);
        }
        
        if (faces.edgeHasPositiveOrientation(faceInd, i, edges)) {
        }
        else {
        }
    }
}



/// ETI
template class Faces<TriFace>;
template class Faces<QuadFace>;

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;

}
