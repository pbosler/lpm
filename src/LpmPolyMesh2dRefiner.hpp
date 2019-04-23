#ifndef LPM_POLYMESH2D_REFINER_HPP
#define LPM_POLYMESH2D_REFINER_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"

namespace Lpm {

template <typename Geo> EdgeDivider {
    struct UnifTag {};
    struct AdaptiveTag {};
    
    ko::View<Real*[Geo::ndim],Dev> physCrds;
    ko::View<Real*[Geo::ndim],Dev> lagCrds;
    ko::View<Index*[4],Dev> edges;
    ko::View<Index*[3],Dev> edgeTree;
    Index nEdges;
    Index nVerts;
    
    KOKKOS_INLINE_FUNCTION
    EdgeDivider(ko::View<Real*[Geo::ndim],Dev> pc, ko::View<Real*[Geo::ndim],Dev> lc, 
        ko::View<Index*[4],Dev> es, ko::View<Index*[3],Dev> et, 
        const Index ne, const Index nl, const Index nv) : 
        physCrds(pc), lagCrds(lc), edges(es), edgeTree(et), nEdges(ne), nLeafEdges(nl), nVerts(nv) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const UnifTag&, const Index i) const {  
        const Index kid0Ind = nEdges + 2*i;
        const Index midptInd = nVerts + i;
        
        auto orig = ko::subview(physCrds, edges(i,0), ko::ALL());
        auto dest = ko::subview(physCrds, edges(i,1), ko::ALL());
        auto lagorig = ko::subview(lagCrds, edges(i,0), ko::ALL());
        auto lagdest = ko::subview(lagCrds, edges(i,1), ko::ALL());
        
        auto midpt = ko::subview(physCrds, midptInd, ko::ALL());
        auto lagmidpt = ko::subview(lagCrds, midptInd, ko::ALL());
        
        Geo::midpoint(midpt, orig, dest);
        Geo::midpoint(lagmidpt, lagorig, lagdest);

        edges(kid0Ind,0) = edges(i,0);  // kid 0: parent origin to new midpoint.
        edges(kid0Ind,1) = midptInd;
        edges(kid0Ind+1,0) = midptInd;  // kid 1: new midpoint to parent dest
        edges(kid0Ind+1,1) = edges(i,1);
        
        for (int j=0; j<2; ++j) { // loop over kids
            edges(kid0Ind+j,2) = edges(i,2);    // kids have same left face as parent
            edges(kid0Ind+j,3) = edges(i,3);    // kids have same right face as parent
            edgeTree(kid0Ind+j,0) = i;  // kids have same parent
            for (int k=1; k<3; ++k) { // loop over kids' kids
                edgeTree(kid0Ind+j, k) = NULL_IND;
            }    
        }
    }
};

template <typename Geo, typename FaceType> FaceDivider {
    struct UnifTag& {};
    struct AdaptiveTag& {};
    
    ko::View<Real*[Geo::ndim],Dev> physVertCrds;
    ko::View<Real*[Geo::ndim],Dev> lagVertCrds;
    Index nVerts;
    
    ko::View<Index*[4],Dev> edges;
    ko::View<Index*[3],Dev> edgeTree;
    Index nEdges;
    
    ko::View<Real*[Geo::ndim],Dev> physFaceCrds;
    ko::View<Real*[Geo::ndim],Dev> lagFaceCrds;
    ko::View<Index*[FaceType::nverts],Dev> faceVerts;
    ko::View<Index*[FaceType::nverts],Dev> faceEdges;
    ko::View<Index*[5],Dev> faceTree;
    Index nFaces;
    
    FaceDivider(ko::View<Real*[Geo::ndim],Dev> pvc, ko::View<Real*[Geo::ndim],Dev> lvc, ko::View<Index*[4],Dev> es,
        ko::View<Index*[3],Dev> et, ko::View<Real*[Geo::ndim],Dev> pfc, ko::View<Real*[Geo::ndim],Dev> lfc, 
        ko::View<Index*[FaceType::nverts],Dev> fv, ko::View<Index*[FaceType::nverts],Dev> fe, ko::View<Index*[5],Dev> ft,
        const Index nv, const Index ne, const Index nf) : physVertCrds(pvc), lagVertCrds(lvc), edges(es), edgeTree(et),
        physFaceCrds(pfc), lagFaceCrds(lfc), faceVerts(fv), faceEdges(fe), faceTree(ft), nVerts(nv), nEdges(ne), nFaces(nf) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const UnifTag&, const Index i) const {}
};



}
#endif
