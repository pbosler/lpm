#ifndef LPM_POLYMESH_2D
#define LPM_POLYMESH_2D

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

struct TriFace {
    static constexpr Int nverts = 3;
};

struct QuadFace {
    static constexpr Int nverts = 4;
};

template <typename FaceType, typename SeedType> class PolyMesh2d {
    public:
        typedef typename SeedType::geo geo;
        typedef ko::View<Real*[geo::ndim],Dev> crd_view_type; // view(i,:) = position vector of particle i
        typedef ko::View<Index,Dev> n_view_type;        // view(0) = n
        typedef ko::View<Index*[4],Dev> edge_view_type; // view(i,:) = (orig, dest, left, right)
        typedef ko::View<Index*[3],Dev> edge_tree_type; // view(i,:) = (parent, kid0, kid1)
        typedef ko::View<Index*[FaceType::nverts],Dev> face_view_type; // view(i,:) = (vert0, vert1, vert2...)
        typedef ko::View<Index*[5],Dev> face_tree_type; // view(i,:) = (parent, kid0, kid1, kid2, kid3)
        
        PolyMesh2d(const int initTreeDepth, const int maxTreeDepth);
        
        virtual ~PolyMesh2d() {}
    
    protected:
        crd_view_type vertCrds;
        crd_view_type lagVertCrds;
        crd_view_type faceCrds;
        crd_view_type lagFaceCrds;
        n_view_type nVerts;
        n_view_type nFaces;
        
        edge_view_type edges;
        edge_tree_type edgeTree;
        n_view_type nEdges;
        n_view_type nLeafEdges;
        
        face_view_type faceVerts;
        face_view_type faceEdges;
        face_tree_type faceTree;
        n_view_type nLeafFaces;
};

}
#endif
