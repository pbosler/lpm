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

/**
    PolyMesh data structure, defined on device.
    
    Vertices:
        vertCrds, lagVertCrds: coordinate vectors in physical and Lagrangian space, respectively.  
            vertCrds(i,:) = coordinates of particle i.
        nVerts = number of vertices currently in memory.  indices >= nVerts are uninitialized.
    
    Edges:
        edges: edge data structure such that:
            edges(i,0) = origin index (to vertCrds)
            edges(i,1) = destination index (to vertCrds)
            edges(i,2) = left face index (to faces)
            edges(i,3) = right face index (to faces)
        edgeTree: binary tree, indices to edges such that: 
            edgeTree(i,0) = parent of edge i
            edgeTree(i,1:2) = children of edge i
        nEdges = number of edges currently in memory. indices >= nEdges are uninitialized
        nLeafEdges = number of undivided edges in tree
    
    Faces:
        faceVerts: indices to vertCrds, lagVertCrds such that:
             faceVerts(i,:) = face i, vertices 0, 1, 2, ... nvertsperface-1
        faceEdges: indices to edges such that:
             faceEdges(i,:) = face i, edges 0, 1, 2, ... nvertsperface-1
        faceTree: quadtree, indices to faces such that:
             faceTree(i,0) = parent of face i, faceTree(i,1:4) = children of face i
        faceCrds: coordinate vectors of faces' interior particles.  For standard face types, 
            faceCrds(i,:) is the coordinate vector of the center particle assocated with face i.
        nFaces = number of faces currently in memory. indices >= nFaces are uninitialized
        nLeafFaces = number of undivided faces in tree
*/
template <typename FaceType, typename SeedType> class PolyMesh2d {
    public:
        typedef typename SeedType::geo geo;
        typedef ko::View<Real*[geo::ndim],Dev> crd_view_type; // view(i,:) = position vector of particle i
        typedef ko::View<Index,Dev> n_view_type;        // view(0) = n
        typedef ko::View<Index*[4],Dev> edge_view_type; // view(i,:) = (orig, dest, left, right)
        typedef ko::View<Index*[3],Dev> edge_tree_type; // view(i,:) = (parent, kid0, kid1)
        typedef ko::View<Index*[FaceType::nverts],Dev> face_view_type; // view(i,:) = verts of face i, (vert0, vert1, vert2...)
        typedef ko::View<Index*[5],Dev> face_tree_type; // view(i,:) = (parent, kid0, kid1, kid2, kid3)
        
        PolyMesh2d(const int initTreeDepth, const int maxTreeDepth);
        
        virtual ~PolyMesh2d() {}
        
        struct FaceDivider {
        };
        
        struct EdgeDivider {
        };
        
    protected:
        crd_view_type vertCrds;
        crd_view_type lagVertCrds;
        n_view_type nVerts;
        
        edge_view_type edges;
        edge_tree_type edgeTree;
        n_view_type nEdges;
        n_view_type nLeafEdges;
        
        crd_view_type faceCrds;
        crd_view_type lagFaceCrds;
        face_view_type faceVerts;
        face_view_type faceEdges;
        face_tree_type faceTree;
        n_view_type nFaces;
        n_view_type nLeafFaces;
        
        void seedInit();
};

}
#endif
