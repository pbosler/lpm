#ifndef LPM_POLYMESH_2D
#define LPM_POLYMESH_2D

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/**
*/
template <typename Geo, typename FaceType> class PolyMesh2d {
    public:
        typedef Geo geo;
        typedef FaceType facekind;
    
        PolyMesh2d(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) : 
            physVerts(nmaxverts), lagVerts(nmaxverts), edges(nmaxedges), faces(nmaxfaces), 
            physFaces(nmaxfaces), lagFaces(nmaxfaces) {}
        
        virtual ~PolyMesh2d() {}
        
        Int baseTreeDepth;
        
        KOKKOS_INLINE_FUNCTION
        Index nverts() const {return physVerts.n();}
        
        KOKKOS_INLINE_FUNCTION
        Index nfaces() const {return faces.n();}
        
        Coords<Geo> physVerts;
        Coords<Geo> lagVerts;
        
        Edges edges;
        
        Faces<FaceType> faces;
        Coords<Geo> physFaces;
        Coords<Geo> lagFaces;
        
        template <typename SeedType>
        void treeInit(const Int initDepth, const MeshSeed<SeedType>& seed);
        
        virtual void outputVtk(const std::string& fname) const;
        
        virtual void updateDevice() const;
        
        virtual void updateHost() const;
        
    protected:
        typedef FaceDivider<Geo,FaceType> divider;
    
        template <typename SeedType>
        void seedInit(const MeshSeed<SeedType>& seed);
        
        
};

}
#endif
