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
        

        typename Coords<Geo>::crd_view_type getVertCrds() const { 
            return typename Coords<Geo>::crd_view_type(physVerts.crds, std::make_pair(0,physVerts.nh()), ko::ALL());}
        
        typename Coords<Geo>::crd_view_type getFaceCrds() const {
            return typename Coords<Geo>::crd_view_type(physFaces.crds, std::make_pair(0,faces.nh()), ko::ALL());}
        
//         void makeFacemask() const {faces.makeMask();}
        
        mask_view_type getFacemask() const {
            return mask_view_type(faces.mask, std::make_pair(0,faces.nh()));}
        
        typename mask_view_type::HostMirror getFacemaskHost() const {return ko::create_mirror_view(faces.mask);}
        
        scalar_view_type getFaceArea() const {
            return scalar_view_type(faces.area, std::make_pair(0,faces.nh()));}
        
        KOKKOS_INLINE_FUNCTION
        Index nverts() const {return physVerts.n();}
        
        KOKKOS_INLINE_FUNCTION
        Index nfaces() const {return faces.n();}

        Index nvertsHost() const {return physVerts.nh();}
        Index nfacesHost() const {return faces.nh();}
        
        Coords<Geo> physVerts;
        Coords<Geo> lagVerts;
        
        Edges edges;
        
        Faces<FaceType> faces;
        Coords<Geo> physFaces;
        Coords<Geo> lagFaces;
        
        typename Coords<Geo>::crd_view_type::HostMirror getFaceCrdsHost() {return physFaces.getHostCrdView();}
        
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
