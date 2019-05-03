#ifndef LPM_BVE_SPHERE_HPP
#define LPM_BVE_SPHERE_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"
#include "LpmBVEKernels.hpp"

namespace Lpm {

template <typename FaceType> class BVESphere : public PolyMesh2d<SphereGeometry, FaceType> {
    public:
        typedef scalar_view_type scalar_field;
        typedef ko::View<Real*[3],Dev> vector_field;
        
        scalar_field relVortVerts;
        scalar_field absVortVerts;
        scalar_field streamFnVerts;
        vector_field velocityVerts;
        
        scalar_field relVortFaces;
        scalar_field absVortFaces;
        scalar_field streamFnFaces;
        vector_field velocityFaces;
        
        BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) : 
            PolyMesh2d<SphereGeometry,FaceType>(nmaxverts, nmaxedges, nmaxfaces), 
            relVortVerts("relVortVerts", nmaxverts), absVortVerts("absVortVerts",nmaxverts), 
            streamFnVerts("streamFnVerts", nmaxverts), velocityVerts("velocityVerts", nmaxverts),
            relVortFaces("relVortFaces", nmaxfaces), absVortFaces("absVortFaces", nmaxfaces), 
            streamFnFaces("streamFnFaces", nmaxfaces), velocityFaces("velocityFaces", nmaxfaces) {
            _hostRelVortVerts = ko::create_mirror_view(relVortVerts);
            _hostAbsVortVerts = ko::create_mirror_view(absVortVerts);
            _hostStreamFnVerts = ko::create_mirror_view(streamFnVerts);
            _hostVelocityVerts = ko::create_mirror_view(velocityVerts);
            _hostRelVortFaces = ko::create_mirror_view(relVortFaces);
            _hostAbsVortFaces = ko::create_mirror_view(absVortFaces);
            _hostStreamFnFaces = ko::create_mirror_view(streamFnFaces);
            _hostVelocityFaces = ko::create_mirror_view(velocityFaces);
        }
        
        KOKKOS_INLINE_FUNCTION
        vector_field getVertexVelocity() const {return vector_field(velocityVerts, std::make_pair(0,this->nverts()), ko::ALL());}
        
        KOKKOS_INLINE_FUNCTION
        scalar_field getVertexVorticity() const {return scalar_field(relVortVerts, std::make_pair(0,this->nverts()));}
        
        KOKKOS_INLINE_FUNCTION
        vector_field getFaceVelocity() const {return vector_field(velocityFaces, std::make_pair(0,this->nfaces()), ko::ALL());}
        
        KOKKOS_INLINE_FUNCTION
        scalar_field getFaceVorticity() const {return scalar_field(relVortFaces, std::make_pair(0,this->nfaces()));}
        
        void outputVtk(const std::string& fname) const override;

        template <typename InitFunctor>
        void initProblem() { 
            InitFunctor setup(*this);
            setup.init();
            updateHost();
        }

        void updateDevice() const override;
        void updateHost() const override;
        
    protected:
        typedef typename scalar_field::HostMirror scalar_host;
        typedef typename vector_field::HostMirror vector_host;
        
        scalar_host _hostRelVortVerts;
        scalar_host _hostAbsVortVerts;
        scalar_host _hostStreamFnVerts;
        vector_host _hostVelocityVerts;
        
        scalar_host _hostRelVortFaces;
        scalar_host _hostAbsVortFaces;
        scalar_host _hostStreamFnFaces;
        vector_host _hostVelocityFaces;
};

}
#endif
