#ifndef LPM_BVE_SPHERE_HPP
#define LPM_BVE_SPHERE_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"
#include "LpmBVEKernels.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmPolyMesh2dVtkInterface.hpp"
#include "LpmPolyMesh2dVtkInterface_Impl.hpp"
#include <vector>
#include <sstream>

namespace Lpm {

template <typename SeedType> class BVESphere : public PolyMesh2d<SeedType> {
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
        n_view_type ntracers;

        Real Omega;
        Real t;

        std::vector<scalar_field> tracer_verts;
        std::vector<scalar_field> tracer_faces;

        BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const Int nq=0);

        void init_vorticity(const VorticityInitialCondition::ptr relvort);

        void outputVtk(const std::string& fname) const override;

        void updateDevice() const override;
        void updateHost() const override;

        void set_omega(const Real& omg);

        Short create_tracer(const std::string& name);

        Real avg_mesh_size_radians() const;

        Real avg_mesh_size_degrees() const;

        void addFieldsToVtk(Polymesh2dVtkInterface<SeedType>& vtk) const;

    protected:
        typedef typename scalar_field::HostMirror scalar_host;
        typedef typename vector_field::HostMirror vector_host;
        typedef typename n_view_type::HostMirror n_host;

        n_host _hostntracers;

        scalar_host _hostRelVortVerts;
        scalar_host _hostAbsVortVerts;
        scalar_host _hostStreamFnVerts;
        vector_host _hostVelocityVerts;

        scalar_host _hostRelVortFaces;
        scalar_host _hostAbsVortFaces;
        scalar_host _hostStreamFnFaces;
        vector_host _hostVelocityFaces;

        std::vector<scalar_host> _hostTracerVerts;
        std::vector<scalar_host> _hostTracerFaces;

        bool omg_set;
};

}
#endif
