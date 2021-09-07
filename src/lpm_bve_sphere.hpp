#ifndef LPM_BVE_SPHERE_HPP
#define LPM_BVE_SPHERE_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include "Kokkos_Core.hpp"

#include <vector>
#include <sstream>

namespace Lpm {

/** @brief Barotropic Vorticity Equation Solver: meshed particles class


*/
template <typename SeedType> class BVESphere : public PolyMesh2d<SeedType> {
    public:
        static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
          "Spherical mesh seed required.");

        typedef scalar_view_type scalar_field;
        typedef ko::View<Real*[3],Dev> vector_field;
        typedef SeedType seed_type;
        typedef typename SeedType::geo Geo;
        typedef typename SeedType::faceKind FaceType;
        typedef Coords<Geo> coords_type;
        typedef std::shared_ptr<Coords<Geo>> coords_ptr;

        scalar_field rel_vort_verts; /// relative vorticity, passive particles [1/time]
        scalar_field abs_vort_verts; /// absolute vorticity, passive particles [1/time]
        scalar_field stream_fn_verts; /// stream function, passive particles [length^2/time]
        vector_field velocity_verts; /// velocity, passive particles [length/time]

        scalar_field rel_vort_faces;  /// relative vorticity, active particles
        scalar_field abs_vort_faces;  /// absolute vorticity, active particles
        scalar_field stream_fn_faces; /// stream function, active particles
        vector_field velocity_faces; /// velocity, active particles
        n_view_type ntracers; /// number of passive tracers

        Real Omega; /// background rotation rate of sphere about positive z-axis
        Real t; /// time

        std::vector<scalar_field> tracer_verts; /// passive tracers at passive particles
        std::vector<scalar_field> tracer_faces; /// passive tracers at active particles

        /** @brief Constructor.  Allocates memory; does not initialize problem data.

          @param [in] nmaxverts: number of vertices to should be allocate
          @param [in] nmaxedges: number of edges to allocate
          @param [in] nmaxfaces: number of faces to allocate
          @param [in] nq: number of passive tracers to allocate
        */
        BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const Int nq=0);

         /** @brief Constructor.  Allocates memory; does not initialize problem data.

          @param [in] nmaxverts: number of vertices to should be allocate
          @param [in] nmaxedges: number of edges to allocate
          @param [in] nmaxfaces: number of faces to allocate
          @param [in] tracers names of passive tracers to allocate
        */
        BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const
          std::vector<std::string>& tracers);

        /** @breif Initialize vorticity on all particles.

          @param relvort: relative vorticity functor.
        */
        template <typename VorticityInitialCondition>
        void init_vorticity(const VorticityInitialCondition& vorticity_fn);

        void init_velocity();

        void init_stream_fn();

        void update_device() const override;

        void update_host() const override;

        void set_omega(const Real& omg);

        Real avg_mesh_size_radians() const;

        Real avg_mesh_size_degrees() const;

    protected:
        typedef typename scalar_field::HostMirror scalar_host;
        typedef typename vector_field::HostMirror vector_host;
        typedef typename n_view_type::HostMirror n_host;

        n_host _host_ntracers;

        scalar_host _host_rel_vort_verts;
        scalar_host _host_abs_vort_verts;
        scalar_host _host_stream_fn_verts;
        vector_host _host_velocity_verts;

        scalar_host _host_rel_vort_faces;
        scalar_host _host_abs_vort_faces;
        scalar_host _host_stream_fn_faces;
        vector_host _host_velocity_faces;

        std::vector<scalar_host> _host_tracer_verts;
        std::vector<scalar_host> _host_tracer_faces;

        bool omg_set;
};

template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(const std::shared_ptr<BVESphere<SeedType>> bve);

}
#endif
