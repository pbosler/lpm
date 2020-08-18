#ifndef LPM_SWE_HPP
#define LPM_SWE_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmGeometry.hpp"
#include "LpmVorticityGallery.hpp"
#include "LpmSWEGallery.hpp"
#include "LpmPolyMesh2dVtkInterface.hpp"
#include "LpmPolyMesh2dVtkInterface_Impl.hpp"

#include "Kokkos_Core.hpp"

#include <vector>
#include <sstream>

namespace Lpm {

template <typename SeedType> class ShallowWater : public PolyMesh2d<SeedType> {
  public:
    typedef scalar_view_type scalar_field;
    typedef ko::View<Real*[SeedType::geo::ndim]> vector_field;

    scalar_field relVortVerts;
    scalar_field potVortVerts;
    scalar_field divVerts;
    vector_field velocityVerts;
    scalar_field surfaceHeightVerts;
    scalar_field depthVerts;
    scalar_field topoVerts;

    scalar_field relVortFaces;
    scalar_field potVortFaces;
    scalar_field divFaces;
    scalar_field surfaceHeightFaces;
    scalar_field depthFaces;
    scalar_field topoFaces;
    scalar_field massFaces;
    vector_field velocityFaces;

    std::vector<scalar_field> scalar_tracer_verts;
    std::vector<scalar_field> scalar_tracer_faces;

    std::vector<vector_field> vector_tracer_verts;
    std::vector<vector_field> vector_tracer_faces;

    Real f0;
    Real beta;
    Real Omega;

    ShallowWater(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces,
       const Int& nq_scaler=0, const Int& nq_vector=0);

    template <typename Geo, typename CV> KOKKOS_INLINE_FUNCTION typename
    std::enable_if<std::is_same<Geo,PlaneGeometry>::value, Real>::type
    coriolis_f(const CV v) const {return f0 + beta*v[1];}

    template <typename Geo, typename CV> KOKKOS_INLINE_FUNCTION typename
    std::enable_if<std::is_same<Geo,SphereGeometry>::value, Real>::type
    coriolis_f(const CV v) const {return 2*Omega*v[2];}

    template <typename ProblemType>
    void set_bottom_topography();

    template <typename ProblemType>
    void init_problem();

    void set_velocity();

    inline Int nscalar_tracers() const {return scalar_tracer_faces.size();}

    inline Int nvector_tracers() const {return vector_tracer_faces.size();}

    void create_scalar_tracer(const std::string& name);

    void create_vector_tracer(const std::string& name);

    void addFieldsToVtk(Polymesh2dVtkInterface<SeedType>& vtk) const;

    inline void set_coriolis(const Real& f, const Real& b) {f0 = f; beta = b;}

    inline void set_coriolis(const Real& rot_rate) {Omega = rot_rate;}

    Real total_mass() const;
    Real total_mass_integral() const;

  protected:
    typedef typename scalar_field::HostMirror scalar_host;
    typedef typename vector_field::HostMirror vector_host;

    scalar_host host_relVortVerts;
    scalar_host host_potVortVerts;
    scalar_host host_divVerts;
    scalar_host host_sfcVerts;
    scalar_host host_depthVerts;
    scalar_host host_topoVerts;
    vector_host host_velocityVerts;

    scalar_host host_relVortFaces;
    scalar_host host_potVortFaces;
    scalar_host host_divFaces;
    scalar_host host_sfcFaces;
    scalar_host host_depthFaces;
    scalar_host host_topoFaces;
    scalar_host host_massFaces;
    vector_host host_velocityFaces;

    std::vector<scalar_host> host_scalar_tracer_verts;
    std::vector<scalar_host> host_scalar_tracer_faces;

    std::vector<vector_host> host_vector_tracer_verts;
    std::vector<vector_host> host_vector_tracer_faces;

};

}
#endif
