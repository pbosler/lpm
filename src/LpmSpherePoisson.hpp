#ifndef LPM_SPHERE_POISSON_HPP
#define LPM_SPHERE_POISSON_HPP

#include <cmath>
#include <iomanip>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmRossbyWaves.hpp"
#include "LpmVtkIO.hpp"

namespace Lpm {

typedef typename SphereGeometry::crd_view_type crd_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef typename ko::TeamPolicy<>::member_type member_type;

/** @brief Evaluates the Poisson equation's Green's function for the
  SpherePoisson

  Returns \f$ g(x,y)f(y)A(y),\f$ where \f$ g(x,y) = -log(1-x\cdot y)/(4\pi) \f$.

  @param psi Output value --- potential response due to a source of strength
  src_f*src_area
  @param tgt_x Coordinate of target location
  @param src_x Source location
  @param src_f Source (e.g., vorticity) value
  @param src_area Area of source panel
*/
template <typename VecType>
KOKKOS_INLINE_FUNCTION void greensFn(Real& psi, const VecType& tgt_x,
                                     const VecType& src_xx, const Real& src_f,
                                     const Real& src_area) {
  psi = -std::log(1.0 - SphereGeometry::dot(tgt_x, src_xx)) * src_f * src_area /
        (4 * PI);
}

/** @brief Computes the spherical Biot-Savart kernel's contribution to velocity
  for a single source

  Returns \f$ K(x,y)f(y)A(y)\f$, where \f$K(x,y) = \nabla g(x,y)\times x =
  \frac{x \times y}{4\pi(1-x\cdot y)}\f$.

  @param psi Output value --- potential response due to a source of strength
  src_f*src_area
  @param tgt_x Coordinate of target location
  @param src_x Source location
  @param src_f Source (e.g., vorticity) value
  @param src_area Area of source panel
*/
template <typename VecType>
KOKKOS_INLINE_FUNCTION void biotSavart(ko::Tuple<Real, 3>& u,
                                       const VecType& tgt_x,
                                       const VecType& src_xx, const Real& src_f,
                                       const Real& src_area) {
  u = SphereGeometry::cross(tgt_x, src_xx);
  const Real str =
      -src_f * src_area / (4 * PI * (1 - SphereGeometry::dot(tgt_x, src_xx)));
  for (int j = 0; j < 3; ++j) {
    u[j] *= str;
  }
}

/** @brief Initializes vorticity on the sphere, and computes exact velocity and
   stream function values.

*/
struct Init {
  scalar_view_type f;  ///< output view holds vorticity values
  scalar_view_type
      exactpsi;     ///< output view holds exact stream function values
  vec_view exactu;  ///< output view holds exact velocity values
  crd_view x;       ///< input view of coordinates

  Init(scalar_view_type ff, scalar_view_type psi, vec_view u, crd_view xx)
      : f(ff), exactpsi(psi), exactu(u), x(xx) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    auto myx = ko::subview(x, i, ko::ALL());
    const Real myf = SphHarm54(myx);
    f(i) = myf;
    exactpsi(i) = myf / 30.0;
    const ko::Tuple<Real, 3> u = RH54Velocity(myx);
    for (int j = 0; j < 3; ++j) {
      exactu(i, j) = u[j];
    }
  }
};

/** Stream function reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$

*/
struct ReduceDistinct {
  typedef Real value_type;  ///< required by kokkos for custom reducers
  Index i;                  ///< index of target point in tgtx view
  crd_view tgtx;  ///< view holding coordinates of target locations (usually,
                  ///< vertices)
  crd_view srcx;  ///< view holding coordinates of source locations (usually,
                  ///< face centers)
  scalar_view_type srcf;  ///< view holding RHS (vorticity) data
  scalar_view_type srca;  ///< view holding panel areas
  mask_view_type
      facemask;  ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  ReduceDistinct(const Index& ii, crd_view x, crd_view xx, scalar_view_type f,
                 scalar_view_type a, mask_view_type fm)
      : i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& pot) const {
    Real potential = 0;
    if (!facemask(j)) {
      auto mytgt = ko::subview(tgtx, i, ko::ALL());
      auto mysrc = ko::subview(srcx, j, ko::ALL());
      greensFn(potential, mytgt, mysrc, srcf(j), srca(j));
    }
    pot += potential;
  }
};

/** Velocity reduction kernel for distinct sets of points on the sphere,
   i.e., \f$x \ne y ~ \forall x\in\text{src_x},~y\in\text{src_y}\f$
*/
struct UReduceDistinct {
  typedef ko::Tuple<Real, 3>
      value_type;  ///< required by kokkos for custom reducers
  Index i;         ///< index of target point in tgtx view
  crd_view tgtx;   ///< view holding coordinates of target locations (usually,
                   ///< vertices)
  crd_view srcx;   ///< view holding coordinates of source locations (usually,
                   ///< face centers)
  scalar_view_type srcf;  ///< view holding RHS (vorticity) data
  scalar_view_type srca;  ///< view holding panel areas
  mask_view_type
      facemask;  ///< mask to exclude divided panels from the computation.

  KOKKOS_INLINE_FUNCTION
  UReduceDistinct(const Index& ii, crd_view x, crd_view xx, scalar_view_type f,
                  scalar_view_type a, mask_view_type fm)
      : i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& vel) const {
    ko::Tuple<Real, 3> u;
    if (!facemask(j)) {
      auto mytgt = ko::subview(tgtx, i, ko::ALL());
      auto mysrc = ko::subview(srcx, j, ko::ALL());
      biotSavart(u, mytgt, mysrc, srcf(j), srca(j));
    }
    vel += u;
  }
};

/** @brief Solves the Poisson equation at panel vertices.


 @device

 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function
 and 1 for velocity

*/
struct VertexSolve {
  crd_view vertx;            ///< [input] target coordinates
  crd_view facex;            ///< [input] source coordinates
  scalar_view_type facef;    ///< [input] source vorticity
  scalar_view_type facea;    ///< [input] source area
  mask_view_type facemask;   ///< [input] source mask (prevent divided panels
                             ///< from contributing to sums)
  scalar_view_type vertpsi;  ///< [output] stream function values
  vec_view vertu;            ///< [output] velocity values
  Index nf;                  ///< [input] number of sources

  VertexSolve(crd_view vx, crd_view fx, scalar_view_type ff,
              scalar_view_type fa, mask_view_type fm, scalar_view_type vpsi,
              vec_view vu)
      : vertx(vx),
        facex(fx),
        facef(ff),
        facea(fa),
        facemask(fm),
        vertpsi(vpsi),
        vertu(vu),
        nf(fx.extent(0)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
                        ReduceDistinct(i, vertx, facex, facef, facea, facemask),
                        p);
    vertpsi(i) = p;
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(
        ko::TeamThreadRange(mbr, nf),
        UReduceDistinct(i, vertx, facex, facef, facea, facemask), u);
    for (int j = 0; j < 3; ++j) {
      vertu(i, j) = u[j];
    }
  }
};

/** Stream function reduction kernel for collocated source and target sets of
   points on the sphere,

*/
struct ReduceCollocated {
  typedef Real value_type;  ///< required by kokkos for custom reducers
  Index i;                  ///< index of target coordinate vector
  crd_view srcx;            ///< collection of source coordinate vectors
  scalar_view_type srcf;    ///< source vorticity values
  scalar_view_type srca;    ///< panel areas
  mask_view_type mask;      ///< mask (excludes non-leaf faces)

  KOKKOS_INLINE_FUNCTION
  ReduceCollocated(const Index& ii, crd_view x, scalar_view_type f,
                   scalar_view_type a, mask_view_type m)
      : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& pot) const {
    Real potential = 0;
    if (!mask(j) && i != j) {
      auto mtgt = ko::subview(srcx, i, ko::ALL());
      auto msrc = ko::subview(srcx, j, ko::ALL());
      greensFn(potential, mtgt, msrc, srcf(j), srca(j));
    }
    pot += potential;
  }
};

/** Velocity reduction kernel for  collocated source and target sets of points
   on the sphere,

*/
struct UReduceCollocated {
  typedef ko::Tuple<Real, 3>
      value_type;         ///< required by kokkos for custom reducers
  Index i;                ///< index of target coordinate vector
  crd_view srcx;          ///< collection of source coordinates
  scalar_view_type srcf;  ///< source vorticity
  scalar_view_type srca;  ///< source areas
  mask_view_type mask;    ///< mask (excludes non-leaf sources)

  KOKKOS_INLINE_FUNCTION
  UReduceCollocated(const Index& ii, crd_view x, scalar_view_type f,
                    scalar_view_type a, mask_view_type m)
      : i(ii), srcx(x), srcf(f), srca(a), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index& j, value_type& vel) const {
    ko::Tuple<Real, 3> u;
    if (!mask(j) && i != j) {
      auto mtgt = ko::subview(srcx, i, ko::ALL());
      auto msrc = ko::subview(srcx, j, ko::ALL());
      biotSavart(u, mtgt, msrc, srcf(j), srca(j));
    }
    vel += u;
  }
};

/** @brief Solves the Poisson equation at panel centers.

 @device

 @par Parallel pattern:
 1 thread team per target site performs two reductions -- 1 for stream function
 and 1 for velocity

*/
struct FaceSolve {
  crd_view facex;
  scalar_view_type facef;
  scalar_view_type facea;
  mask_view_type facemask;
  scalar_view_type facepsi;
  vec_view faceu;
  Index nf;

  FaceSolve(crd_view x, scalar_view_type f, scalar_view_type a,
            mask_view_type m, scalar_view_type p, vec_view u)
      : facex(x),
        facef(f),
        facea(a),
        facemask(m),
        facepsi(p),
        faceu(u),
        nf(x.extent(0)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real p;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
                        ReduceCollocated(i, facex, facef, facea, facemask), p);
    facepsi(i) = p;
    ko::Tuple<Real, 3> u;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nf),
                        UReduceCollocated(i, facex, facef, facea, facemask), u);
    for (int j = 0; j < 3; ++j) {
      faceu(i, j) = u[j];
    }
  }
};

/** @brief custom reducer for computing relative error in scalar fields

  @todo move to its own header


*/
struct Error {
  typedef Real value_type;     ///< required by kokkos for custom reducers
  scalar_view_type fcomputed;  ///< [input] computed values with possible error
  scalar_view_type fexact;     ///< [input] exact values
  scalar_view_type ferror;     ///< [output] abs(computed-exact)
  mask_view_type mask;         ///< [input] exclude non-leaves

  struct VertTag {};
  struct FaceTag {};
  struct MaxTag {};

  Error(scalar_view_type e, scalar_view_type fc, scalar_view_type fe,
        mask_view_type m)
      : ferror(e), fcomputed(fc), fexact(fe), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const VertTag&, const Index& i) const {
    ferror(i) = abs(fcomputed(i) - fexact(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const FaceTag&, const Index& i) const {
    ferror(i) = (mask(i) ? 0 : abs(fcomputed(i) - fexact(i)));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const MaxTag&, const Index& i, value_type& max) const {
    if (ferror(i) > max) max = ferror(i);
  }
};

/** @brief custom reducer for computing relative error in vector fields

  @todo move to its own header


*/
struct UError {
  typedef Real value_type;  ///< required by kokkos for custom reducers
  vec_view uerror;          ///< [output] abs(computed-exact)
  vec_view ucomputed;       ///< [input] computed values with possible error
  vec_view uexact;          ///< [input] exact values
  mask_view_type mask;      ///< [input] exclude non-leaves

  struct VertTag {};
  struct FaceTag {};
  struct MaxTag {};

  UError(vec_view e, vec_view uc, vec_view ue, mask_view_type m)
      : uerror(e), ucomputed(uc), uexact(ue), mask(m) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const VertTag&, const Index& i) const {
    for (int j = 0; j < 3; ++j) {
      uerror(i, j) = abs(ucomputed(i, j) - uexact(i, j));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const FaceTag&, const Index& i) const {
    if (mask(i)) {
      for (int j = 0; j < 3; ++j) {
        uerror(i, j) = 0.0;
      }
    } else {
      for (int j = 0; j < 3; ++j) {
        uerror(i, j) = abs(ucomputed(i, j) - uexact(i, j));
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const MaxTag&, const Index& i, value_type& max) const {
    Real emag = 0;
    for (int j = 0; j < 3; ++j) {
      emag += square(uerror(i, j));
    }
    if (emag > max) max = emag;
  }
};

/** @brief Solves the Poisson equation on the sphere, \f$-\nabla^2\psi = f\f$.

    Adds data fields to PolyMesh2d.

*/
template <typename SeedType>
class SpherePoisson : public PolyMesh2d<SeedType> {
 public:
  typedef typename SeedType::faceKind FaceKind;
  scalar_view_type fverts;         ///< RHS of Poisson equation at vertices
  scalar_view_type psiverts;       ///< Stream function (solution) at vertices
  scalar_view_type psiexactverts;  ///< Exact solution at vertices
  scalar_view_type ffaces;         ///< RHS of Poisson equation at faces
  scalar_view_type psifaces;       ///< Stream function (solution) at faces
  scalar_view_type psiexactfaces;  ///< Exact solution at faces
  scalar_view_type everts;         ///< error at vertices
  scalar_view_type efaces;         ///< error at faces
  vec_view uverts;                 ///< velocity computed at vertices
  vec_view ufaces;                 ///< velocity computed at faces
  vec_view uvertsexact;            ///< exact velocity at vertices
  vec_view ufacesexact;            ///< exact velocity at faces
  vec_view uerrverts;              ///< velocity error at vertices
  vec_view uerrfaces;              ///< velocity error at faces
  Real linf_verts;   ///< linf relative error in \f$\psi\f$ at vertices
  Real linf_faces;   ///< linf relative error in \f$\psi\f$ at faces
  Real linf_uverts;  ///< linf relative error in velocity at vertices
  Real linf_ufaces;  ///< linf relative error in velocity at faces

  /** @brief Constructor.  Same interface as PolyMesh2d.
   */
  SpherePoisson(const Index nmaxverts, const Index nmaxedges,
                const Index nmaxfaces)
      : PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
        fverts("f", nmaxverts),
        psiverts("psi", nmaxverts),
        ffaces("f", nmaxfaces),
        psifaces("psi", nmaxfaces),
        psiexactverts("psi_exact", nmaxverts),
        psiexactfaces("psi_exact", nmaxfaces),
        everts("error", nmaxverts),
        efaces("error", nmaxfaces),
        uverts("u", nmaxverts),
        ufaces("u", nmaxfaces),
        uvertsexact("u_exact", nmaxverts),
        ufacesexact("u_exact", nmaxfaces),
        uerrverts("u_error", nmaxverts),
        uerrfaces("u_error", nmaxfaces) {
    _fverts = ko::create_mirror_view(fverts);
    _psiverts = ko::create_mirror_view(psiverts);
    _ffaces = ko::create_mirror_view(ffaces);
    _psifaces = ko::create_mirror_view(psifaces);
    _psiexactverts = ko::create_mirror_view(psiexactverts);
    _psiexactfaces = ko::create_mirror_view(psiexactfaces);
    _everts = ko::create_mirror_view(everts);
    _efaces = ko::create_mirror_view(efaces);
    _uverts = ko::create_mirror_view(uverts);
    _ufaces = ko::create_mirror_view(ufaces);
    _uvertsexact = ko::create_mirror_view(uvertsexact);
    _ufacesexact = ko::create_mirror_view(ufacesexact);
    _uerrverts = ko::create_mirror_view(uerrverts);
    _uerrfaces = ko::create_mirror_view(uerrfaces);
  }

  /** @brief Returns the average mesh spacing.

  @hostfn

  */
  inline Real meshSize() const {
    return std::sqrt(4 * PI / this->faces.nLeavesHost());
  }

  /** @brief Initializes the Poisson problem data on the mesh.
   */
  void init() {
    ko::parallel_for(
        this->nvertsHost(),
        Init(fverts, psiexactverts, uvertsexact, this->getVertCrds()));
    ko::parallel_for(
        this->nfacesHost(),
        Init(ffaces, psiexactfaces, ufacesexact, this->getFaceCrds()));
  }

  /** @brief Solves the Poisson equation


  */
  void solve(const int& nthreads = 0) {
    /** Set parallel team policy

    */
    ko::TeamPolicy<> vertex_policy(this->nvertsHost(), ko::AUTO());
    ko::TeamPolicy<> face_policy(this->nfacesHost(), ko::AUTO());
    if (nthreads != 0) {
      vertex_policy = ko::TeamPolicy<>(this->nvertsHost(), nthreads);
      face_policy = ko::TeamPolicy<>(this->nfacesHost(), nthreads);
    }

    ko::Profiling::pushRegion("poisson solve");
    ko::Profiling::pushRegion("vertex solve");
    /// parallel vertex solve (kernel launch)
    ko::parallel_for(vertex_policy,
                     VertexSolve(this->getVertCrds(), this->getFaceCrds(),
                                 ffaces, this->getFaceArea(),
                                 this->getFacemask(), psiverts, uverts));
    ko::Profiling::popRegion();
    /// parallel face solve (kernel launch)
    ko::Profiling::pushRegion("face solve");
    ko::parallel_for(face_policy,
                     FaceSolve(this->getFaceCrds(), ffaces, this->getFaceArea(),
                               this->getFacemask(), psifaces, ufaces));
    ko::Profiling::popRegion();
    ko::Profiling::popRegion();

    /// compute stream function error in potential
    ko::parallel_for(
        ko::RangePolicy<Error::VertTag>(0, this->nvertsHost()),
        Error(everts, psiverts, psiexactverts, this->getFacemask()));
    ko::parallel_for(
        ko::RangePolicy<Error::FaceTag>(0, this->nfacesHost()),
        Error(efaces, psifaces, psiexactfaces, this->getFacemask()));
    ko::parallel_reduce(
        "MaxReduce", ko::RangePolicy<Error::MaxTag>(0, this->nvertsHost()),
        Error(everts, psiverts, psiexactverts, this->getFacemask()),
        ko::Max<Real>(linf_verts));
    ko::parallel_reduce(
        "MaxReduce", ko::RangePolicy<Error::MaxTag>(0, this->nfacesHost()),
        Error(efaces, psifaces, psiexactfaces, this->getFacemask()),
        ko::Max<Real>(linf_faces));
    /// compute error in velocity
    ko::parallel_for(
        ko::RangePolicy<UError::VertTag>(0, this->nvertsHost()),
        UError(uerrverts, uverts, uvertsexact, this->getFacemask()));
    ko::parallel_for(
        ko::RangePolicy<UError::FaceTag>(0, this->nfacesHost()),
        UError(uerrfaces, ufaces, ufacesexact, this->getFacemask()));
    ko::parallel_reduce(
        "MaxReduce", ko::RangePolicy<UError::MaxTag>(0, this->nvertsHost()),
        UError(uerrverts, uverts, uvertsexact, this->getFacemask()),
        ko::Max<Real>(linf_uverts));
    ko::parallel_reduce(
        "MaxReduce", ko::RangePolicy<UError::MaxTag>(0, this->nfacesHost()),
        UError(uerrfaces, ufaces, ufacesexact, this->getFacemask()),
        ko::Max<Real>(linf_ufaces));

    std::cout << meshSize() * RAD2DEG << ":\n"
              << "\tpotential: linf_verts = " << linf_verts
              << ", linf_faces = " << linf_faces << '\n'
              << "\tvelocity:  linf_verts = " << std::sqrt(linf_uverts)
              << ", linf_faces = " << std::sqrt(linf_ufaces) << '\n';
  }

  /** @brief copies data from device to host
   */
  void updateHost() const override {
    PolyMesh2d<SeedType>::updateHost();
    ko::deep_copy(_fverts, fverts);
    ko::deep_copy(_psiverts, psiverts);
    ko::deep_copy(_ffaces, ffaces);
    ko::deep_copy(_psifaces, psifaces);
    ko::deep_copy(_psiexactverts, psiexactverts);
    ko::deep_copy(_psiexactfaces, psiexactfaces);
    ko::deep_copy(_everts, everts);
    ko::deep_copy(_efaces, efaces);
    ko::deep_copy(_uverts, uverts);
    ko::deep_copy(_ufaces, ufaces);
    ko::deep_copy(_uvertsexact, uvertsexact);
    ko::deep_copy(_ufacesexact, ufacesexact);
    ko::deep_copy(_uerrverts, uerrverts);
    ko::deep_copy(_uerrfaces, uerrfaces);
  }

  /** @brief copies data from host to device

    @note Initialization is done on the device using Init.
  */
  void updateDevice() const override { PolyMesh2d<SeedType>::updateDevice(); }

  /** @brief Output all data to vtk

    @param fname filename for vtk output
  */
  void outputVtk(const std::string& fname) const override {
    VtkInterface<SphereGeometry, Faces<FaceKind>> vtk;
    auto ptdata = vtkSmartPointer<vtkPointData>::New();
    vtk.addScalarToPointData(ptdata, _fverts, "f", this->nvertsHost());
    vtk.addScalarToPointData(ptdata, _psiverts, "psi", this->nvertsHost());
    vtk.addScalarToPointData(ptdata, _psiexactverts, "psi_exact",
                             this->nvertsHost());
    vtk.addScalarToPointData(ptdata, _everts, "psi_error", this->nvertsHost());
    vtk.addVectorToPointData(ptdata, _uverts, "velocity", this->nvertsHost());
    vtk.addVectorToPointData(ptdata, _uvertsexact, "velocity_exact",
                             this->nvertsHost());
    vtk.addVectorToPointData(ptdata, _uerrverts, "velocity_error",
                             this->nvertsHost());

    auto celldata = vtkSmartPointer<vtkCellData>::New();
    vtk.addScalarToCellData(celldata, this->faces.getAreaHost(), "area",
                            this->faces);
    vtk.addScalarToCellData(celldata, _ffaces, "f", this->faces);
    vtk.addScalarToCellData(celldata, _psifaces, "psi", this->faces);
    vtk.addScalarToCellData(celldata, _psiexactfaces, "psi_exact", this->faces);
    vtk.addScalarToCellData(celldata, _efaces, "psi_error", this->faces);
    vtk.addVectorToCellData(celldata, _ufaces, "velocity", this->faces);
    vtk.addVectorToCellData(celldata, _ufacesexact, "velocity_exact",
                            this->faces);
    vtk.addVectorToCellData(celldata, _uerrfaces, "velocity_error",
                            this->faces);

    auto pd = vtk.toVtkPolyData(this->faces, this->edges, this->physFaces,
                                this->physVerts, ptdata, celldata);
    vtk.writePolyData(fname, pd);
  }

 protected:
  typedef typename scalar_view_type::HostMirror host_scalar_view;
  typedef typename vec_view::HostMirror host_vec_view;
  host_scalar_view _fverts;
  host_scalar_view _psiverts;
  host_scalar_view _ffaces;
  host_scalar_view _psifaces;
  host_scalar_view _psiexactverts;
  host_scalar_view _psiexactfaces;
  host_scalar_view _everts;
  host_scalar_view _efaces;
  host_vec_view _uverts;
  host_vec_view _ufaces;
  host_vec_view _uvertsexact;
  host_vec_view _ufacesexact;
  host_vec_view _uerrverts;
  host_vec_view _uerrfaces;
};

}  // namespace Lpm
#endif
