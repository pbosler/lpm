#ifndef LPM_SPHERE_POISSON_HPP
#define LPM_SPHERE_POISSON_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"
#include "LpmVtkIO.hpp"
#include <cmath>
#include <iomanip>

namespace Lpm {

typedef typename SphereGeometry::crd_view_type crd_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef typename ko::TeamPolicy<>::member_type member_type;

template <typename VecType> KOKKOS_INLINE_FUNCTION
void greensFn(Real& psi, const VecType& tgt_x, const VecType& src_xx, const Real& src_f, const Real& src_area) {
    psi = -std::log(1.0 - SphereGeometry::dot(tgt_x, src_xx))*src_f*src_area/(4*PI);
}

template <typename VecType> KOKKOS_INLINE_FUNCTION
void biotSavart(ko::Tuple<Real,3>& u, const VecType& tgt_x, const VecType& src_xx, const Real& src_f, const Real& src_area) {
    u = SphereGeometry::cross(tgt_x, src_xx);
    const Real str = -src_f*src_area/(4*PI*(1 - SphereGeometry::dot(tgt_x, src_xx)));
    for (int j=0; j<3; ++j) {
        u[j] *= str;
    }
}

KOKKOS_INLINE_FUNCTION
Real legendre54(const Real& z) {
    return z * (z*z - 1.0) * (z*z - 1.0);
}

template <typename VecType> KOKKOS_INLINE_FUNCTION
Real SphHarm54(const VecType& x) {
    const Real lam = SphereGeometry::longitude(x);
    return 30*std::cos(4*lam)*legendre54(x[2]);
}

template <typename VT> KOKKOS_INLINE_FUNCTION
ko::Tuple<Real,3> RH54Velocity(const VT& x) {
    const Real theta = SphereGeometry::latitude(x);
    const Real lambda = SphereGeometry::longitude(x);
    const Real usph = 0.5*std::cos(4*lambda)*cube(std::cos(theta))*(5*std::cos(2*theta) - 3);
    const Real vsph = 4*cube(std::cos(theta))*std::sin(theta)*std::sin(4*lambda);
    const Real u = -usph*std::sin(lambda) - vsph*std::sin(theta)*std::cos(lambda);
    const Real v =  usph*std::cos(lambda) - vsph*std::sin(theta)*std::sin(lambda);
    const Real w = vsph*std::cos(theta);
    return ko::Tuple<Real,3>(u,v,w);
}

struct Init {
    scalar_view_type f;
    scalar_view_type exactpsi;
    vec_view exactu;
    crd_view x;
    
    Init(scalar_view_type ff, scalar_view_type psi, vec_view u, crd_view xx) : 
        f(ff), exactpsi(psi), exactu(u), x(xx) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
        auto myx = ko::subview(x, i, ko::ALL());
        const Real myf = SphHarm54(myx);
        f(i) = myf;
        exactpsi(i) = myf/30.0;
        const ko::Tuple<Real,3> u = RH54Velocity(myx);
        for (int j=0; j<3; ++j) {
            exactu(i,j) = u[j];
        }
    }
};

struct ReduceDistinct {
    typedef Real value_type;
    Index i;
    crd_view tgtx;
    crd_view srcx;
    scalar_view_type srcf;
    scalar_view_type srca;
    mask_view_type facemask;
    
    KOKKOS_INLINE_FUNCTION
    ReduceDistinct(const Index& ii, crd_view x, crd_view xx, scalar_view_type f, scalar_view_type a, mask_view_type fm) :
        i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& pot) const {
        Real potential = 0;
        if (!facemask(j)) {
            auto mytgt = ko::subview(tgtx,i,ko::ALL());
            auto mysrc = ko::subview(srcx,j,ko::ALL());
            greensFn(potential, mytgt, mysrc, srcf(j), srca(j));
        }
        pot += potential;
    }
};

struct UReduceDistinct {
    typedef ko::Tuple<Real,3> value_type;
    Index i;
    crd_view tgtx;
    crd_view srcx;
    scalar_view_type srcf;
    scalar_view_type srca;
    mask_view_type facemask;
    
    KOKKOS_INLINE_FUNCTION
    UReduceDistinct(const Index& ii, crd_view x, crd_view xx, scalar_view_type f, scalar_view_type a, mask_view_type fm):
    i(ii), tgtx(x), srcx(xx), srcf(f), srca(a), facemask(fm) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& vel) const {
        ko::Tuple<Real,3> u;
        if (!facemask(j)) {
            auto mytgt = ko::subview(tgtx, i, ko::ALL());
            auto mysrc = ko::subview(srcx, j, ko::ALL());
            biotSavart(u, mytgt, mysrc, srcf(j), srca(j));
        }
        vel += u;
    }
};

struct VertexSolve {
    crd_view vertx;
    crd_view facex;
    scalar_view_type facef;
    scalar_view_type facea;
    mask_view_type facemask;
    scalar_view_type vertpsi;
    vec_view vertu;
    Index nf;
    
    VertexSolve(crd_view vx, crd_view fx, scalar_view_type ff, scalar_view_type fa, mask_view_type fm, 
        scalar_view_type vpsi, vec_view vu) :
        vertx(vx), facex(fx), facef(ff), facea(fa), facemask(fm), vertpsi(vpsi), vertu(vu), nf(fx.extent(0)) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const member_type& mbr) const {
        const Index i = mbr.league_rank();
        Real p;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), ReduceDistinct(i, vertx, facex, facef, facea, facemask), p);
        vertpsi(i) = p;
        ko::Tuple<Real,3> u;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), UReduceDistinct(i, vertx, facex, facef, facea, facemask), u);
        for (int j=0; j<3; ++j) {
            vertu(i,j) = u[j];
        }
    }
};

struct ReduceCollocated {
    typedef Real value_type;
    Index i;
    crd_view srcx;
    scalar_view_type srcf;
    scalar_view_type srca;
    mask_view_type mask;
    
    KOKKOS_INLINE_FUNCTION
    ReduceCollocated(const Index& ii, crd_view x, scalar_view_type f, scalar_view_type a, mask_view_type m) : 
        i(ii), srcx(x), srcf(f), srca(a), mask(m) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& pot) const {
        Real potential = 0;
        if (!mask(j) && i != j) {
            auto mtgt = ko::subview(srcx, i, ko::ALL());
            auto msrc = ko::subview(srcx, j, ko::ALL());
            greensFn(potential, mtgt, msrc, srcf(j), srca(j));
        }
        pot += potential;
    }
};

struct UReduceCollocated {
    typedef ko::Tuple<Real,3> value_type;
    Index i;
    crd_view srcx;
    scalar_view_type srcf;
    scalar_view_type srca;
    mask_view_type mask;
    
    KOKKOS_INLINE_FUNCTION
    UReduceCollocated(const Index& ii, crd_view x, scalar_view_type f, scalar_view_type a, mask_view_type m) :
        i(ii), srcx(x), srcf(f), srca(a), mask(m) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& j, value_type& vel) const {
        ko::Tuple<Real,3> u;
        if (!mask(j) && i != j) {
            auto mtgt = ko::subview(srcx, i, ko::ALL());
            auto msrc = ko::subview(srcx, j, ko::ALL());
            biotSavart(u, mtgt, msrc, srcf(j), srca(j));
        }
        vel += u;
    }
};

struct FaceSolve {
    crd_view facex;
    scalar_view_type facef;
    scalar_view_type facea;
    mask_view_type facemask;
    scalar_view_type facepsi;
    vec_view faceu;
    Index nf;
    
    FaceSolve(crd_view x, scalar_view_type f, scalar_view_type a, mask_view_type m, scalar_view_type p, vec_view u) :
        facex(x), facef(f), facea(a), facemask(m), facepsi(p), faceu(u), nf(x.extent(0)) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index i=mbr.league_rank();
        Real p;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), ReduceCollocated(i, facex, facef, facea, facemask), p);
        facepsi(i) = p;
        ko::Tuple<Real,3> u;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), UReduceCollocated(i, facex, facef, facea, facemask), u);
        for (int j=0; j<3; ++j) {
            faceu(i,j) = u[j];
        }
    }
};

struct Error {
    typedef Real value_type;
    scalar_view_type fcomputed;
    scalar_view_type fexact;
    scalar_view_type ferror;
    mask_view_type mask;
    
    struct VertTag{};
    struct FaceTag{};
    struct MaxTag{};
    
    Error(scalar_view_type e, scalar_view_type fc, scalar_view_type fe, mask_view_type m) : 
        ferror(e), fcomputed(fc), fexact(fe), mask(m) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const VertTag&, const Index& i) const {
        ferror(i) = abs(fcomputed(i) - fexact(i));
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const FaceTag&, const Index& i) const {
        ferror(i) = (mask(i) ? 0 : abs(fcomputed(i) - fexact(i)));
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const MaxTag&, const Index& i, value_type& max) const {
        if (ferror(i) > max) max=ferror(i);
    }
};

struct UError {
    typedef Real value_type;
    vec_view uerror;
    vec_view ucomputed;
    vec_view uexact;
    mask_view_type mask;
    
    struct VertTag{};
    struct FaceTag{};
    struct MaxTag{};
    
    UError(vec_view e, vec_view uc, vec_view ue, mask_view_type m) : 
    	uerror(e), ucomputed(uc), uexact(ue), mask(m) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const VertTag&, const Index& i) const {
        for (int j=0; j<3; ++j) {
            uerror(i,j) = abs(ucomputed(i,j) - uexact(i,j));
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const FaceTag&, const Index& i) const {
        if (mask(i)) {
            for (int j=0; j<3; ++j) {
                uerror(i,j) = 0.0;
            }
        }
        else {
            for (int j=0; j<3; ++j) {
                uerror(i,j) = abs(ucomputed(i,j) - uexact(i,j));
            }
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const MaxTag&, const Index& i, value_type& max) const {
        Real emag = 0;
        for (int j=0; j<3; ++j) {
            emag += square(uerror(i,j));
        }
        if (emag > max) max = emag;
    }
};

/**
    Solves laplacian(psi) = -f on the sphere
*/
template <typename FaceKind> class SpherePoisson : public PolyMesh2d<SphereGeometry,FaceKind> {
    public:
        scalar_view_type fverts;
        scalar_view_type psiverts;
        scalar_view_type psiexactverts;
        scalar_view_type ffaces;
        scalar_view_type psifaces;
        scalar_view_type psiexactfaces;
        scalar_view_type everts;
        scalar_view_type efaces;
        vec_view uverts;
        vec_view ufaces;
        vec_view uvertsexact;
        vec_view ufacesexact;
        vec_view uerrverts;
        vec_view uerrfaces;
        Real linf_verts;
        Real linf_faces;
        Real linf_uverts;
        Real linf_ufaces;
    
        SpherePoisson(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) : 
            PolyMesh2d<SphereGeometry,FaceKind>(nmaxverts, nmaxedges, nmaxfaces),
            fverts("f",nmaxverts), psiverts("psi",nmaxverts),
            ffaces("f",nmaxfaces), psifaces("psi",nmaxfaces),
            psiexactverts("psi_exact",nmaxverts), psiexactfaces("psi_exact",nmaxfaces),
            everts("error", nmaxverts), efaces("error",nmaxfaces),
            uverts("u",nmaxverts), ufaces("u",nmaxfaces), 
            uvertsexact("u_exact",nmaxverts), ufacesexact("u_exact",nmaxfaces),
            uerrverts("u_error",nmaxverts), uerrfaces("u_error",nmaxfaces) 
        {
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
        
        inline Real meshSize() const {return std::sqrt(4*PI / this->faces.nLeavesHost());}
        
        void init() {
            ko::parallel_for(this->nvertsHost(), Init(fverts, psiexactverts, uvertsexact, this->getVertCrds()));
            ko::parallel_for(this->nfacesHost(), Init(ffaces, psiexactfaces, ufacesexact, this->getFaceCrds()));
        }
        
        void solve() {
            ko::TeamPolicy<> vertex_policy(this->nvertsHost(), ko::AUTO());
            ko::TeamPolicy<> face_policy(this->nfacesHost(), ko::AUTO());
            ko::parallel_for(vertex_policy, VertexSolve(this->getVertCrds(), this->getFaceCrds(), ffaces, 
                this->getFaceArea(), this->getFacemask(), psiverts, uverts));
            ko::parallel_for(face_policy, FaceSolve(this->getFaceCrds(), ffaces, this->getFaceArea(), 
                this->getFacemask(), psifaces, ufaces));
            /// compute error in potential
            ko::parallel_for(ko::RangePolicy<Error::VertTag>(0,this->nvertsHost()),
                Error(everts, psiverts, psiexactverts, this->getFacemask()));
            ko::parallel_for(ko::RangePolicy<Error::FaceTag>(0,this->nfacesHost()),
                Error(efaces, psifaces, psiexactfaces, this->getFacemask()));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<Error::MaxTag>(0,this->nvertsHost()),
                Error(everts, psiverts, psiexactverts, this->getFacemask()), ko::Max<Real>(linf_verts));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<Error::MaxTag>(0,this->nfacesHost()),
                Error(efaces, psifaces, psiexactfaces, this->getFacemask()), ko::Max<Real>(linf_faces));
            /// compute error in velocity
            ko::parallel_for(ko::RangePolicy<UError::VertTag>(0, this->nvertsHost()),
                UError(uerrverts, uverts, uvertsexact, this->getFacemask()));
            ko::parallel_for(ko::RangePolicy<UError::FaceTag>(0, this->nfacesHost()),
                UError(uerrfaces, ufaces, ufacesexact, this->getFacemask()));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<UError::MaxTag>(0,this->nvertsHost()),
                UError(uerrverts, uverts, uvertsexact, this->getFacemask()), ko::Max<Real>(linf_uverts));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<UError::MaxTag>(0,this->nfacesHost()),
                UError(uerrfaces, ufaces, ufacesexact, this->getFacemask()), ko::Max<Real>(linf_ufaces));
            
            std::cout << meshSize()*RAD2DEG << ":\n"
                      << "\tpotential: linf_verts = " << linf_verts << ", linf_faces = " << linf_faces << '\n'
                      << "\tvelocity:  linf_verts = " << std::sqrt(linf_uverts) << ", linf_faces = " << std::sqrt(linf_ufaces) << '\n';
                    
        }

        void updateHost() const override {
            PolyMesh2d<SphereGeometry,FaceKind>::updateHost();
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
        
        void updateDevice() const override {
            PolyMesh2d<SphereGeometry,FaceKind>::updateDevice();
        }
    
        void outputVtk(const std::string& fname) const override {
            VtkInterface<SphereGeometry,Faces<FaceKind>> vtk;
            auto ptdata = vtkSmartPointer<vtkPointData>::New();
            vtk.addScalarToPointData(ptdata, _fverts, "f", this->nvertsHost());
            vtk.addScalarToPointData(ptdata, _psiverts, "psi", this->nvertsHost());
            vtk.addScalarToPointData(ptdata, _psiexactverts, "psi_exact", this->nvertsHost()); 
            vtk.addScalarToPointData(ptdata, _everts, "psi_error", this->nvertsHost()); 
            vtk.addVectorToPointData(ptdata, _uverts, "velocity", this->nvertsHost());
            vtk.addVectorToPointData(ptdata, _uvertsexact, "velocity_exact", this->nvertsHost());
            vtk.addVectorToPointData(ptdata, _uerrverts, "velocity_error", this->nvertsHost());
            
            auto celldata = vtkSmartPointer<vtkCellData>::New();
            vtk.addScalarToCellData(celldata, this->faces.getAreaHost(), "area", this->faces);
            vtk.addScalarToCellData(celldata, _ffaces, "f", this->faces);
            vtk.addScalarToCellData(celldata, _psifaces, "psi", this->faces);
            vtk.addScalarToCellData(celldata, _psiexactfaces, "psi_exact", this->faces);
            vtk.addScalarToCellData(celldata, _efaces, "psi_error", this->faces);
            vtk.addVectorToCellData(celldata, _ufaces, "velocity", this->faces);
            vtk.addVectorToCellData(celldata, _ufacesexact, "velocity_exact", this->faces);
            vtk.addVectorToCellData(celldata, _uerrfaces, "velocity_error", this->faces);
            
            auto pd = vtk.toVtkPolyData(this->faces, this->edges, this->physFaces, this->physVerts, ptdata, celldata);
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

}
#endif
