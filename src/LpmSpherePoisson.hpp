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
typedef typename ko::TeamPolicy<>::member_type member_type;

template <typename VecType> KOKKOS_INLINE_FUNCTION
void greensFn(Real& psi, const VecType& tgt_x, const VecType& src_xx, const Real& src_f, const Real& src_area) {
    psi = -std::log(1.0 - SphereGeometry::dot(tgt_x, src_xx))*src_f*src_area/(4*PI);
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

struct Init {
    scalar_view_type f;
    scalar_view_type exactpsi;
    crd_view x;
    
    Init(scalar_view_type ff, scalar_view_type psi, crd_view xx) : 
        f(ff), exactpsi(psi), x(xx) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
        auto myx = ko::subview(x, i, ko::ALL());
        const Real myf = SphHarm54(myx);
        f(i) = myf;
        exactpsi(i) = myf/30.0;
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

struct VertexSolve {
    crd_view vertx;
    crd_view facex;
    scalar_view_type facef;
    scalar_view_type facea;
    mask_view_type facemask;
    scalar_view_type vertpsi;
    Index nf;
    
    VertexSolve(crd_view vx, crd_view fx, scalar_view_type ff, scalar_view_type fa, mask_view_type fm, scalar_view_type vpsi) :
        vertx(vx), facex(fx), facef(ff), facea(fa), facemask(fm), vertpsi(vpsi), nf(fx.extent(0)) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const member_type& mbr) const {
        const Index i = mbr.league_rank();
        Real p;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), ReduceDistinct(i, vertx, facex, facef, facea, facemask), p);
        vertpsi(i) = p;
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

struct FaceSolve {
    crd_view facex;
    scalar_view_type facef;
    scalar_view_type facea;
    mask_view_type facemask;
    scalar_view_type facepsi;
    Index nf;
    
    FaceSolve(crd_view x, scalar_view_type f, scalar_view_type a, mask_view_type m, scalar_view_type p) :
        facex(x), facef(f), facea(a), facemask(m), facepsi(p), nf(x.extent(0)) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index i=mbr.league_rank();
        Real p;
        ko::parallel_reduce(ko::TeamThreadRange(mbr, nf), ReduceCollocated(i, facex, facef, facea, facemask), p);
        facepsi(i) = p;
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
        Real linf_verts;
        Real linf_faces;
    
        SpherePoisson(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) : 
            PolyMesh2d<SphereGeometry,FaceKind>(nmaxverts, nmaxedges, nmaxfaces),
            fverts("f",nmaxverts), psiverts("psi",nmaxverts),
            ffaces("f",nmaxfaces), psifaces("psi",nmaxfaces),
            psiexactverts("psi_exact",nmaxverts), psiexactfaces("psi_exact",nmaxfaces),
            everts("error", nmaxverts), efaces("error",nmaxfaces) 
        {
            _fverts = ko::create_mirror_view(fverts);
            _psiverts = ko::create_mirror_view(psiverts);
            _ffaces = ko::create_mirror_view(ffaces);
            _psifaces = ko::create_mirror_view(psifaces);
            _psiexactverts = ko::create_mirror_view(psiexactverts);
            _psiexactfaces = ko::create_mirror_view(psiexactfaces);
            _everts = ko::create_mirror_view(everts);
            _efaces = ko::create_mirror_view(efaces);
        }
        
        inline Real meshSize() const {return std::sqrt(4*PI / this->faces.nLeavesHost());}
        
        void init() {
//             std::cout << "initializing problem data nhv = " << this->nvertsHost() <<std::endl;
            ko::parallel_for(this->nvertsHost(), Init(fverts, psiexactverts, this->getVertCrds()));
//             std::cout << "vertex problem data defined." << std::endl;
            ko::parallel_for(this->nfacesHost(), Init(ffaces, psiexactfaces, this->getFaceCrds()));
//             std::cout << "face problem data defined." << std::endl;
        }
        
        void solve() {
            ko::TeamPolicy<> vertex_policy(this->nvertsHost(), ko::AUTO());
            ko::TeamPolicy<> face_policy(this->nfacesHost(), ko::AUTO());
            ko::parallel_for(vertex_policy, VertexSolve(this->getVertCrds(), this->getFaceCrds(), ffaces, 
                this->getFaceArea(), this->getFacemask(), psiverts));
            ko::parallel_for(face_policy, FaceSolve(this->getFaceCrds(), ffaces, this->getFaceArea(), 
                this->getFacemask(), psifaces));
            /// compute error
            ko::parallel_for(ko::RangePolicy<Error::VertTag>(0,this->nvertsHost()),
                Error(everts, psiverts, psiexactverts, this->getFacemask()));
            ko::parallel_for(ko::RangePolicy<Error::FaceTag>(0,this->nfacesHost()),
                Error(efaces, psifaces, psiexactfaces, this->getFacemask()));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<Error::MaxTag>(0,this->nvertsHost()),
                Error(everts, psiverts, psiexactverts, this->getFacemask()), ko::Max<Real>(linf_verts));
            ko::parallel_reduce("MaxReduce", ko::RangePolicy<Error::MaxTag>(0,this->nfacesHost()),
                Error(efaces, psifaces, psiexactfaces, this->getFacemask()), ko::Max<Real>(linf_faces));
            
            std::cout << meshSize()*RAD2DEG 
                      << ": linf_verts = " << linf_verts << ", linf_faces = " << linf_faces << std::endl;
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
        }
        
        void updateDevice() const override {
            PolyMesh2d<SphereGeometry,FaceKind>::updateDevice();
            ko::deep_copy(fverts, _fverts);
            ko::deep_copy(psiverts, _psiverts);
            ko::deep_copy(ffaces, _ffaces);
            ko::deep_copy(psifaces, _psifaces);
            ko::deep_copy(psiexactverts, _psiexactverts);
            ko::deep_copy(psiexactfaces, _psiexactfaces);
        }
    
        void outputVtk(const std::string& fname) const override {
            VtkInterface<SphereGeometry,Faces<FaceKind>> vtk;
            auto ptdata = vtkSmartPointer<vtkPointData>::New();
            vtk.addScalarToPointData(ptdata, _fverts, "f", this->nvertsHost());
            vtk.addScalarToPointData(ptdata, _psiverts, "psi", this->nvertsHost());
            vtk.addScalarToPointData(ptdata, _psiexactverts, "psi_exact", this->nvertsHost()); 
            vtk.addScalarToPointData(ptdata, _everts, "error", this->nvertsHost()); 
            
            auto celldata = vtkSmartPointer<vtkCellData>::New();
            vtk.addScalarToCellData(celldata, this->faces.getAreaHost(), "area", this->faces);
            vtk.addScalarToCellData(celldata, _ffaces, "f", this->faces);
            vtk.addScalarToCellData(celldata, _psifaces, "psi", this->faces);
            vtk.addScalarToCellData(celldata, _psiexactfaces, "psi_exact", this->faces);
            vtk.addScalarToCellData(celldata, _efaces, "error", this->faces);
            
            auto pd = vtk.toVtkPolyData(this->faces, this->edges, this->physFaces, this->physVerts, ptdata, celldata);
            vtk.writePolyData(fname, pd);
        }
        
    protected:
        typedef typename scalar_view_type::HostMirror host_scalar_view;
        host_scalar_view _fverts;
        host_scalar_view _psiverts;
        host_scalar_view _ffaces;
        host_scalar_view _psifaces;
        host_scalar_view _psiexactverts;
        host_scalar_view _psiexactfaces;
        host_scalar_view _everts;
        host_scalar_view _efaces;
        typedef typename ko::View<Real,Dev>::HostMirror host_linf_view;
        host_linf_view _linf_verts;
        host_linf_view _linf_faces;

};

}
#endif
