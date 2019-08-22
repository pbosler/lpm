#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmVtkIO.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmCompadre.hpp"
#include "LpmSpherePoisson.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include "Kokkos_Core.hpp"
#include <fstream>
#include <cmath>

using namespace Lpm;

struct Input {
    int max_depth;
    int tgt_nlon;
    std::string mfilename;
    std::vector<Int> nlons;
    std::vector<Int> nlats;

    Input(int argc, char* argv[]);
};

void targetMesh(ko::View<Real*[3]> ll, const int nlat, const int nlon, const Real dlam) ;
template <typename G, typename F>
ko::View<Real*[3]> sourceCoords(const PolyMesh2d<G,F>& pm);

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
/**
    Pointwise interpolation tests
    Data defined on a polymesh object (cubed sphere or icosahedral triangulation of the sphere).
    Interpolated to uniform lat-lon mesh.
    Objectives:
        1. Exercise LpmCompadre classes/structs.
        2. Determine convergence properties of GMLS on these meshes.
        3. Identify parameter ranges for each tree depth.
*/

    Input input(argc, argv);

    CompadreParams gmlsParams;
    std::cout << gmlsParams.infoString();
    
    for (int i=2; i<=input.max_depth; ++i) {
        /**
            Build source mesh 
        */
        Index nmaxverts, nmaxedges, nmaxfaces;
        MeshSeed<IcosTriSphereSeed> icseed;
        icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, i);
        PolyMesh2d<SphereGeometry,TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
        trisphere.treeInit(i, icseed);
        trisphere.updateDevice();
        auto srcCrds = sourceCoords<SphereGeometry,TriFace>(trisphere);
        typename ko::View<Real*[3]>::HostMirror src_host = ko::create_mirror_view(srcCrds);
        ko::deep_copy(src_host, srcCrds);
        /** 
            Define source data
        */
        ko::View<Real*> zeta("zeta", srcCrds.extent(0));
        ko::View<Real*> psi("psi", srcCrds.extent(0));
        ko::parallel_for(srcCrds.extent(0), KOKKOS_LAMBDA (const Index i) {
            auto myx = ko::subview(srcCrds, i, ko::ALL());
            const Real zt = SphHarm54(myx);
            zeta(i) = zt;
            psi(i) = zt/30.0;
        });

        for (int j=0; j<1; ++j) {
            /** 
                Build target mesh
            */
            const int nlon = input.nlons[j];
            const int nlat = input.nlats[j];
            const int nunif = nlon*nlat;
            const Real dlam = 2*PI/nlon;
            ko::View<Real*[3]> ll("llmesh", nunif);
            targetMesh(ll, nlat, nlon, dlam);
            typename ko::View<Real*[3]>::HostMirror llhost = ko::create_mirror_view(ll);
            ko::deep_copy(llhost, ll);
            
            ko::View<Real*> exact_zeta("zeta_exact", nunif);
            ko::View<Real*> exact_psi("psi_exact", nunif);
            ko::parallel_for(nunif, KOKKOS_LAMBDA (const Index i) { 
                auto myx = ko::subview(ll, i, ko::ALL());
                const Real zt = SphHarm54(myx);
                exact_zeta(i) = zt;
                exact_psi(i) = zt/30.0;
            });
            
            /**
                Construct neighbor lists
            */
            CompadreNeighborhoods nn(src_host, llhost, gmlsParams);
            std::cout << nn.infoString();
            
            /**
                Setup gmls
            */
            ko::View<Real*> tgt_psi("psi", nunif);
            ko::View<Real*> tgt_zeta("zeta", nunif);
            ko::View<Real*> tgt_zeta_lap("zeta", nunif);
            
            std::vector<Compadre::TargetOperation> s_ops = {Compadre::ScalarPointEvaluation,
                                                            Compadre::LaplacianOfScalarPointEvaluation};
            Compadre::GMLS sgmls = scalarGMLS(srcCrds, ll, nn, gmlsParams, s_ops);

            /**
                Compute solutions: lap(psi) = -zeta
            */
            Compadre::Evaluator e_scalar(&sgmls);
            tgt_psi = e_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(psi, s_ops[0],
                Compadre::PointSample);
            tgt_zeta = e_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(zeta, s_ops[1],
                Compadre::PointSample);
            
        }
    }

}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
    max_depth = 2;
    tgt_nlon = 180;
    mfilename = "compadre_tests.m";
    nlats = {45, 91, 181, 361, 721};
    nlons = {90, 180, 360, 720, 1440};
    for (int i=1; i<argc; ++i) {
        const std::string& token = argv[i];
        if (token == "-d" || token == "-tree") {
            max_depth = std::stoi(argv[++i]);
        }
        else if (token == "-nlon") {
            tgt_nlon = std::stoi(argv[++i]);
        }
        else if (token == "-m") {
            mfilename = argv[++i];
        }
    }
}

void targetMesh(ko::View<Real*[3]> ll, const int nlat, const int nlon, const Real dlam) {
    ko::parallel_for(nlat, KOKKOS_LAMBDA (int i) {
        const int start_ind = i*nlon;
        const Real lat = -0.5*PI +i*dlam;
        const Real coslat = std::cos(lat);
        const Real z = std::sin(lat);
        for (int j=0; j<nlon; ++j) {
            const Int insert_ind = start_ind + j;
            const Real x = std::cos(j*dlam)*coslat;
            const Real y = std::sin(j*dlam)*coslat;
            ll(insert_ind,0) = x;
            ll(insert_ind,1) = y;
            ll(insert_ind,2) = z;
        }
    });
}

template <typename G, typename F>
ko::View<Real*[3]> sourceCoords(const PolyMesh2d<G,F>& pm) {
    const Index nv = pm.nverts();
    ko::View<Real*[3]> result("source_coords", nv + pm.faces.nLeavesHost());
    ko::parallel_for(nv, KOKKOS_LAMBDA (int i) {
        for (int j=0; j<3; ++j) {
            result(i,j) = pm.physVerts.crds(i,j);
        }
    });
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
        Int offset = nv;
        for (int j=0; j<pm.nfaces(); ++j) {
            if (!pm.faces.mask(j)) {
                result(offset,0) = pm.physFaces.crds(j,0);
                result(offset,1) = pm.physFaces.crds(j,1);
                result(offset++,2) = pm.physFaces.crds(j,2);
            }
        }
    });
    return result;
}