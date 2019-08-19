#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmVtkIO.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmCompadre.hpp"
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

    Input(int argc, char* argv[]);
};

ko::View<Real*[3]> targetMesh(const int nlon);
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
    
//     for (int i=2; i<=input.max_depth; ++i) {
//         Index nmaxverts, nmaxedges, nmaxfaces;
//         MeshSeed<IcosTriSphereSeed> icseed;
//         icseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, i);
//         PolyMesh2d<SphereGeometry,TriFace> trisphere(nmaxverts, nmaxedges, nmaxfaces);
//         trisphere.treeInit(i, icseed);
//         trisphere.updateDevice();
//         auto srcCrds = sourceCoords<SphereGeometry,TriFace>(trisphere);
//         typename ko::View<Real*[3]>::HostMirror src_host = ko::create_mirror_view(srcCrds);
//         ko::deep_copy(src_host, srcCrds);
//         for (int j=0; j<input.nlons.size(); ++j) {
//             auto ll = targetMesh(input.nlons[j]);
//             typename ko::View<Real*[3]>::HostMirror llhost = ko::create_mirror_view(ll);
//             ko::deep_copy(llhost, ll);
//             CompadreNeighborhoods nn(src_host, llhost, gmlsParams);
//             std::cout << "tree_depth = " << i << " nlon = " << input.nlons[j] << ":\n";
//             std::cout << nn.infoString(1);
//         }
//     }

}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
    max_depth = 4;
    tgt_nlon = 180;
    mfilename = "compadre_tests.m";
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

ko::View<Real*[3]> targetMesh(const int nlon) {
    const int nlat = nlon/2 + 1;
    const int nunif = nlon*nlat;
    const Real dlam = 2*PI/nlon;
    ko::View<Real*[3]> result("llmesh", nunif);
    ko::parallel_for(nlat, KOKKOS_LAMBDA (int i) {
        const int start_ind = i*nlon;
        const Real lat = -0.5*PI +i*dlam;
        const Real coslat = std::cos(lat);
        const Real z = std::sin(lat);
        for (int j=0; i<nlon; ++j) {
            const Int insert_ind = start_ind + j;
            const Real x = std::cos(j*dlam)*coslat;
            const Real y = std::sin(j*dlam)*coslat;
            result(insert_ind,0) = x;
            result(insert_ind,1) = y;
            result(insert_ind,2) = z;
        }
    });
    return result;
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