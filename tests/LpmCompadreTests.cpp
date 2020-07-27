#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmLatLonMesh.hpp"
#include "LpmMatlabIO.hpp"
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
#include <iomanip>


using namespace Lpm;

struct Input {
    int max_depth;
    int tgt_nlon;
    std::string mfilename;
    std::vector<Int> nlons;
    std::vector<Int> nlats;
    Real eps;
    Int order;
    Input(int argc, char* argv[]);
};

struct Output {
    int order;
    std::vector<Real> nsrc;
    std::vector<Real> interp_l1;
    std::vector<Real> interp_l2;
    std::vector<Real> interp_linf;
    std::vector<Real> interp_l2rate;
    std::vector<Real> interp_linfrate;
    std::vector<Real> lap_l1;
    std::vector<Real> lap_l2;
    std::vector<Real> lap_linf;
    std::vector<Real> lap_l2rate;
    std::vector<Real> lap_linfrate;

    Output(const int ord, const int ntrials) : order(ord),
        nsrc(ntrials), interp_l1(ntrials), interp_l2(ntrials), interp_linf(ntrials),
        lap_l1(ntrials), lap_l2(ntrials), lap_linf(ntrials),
        interp_l2rate(ntrials), interp_linfrate(ntrials), lap_l2rate(ntrials), lap_linfrate(ntrials) {}

    std::string infoString() const;

    void computeRates();

    void writeInterpData() const;
    void writeLapData() const;
};

inline Real appxMeshSize(const Int nsrc) {return std::sqrt(4*PI/nsrc);}

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

    CompadreParams gmlsParams(input.order);
    gmlsParams.gmls_eps_mult = input.eps;
    gmlsParams.gmls_order = input.order;
    gmlsParams.gmls_manifold_order = input.order;
    std::cout << gmlsParams.infoString();

    typedef CubedSphereSeed seed_type;
    //typedef IcosTriSphereSeed seed_type;

    Output output(gmlsParams.gmls_order, input.max_depth-2+1);
//     std::cout << output.infoString();
    for (int i=2; i<=input.max_depth; ++i) {
        /**
            Build source mesh
        */
        Index nmaxverts, nmaxedges, nmaxfaces;
        MeshSeed<seed_type> seed;
        seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, i);
        PolyMesh2d<seed_type> sphere(nmaxverts, nmaxedges, nmaxfaces);
        sphere.treeInit(i, seed);
        sphere.updateDevice();
        ko::View<Real*[3]> srcCrds = sourceCoords<seed_type>(sphere);
        auto src_host = ko::create_mirror_view(srcCrds);
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
            psi(i) = -zt/30.0;
        });
        std::cout << "source data ready.\n";
        for (int j=0; j<input.nlons.size(); ++j) {
            /**
                Build target mesh
            */
            const int nlon = input.nlons[j];
            const int nlat = input.nlats[j];
            const int nunif = nlon*nlat;
            LatLonMesh ll(nlat, nlon);

            std::cout << "nsrc = " << srcCrds.extent(0) << " ntgt = " << nunif << "\n";

            /**
                Construct neighbor lists
            */
            CompadreNeighborhoods nn(src_host, ll.pts_host, gmlsParams);
            std::cout << nn.infoString(1);

            /**
                Setup gmls
            */
            ko::View<Real*> tgt_psi("psi", nunif);
            ko::View<Real*> tgt_zeta("zeta", nunif);
            ko::View<Real*> tgt_zeta_lap("zeta", nunif);
            std::vector<Compadre::TargetOperation> s_ops = {Compadre::ScalarPointEvaluation,
                                                            Compadre::LaplacianOfScalarPointEvaluation};
            Compadre::GMLS sgmls = scalarGMLS(srcCrds, ll.pts, nn, gmlsParams, s_ops);


            /**
                Compute solutions: lap(psi) = -zeta
            */
            Compadre::Evaluator e_scalar(&sgmls);
            tgt_psi = e_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(psi, s_ops[0],
                Compadre::PointSample);
            tgt_zeta = e_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(zeta, s_ops[0],
            	Compadre::PointSample);
            tgt_zeta_lap = e_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(psi, s_ops[1],
                Compadre::PointSample);

            /**
                Calculate error
            */
            ko::View<Real*> psi_exact("psi_exact", nunif);
            ko::View<Real*> zeta_exact("zeta_exact", nunif);
            ko::parallel_for(nunif, KOKKOS_LAMBDA(const Index& i) {
                auto myx = ko::subview(ll.pts, i, ko::ALL());
                const Real zt = SphHarm54(myx);
                zeta_exact(i) = zt;
                psi_exact(i) = -zt/30.0;
            });

            ko::View<Real*> psi_err("abs_err(psi)", nunif);
            ko::View<Real*> zeta_err("abs_err(zeta)", nunif);
            ko::View<Real*> zeta_lap_err("abs_err(zeta_lap)", nunif);
            ll.computeScalarError(psi_err, tgt_psi, psi_exact);
            ll.computeScalarError(zeta_err, tgt_zeta, zeta_exact);
            ll.computeScalarError(zeta_lap_err, tgt_zeta_lap, zeta_exact);
            ErrNorms<> enrm_psi(psi_err, psi_exact, ll.wts);
            ErrNorms<> enrm_zeta(zeta_err, zeta_exact, ll.wts);
            ErrNorms<> enrm_zeta_lap(zeta_lap_err, zeta_exact, ll.wts);

            output.nsrc[i-2] = srcCrds.extent(0);
            output.interp_l1[i-2] = enrm_psi.l1;
            output.interp_l2[i-2] = enrm_psi.l2;
            output.interp_linf[i-2] = enrm_psi.linf;

            output.lap_l1[i-2] = enrm_zeta_lap.l1;
            output.lap_l2[i-2] = enrm_zeta_lap.l2;
            output.lap_linf[i-2] = enrm_zeta_lap.linf;

            std::cout << enrm_psi.infoString("psi error:", 1);
            std::cout << enrm_zeta.infoString("zeta_error:", 1);
            std::cout << enrm_zeta_lap.infoString("zeta_lap_error",1);

            auto psi_host = ko::create_mirror_view(tgt_psi);
            auto zeta_host = ko::create_mirror_view(tgt_zeta);
            auto zeta_lap_host = ko::create_mirror_view(tgt_zeta_lap);
            ko::deep_copy(psi_host, tgt_psi);
            ko::deep_copy(zeta_host, tgt_zeta);
            ko::deep_copy(zeta_lap_err, tgt_zeta_lap);

            auto exact_psi_host = ko::create_mirror_view(psi_exact);
            auto exact_zeta_host = ko::create_mirror_view(zeta_exact);
            ko::deep_copy(exact_psi_host, psi_exact);
            ko::deep_copy(exact_zeta_host, zeta_exact);

            std::ostringstream ss;
            ss << "gmls_icostri" << i << "_latlon" << nlon << ".m";
            std::ofstream mfile(ss.str());
            ll.writeLatLonMeshgrid(mfile);
            ll.writeLatLonScalar(mfile, "psi", psi_host);
            ll.writeLatLonScalar(mfile, "zeta", zeta_host);
            ll.writeLatLonScalar(mfile, "zeta_lap", zeta_lap_host);
            ll.writeLatLonScalar(mfile, "psi_exact", exact_psi_host);
            ll.writeLatLonScalar(mfile, "zeta_exact", exact_zeta_host);
            mfile.close();
        }
    }
    output.computeRates();
    std::cout << output.infoString();
    output.writeInterpData();
    output.writeLapData();
}
ko::finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
    max_depth = 2;
    tgt_nlon = 180;
    mfilename = "compadre_tests.m";
    nlats = {91};//, 181, 361, 721};
    nlons = {180};//, 360, 720, 1440};
    eps = 2.0;
    order = 3;
    for (int i=1; i<argc; ++i) {
        const std::string& token = argv[i];
        if (token == "-d" || token == "-tree") {
            max_depth = std::stoi(argv[++i]);
        }
        else if (token == "-m") {
            mfilename = argv[++i];
        }
        else if (token == "-order") {
            order = std::stoi(argv[++i]);
        }
        else if (token == "-eps") {
            eps = std::stod(argv[++i]);
        }
    }
}

std::string Output::infoString() const {
    std::ostringstream ss;
    ss << "Order " << order << " error(interpolate(psi),psi)\n";
    ss << std::setw(20) << "nsrc" << std::setw(20) << "l1" << std::setw(20) << "l2" << std::setw(20) << "l2rate" << std::setw(20) << "linf" << std::setw(20) << "linfrate\n";
    for (int i=0; i<nsrc.size(); ++i) {
        ss << std::setw(20) << nsrc[i] << std::setw(20) << interp_l1[i] << std::setw(20)
           << interp_l2[i] << std::setw(20) << interp_l2rate[i] << std::setw(20)
           << interp_linf[i] << std::setw(20) << interp_linfrate[i] << "\n";
    }
    ss << "Order " << order << "error(laplacian(-psi),zeta)\n";
    ss << std::setw(20) << "nsrc" << std::setw(20) << "l1" << std::setw(20) << "l2" << std::setw(20) << "l2rate" << std::setw(20) << "linf" << std::setw(20) << "linfrate\n";
    for (int i=0; i<nsrc.size(); ++i) {
        ss << std::setw(20) << nsrc[i] << std::setw(20) << lap_l1[i] << std::setw(20) << lap_l2[i] << std::setw(20)
           << lap_l2rate[i] << std::setw(20) << lap_linf[i] << std::setw(20) << lap_linfrate[i] << "\n";
    }
    return ss.str();
}

void Output::computeRates() {
    for (int i=1; i<nsrc.size(); ++i) {
        const Real run = std::log(appxMeshSize(nsrc[i])) - std::log(appxMeshSize(nsrc[i-1]));
        interp_l2rate[i] = (std::log(interp_l2[i]) - std::log(interp_l2[i-1]))/run;
        interp_linfrate[i] = (std::log(interp_linf[i]) - std::log(interp_linf[i-1]))/run;
        lap_l2rate[i] = (std::log(lap_l2[i]) - std::log(lap_l2[i-1]))/run;
        lap_linfrate[i] = (std::log(lap_linf[i]) - std::log(lap_linf[i-1]))/run;
    }
}

void Output::writeInterpData() const {
    std::ostringstream ss;
    ss << "gmls_interp_icostri_order" << order << ".csv";
    std::ofstream f(ss.str());
    f << "nsrc" << "," << "l1" << "," << "l2" << ",l2rate" << "," << "linf" << ",linfrate\n";
    for (int i=0; i<nsrc.size(); ++i) {
        f << nsrc[i] << "," << interp_l1[i] << "," << interp_l2[i] << ","
          << interp_l2rate[i] << "," << interp_linf[i] << "," << interp_linfrate[i] << "\n";
    }
}

void Output::writeLapData() const {
    std::ostringstream ss;
    ss << "gmls_lap_icostri_order" << order << ".csv";
    std::ofstream f(ss.str());
    f << "nsrc" << "," << "l1" << "," << "l2" << ",l2rate" << ",linf" << ",linfrate\n";
    for (int i=0; i<nsrc.size(); ++i) {
        f << nsrc[i] << "," << lap_l1[i] << "," << lap_l2[i] << ","
          << lap_l2rate[i] << "," << lap_linf[i] <<"," << lap_linfrate[i] << "\n";
    }
}
