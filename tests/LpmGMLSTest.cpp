#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSpherePoisson.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmVtkIO.hpp"
#include <iostream>
#include <sstream>
#ifdef HAVE_COMPADRE
#include "Compadre_GMLS.hpp"
#include "Compadre_Config.h"
#include "Compadre_Evaluator.hpp"
#include "Compadre_PointCloudSearch.hpp"
#include "Compadre_Operators.hpp"
#endif
#include "Kokkos_Core.hpp"
#include <fstream>
#include <cmath>

using namespace Lpm;

struct Input {
    Real gmls_epsilon_multiplier;
    Int nlon;
    Int lpm_tree_depth;
    Int gmls_order;
    Int gmls_manifold_order;
    Real manifold_weight_power;
    Real gmls_weight_power;

    std::string mfile_out;

    Input(int argc, char* argv[]);
    
    std::string infoString() const;
};


struct Cart2LL {
    ko::View<Real*[3]> cart;
    ko::View<Real*[2]> ll;
    
    Cart2LL(ko::View<Real*[2]> llout, const ko::View<Real*[3]> cartin) :cart(cartin), ll(llout) {}
    
    void operator() (const int i) const {
        auto xyz = slice(cart,i);
        const Real lon = SphereGeometry::longitude(xyz);
        const Real lat = SphereGeometry::latitude(xyz);
        const Real coslat = std::cos(lat);
        const Real sinlat = std::sin(lat);
        const Real coslon = std::cos(lon);
        const Real sinlon = std::sin(lon);
        const Real unit_lon[3] = {-sinlon, coslon, 0};
        const Real unit_lat[3] = {-sinlat*coslon, -sinlat*sinlon, coslat};
        ll(i,0) = SphereGeometry::dot(unit_lon, xyz);
        ll(i,1) = SphereGeometry::dot(unit_lat, xyz);
    }
};

struct PackScalarForContourPlot {
    ko::View<Real*> data;
    ko::View<Real**> cdata;
    int nlon;
    int nlat;
    
    PackScalarForContourPlot(ko::View<Real**> cdataout, ko::View<Real*> datain, const int nlonin) : data(datain), cdata(cdataout), nlon(nlonin), nlat(nlonin/2 + 1) {}
    
    void operator() (const int k) const {
        const int i = k/nlon;
        const int j = k%nlon;
        cdata(i,j) = data(k);
    }
};

int main(int argc, char* argv[]) {
#ifdef HAVE_COMPADRE
Input input(argc, argv);
std::cout << input.infoString();
ko::initialize(argc, argv);
{
    /// Build an LPM particle set, with mesh
    const int tree_depth = input.lpm_tree_depth;
    
    Index nmaxverts, nmaxedges, nmaxfaces;
    std::ostringstream ss;

    MeshSeed<IcosTriSphereSeed> triseed;
    triseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);

    SpherePoisson<TriFace> ic(nmaxverts, nmaxedges, nmaxfaces);
    ic.treeInit(tree_depth, triseed);
    ic.updateDevice();
    ic.init();

    
    /// Setup Target Mesh (uniform lat-lon grid)
    const int tgt_nlat = input.nlon/2 + 1;
    const int tgt_nlon = input.nlon;
    const int n_unif = tgt_nlat*tgt_nlon;
    const Real dlam = 2*PI/tgt_nlon;
    ko::View<Real*> lats("lats", tgt_nlat);
    ko::View<Real*> lons("lons", tgt_nlon);
    ko::parallel_for(tgt_nlat, KOKKOS_LAMBDA (int i) {
        lats(i) = -0.5*PI + i*dlam;
    });
    ko::parallel_for(tgt_nlon, KOKKOS_LAMBDA (int j) {
        lons(j) = j*dlam;
    }); 
    ko::View<Real*[3]> llmesh("llmesh", n_unif);
    ko::parallel_for(tgt_nlat, KOKKOS_LAMBDA (int i) {
        const int start_ind = i*tgt_nlon;
        const Real z = std::sin(lats(i));
        const Real coslat = std::cos(lats(i));
        for (int j=0; j<tgt_nlon; ++j) {
            const Real x = std::cos(lons(j))*coslat;
            const Real y = std::sin(lons(j))*coslat;
            llmesh(start_ind+j,0) = x;
            llmesh(start_ind+j,1) = y;
            llmesh(start_ind+j,2) = z;
        }
    });
    ko::View<Real*[3]>::HostMirror llhost = ko::create_mirror_view(llmesh);
    ko::deep_copy(llhost, llmesh);
    
    ko::View<Real*> llharm54("sphharmonic54", n_unif);
    ko::View<Real*[3]> llvel54("rh54_velocity", n_unif);
    ko::View<Real*> llpsi54("streamfn54", n_unif);
    ko::parallel_for(n_unif, KOKKOS_LAMBDA (int i) {
        auto x = slice(llmesh,i);
        llharm54(i) = SphHarm54(x);
        llpsi54(i) = -llharm54(i)/30.0;
        ko::Tuple<Real,3> vel = RH54Velocity(x);
        for (int j=0; j<3; ++j) {
            llvel54(i,j) = vel[j];
        }
    });
    ko::View<Real*[2]> exact_uv("uv_exact", n_unif);
    ko::parallel_for(n_unif, Cart2LL(exact_uv, llvel54));
    auto uv_host = ko::create_mirror_view(exact_uv);
    ko::deep_copy(uv_host, exact_uv);
    
    auto llzeta = ko::create_mirror_view(llharm54);
    auto llu = ko::create_mirror_view(llvel54);
    ko::deep_copy(llzeta, llharm54);
    ko::deep_copy(llu, llvel54);
    
    
    /// SETUP GMLS Neighborhoods
    const int ambient_dim = 3;
    const int topo_dim = 2;
    const int gmls_order = input.gmls_order;
    const int manifold_order = input.gmls_manifold_order;
    const int min_neighbors = Compadre::GMLS::getNP(gmls_order, topo_dim);  
    
    /// CURRENTLY: Neighborhoods constructed on host
    auto point_cloud_search(Compadre::CreatePointCloudSearch(ic.getFaceCrdsHost()));
    
    const Real eps_mult = input.gmls_epsilon_multiplier;
    int estimated_upper_bound_number_neighbors = 
        point_cloud_search.getEstimatedNumberNeighborsUpperBound(min_neighbors, ambient_dim, eps_mult);
    ko::View<int**> neighbor_lists("neighbor_list", n_unif, estimated_upper_bound_number_neighbors);
    ko::View<int**>::HostMirror host_neighbors = ko::create_mirror_view(neighbor_lists);
    ko::View<Real*> neighborhood_radius("neighborhood_radius", n_unif);
    ko::View<Real*>::HostMirror host_radii = ko::create_mirror_view(neighborhood_radius);

    point_cloud_search.generateNeighborListsFromKNNSearch(llhost, host_neighbors, host_radii, min_neighbors, ambient_dim, 
        eps_mult);
    
    ko::deep_copy(neighbor_lists, host_neighbors);
    ko::deep_copy(neighborhood_radius, host_radii);
    
    Real min_radius;
    Real max_radius;
    ko::parallel_reduce("MinReduce", n_unif, KOKKOS_LAMBDA (int i, Real& r) {
        if (neighborhood_radius(i) < r) r = neighborhood_radius(i);
    }, min_radius);
    ko::parallel_reduce("MaxReduce", n_unif, KOKKOS_LAMBDA (int i, Real& r) {
        if (neighborhood_radius(i) > r) r = neighborhood_radius(i);
    }, max_radius);
    std::cout << "min/max neighborhood_radius = (" << min_radius << ", " << max_radius << ")\n";

    /// Define GMLS scalar operators
    const std::string gmls_solver_name = "MANIFOLD";
    Compadre::GMLS gmls_scalar(gmls_order, gmls_solver_name.c_str(), gmls_order, 3);
    gmls_scalar.setProblemData(neighbor_lists, ic.getFaceCrds(), llmesh, neighborhood_radius);
    gmls_scalar.setReferenceOutwardNormalDirection(llmesh);
    
    std::vector<Compadre::TargetOperation> gmls_scalar_ops = {Compadre::ScalarPointEvaluation,
                                                              Compadre::LaplacianOfScalarPointEvaluation};
    gmls_scalar.addTargets(gmls_scalar_ops);
    
    /// Setup GMLS manifold reconstruction
    const int curve_power = input.manifold_weight_power;
    const int kernel_power = input.gmls_weight_power;
    gmls_scalar.setCurvatureWeightingType(Compadre::WeightingFunctionType::Power);
    gmls_scalar.setWeightingType(Compadre::WeightingFunctionType::Power);
    gmls_scalar.setCurvatureWeightingPower(curve_power);
    gmls_scalar.setWeightingPower(kernel_power);
    
    /// Compute GMLS coefficients
    gmls_scalar.generateAlphas();
    
    /// Define GMLS vector operators
    Compadre::GMLS gmls_vector(Compadre::ReconstructionSpace::VectorTaylorPolynomial,
        Compadre::VaryingManifoldVectorPointSample, gmls_order, gmls_solver_name.c_str(), manifold_order, ambient_dim);
    gmls_vector.setProblemData(neighbor_lists, ic.getFaceCrds(), llmesh, neighborhood_radius);
    std::vector<Compadre::TargetOperation> gmls_vector_ops = {Compadre::VectorPointEvaluation,
                                                              /*Compadre::DivergenceOfVectorPointEvaluation,*/
                                                              Compadre::CurlOfVectorPointEvaluation};
    gmls_vector.addTargets(gmls_vector_ops);
    gmls_vector.setCurvatureWeightingType(Compadre::Power);
    gmls_vector.setWeightingType(Compadre::Power);
    gmls_vector.setCurvatureWeightingPower(curve_power);
    gmls_vector.setWeightingPower(kernel_power);
    gmls_vector.generateAlphas();
    
    Compadre::Evaluator eval_scalar(&gmls_scalar);
    Compadre::Evaluator eval_vector(&gmls_vector);

    auto llvort_sol = eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(ic.ffaces, Compadre::ScalarPointEvaluation,
         Compadre::PointSample);
    auto llpsi_sol = eval_scalar.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(ic.psifaces,
         Compadre::LaplacianOfScalarPointEvaluation,  Compadre::PointSample);
    auto llvel_interp = eval_vector.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMem>(ic.ufacesexact,
         Compadre::VectorPointEvaluation, Compadre::VectorPointSample);
//     auto llvel_div = eval_vector.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMem>(ic.ufacesexact,
//          Compadre::DivergenceOfVectorPointEvaluation, Compadre::VectorPointSample);
    auto llvel_curl = eval_vector.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMem>(ic.ufacesexact, 
         Compadre::CurlOfVectorPointEvaluation, Compadre::VectorPointSample);
    ko::View<Real*> curl("surf_curl", n_unif);
    ko::parallel_for(n_unif, KOKKOS_LAMBDA (int i) {
        curl(i) = SphereGeometry::dot(slice(llvel_curl,i),slice(llmesh,i));
    });
    
    auto llvort_host = ko::create_mirror_view(llvort_sol);
    auto llpsi_host = ko::create_mirror_view(llpsi_sol);
    auto llvel_interp_host = ko::create_mirror_view(llvel_interp);
    auto llcurl_host = ko::create_mirror_view(curl);
    ko::deep_copy(llvort_host, llvort_sol);
    ko::deep_copy(llpsi_host, llpsi_sol);
    ko::deep_copy(llvel_interp_host, llvel_interp);
    ko::deep_copy(llcurl_host, curl);
    
    Real scalar_interp_linf = 0.0;
    ko::View<Real*> scalar_interp_error("scalar_interp_error", n_unif);
    ko::parallel_for(n_unif, KOKKOS_LAMBDA (int i) {
        scalar_interp_error(i) = abs(llvort_sol(i) - llharm54(i));
    });
    ko::parallel_reduce("MaxReduce", n_unif, KOKKOS_LAMBDA (int i, Real& err) {
        if (scalar_interp_error(i) > err) err = scalar_interp_error(i);
    }, scalar_interp_linf);
    std::cout << "scalar_interp_linf = " << scalar_interp_linf << '\n';
    
    Real scalar_lap_linf = 0.0;
    ko::View<Real*> scalar_lap_error("scalar_lap_error", n_unif);
    ko::parallel_for(n_unif, KOKKOS_LAMBDA (int i) {
        scalar_lap_error(i) = abs(llpsi54(i) + llpsi_sol(i));
    });
    ko::parallel_reduce("MaxReduce", n_unif, KOKKOS_LAMBDA (int i, Real& err) {
        if (scalar_lap_error(i) > err) err = scalar_lap_error(i);
    }, scalar_lap_linf);
    std::cout << "scalar_laplacian_linf = " << scalar_lap_linf << '\n';
    
    std::ofstream f(input.mfile_out);
    auto mask = ic.getFacemaskHost();
    auto src_crds = ic.getFaceCrdsHost();
    f << "src_lats = [";
    for (int i=0; i<ic.nfacesHost(); ++i) {
        if (!mask(i)) {
            f << SphereGeometry::latitude(slice(src_crds,i)) << ' ';
        }
    }
    f << "];\n";
    f << "src_lons = [";
    for (int i=0; i<ic.nfacesHost(); ++i) {
        if (!mask(i)) {
            f << SphereGeometry::longitude(slice(src_crds,i)) << ' ';
        }
    }
    f << "];\n";
    f << "src_zeta = [";
    for (int i=0; i<ic.nfacesHost(); ++i) {
        if (!mask(i)) {
            f << ic.ffaces(i) << ' ';
        }
    }
    f << "];\n";
    f << "neighbor_lists = [";
    for (int i=0; i<host_neighbors.extent(0); ++i) {
        for (int j=0; j<host_neighbors.extent(1); ++j) {
            f << host_neighbors(i,j) << (j < host_neighbors.extent(1)-1 ? "," : (i < host_neighbors.extent(0)-1 ? ";" : "];\n"));
        }
    }
    f << "neighborhood_radius = [";
    for (int i=0; i<host_radii.extent(0); ++i) {
        f << host_radii(i) << ' ';
    }
    f << "];\n";
    f << "lats = [";
    for (int i=0; i<n_unif; ++i) {
        f << SphereGeometry::latitude(slice(llhost,i)) << ' ';
    }
    f << "];\n";
    f << "lons = [";
    for (int i=0; i<n_unif; ++i) {
        f << SphereGeometry::longitude(slice(llhost,i)) << ' ';
    }
    f << "];\n";
    f << "exact_sphharm54 = [";
    for (int i=0; i<n_unif; ++i) {
        f << llzeta(i) << ' ';
    }
    f << "];\n";
    f << "gmls_sphharm54 = [";
    for (int i=0; i<n_unif; ++i) {
        f << llvort_host(i) << ' ';
    }
    f << "];\n";
    f << "exact_stream54 = [";
    for (int i=0; i<n_unif; ++i) {
        f << llpsi54(i) << ' ';
    }
    f << "];\n";
    f << "gmls_stream54 = [";
    for (int i=0; i<n_unif; ++i) {
        f << llpsi_host(i) << ' ';
    }
    f << "];\n";
    f << "gmls_curl = [";
    for (int i=0; i<n_unif; ++i) {
        f << llcurl_host(i) << ' ';
    }
    f << "];\n";
    f << "tri_src = delaunay(src_lons, src_lats);\n";
    f << "tri_tgt = delaunay(lons, lats);\n";
    f.close();
}
ko::finalize();
return 0;
#else 
    return 1;
#endif
}

Input::Input(int argc, char* argv[]) {
    gmls_epsilon_multiplier = 2;
    nlon = 180;
    lpm_tree_depth = 4;
    gmls_order = 3;
    gmls_manifold_order = 3;
    manifold_weight_power = 2;
    gmls_weight_power = 2;
    mfile_out = "gmls_tests.m";
    for (int i=1; i<argc; ++i) {
        const std::string& token = argv[i];
        if (token == "-o") {
            mfile_out = argv[++i];
        }
        else if (token == "-eps") {
            gmls_epsilon_multiplier = std::stod(argv[++i]);
        }
        else if (token == "-nlon") {
            nlon = std::stoi(argv[++i]);
        }
        else if (token == "-tree") {
            lpm_tree_depth = std::stoi(argv[++i]);
        }
        else if (token == "-gmls") {
            gmls_order = std::stoi(argv[++i]);
        }
        else if (token == "-manif") {
            gmls_manifold_order = std::stoi(argv[++i]);
        }
        else if (token == "-pow") {
            gmls_weight_power = std::stod(argv[++i]);
            manifold_weight_power = gmls_weight_power;
        }
    }
}

std::string Input::infoString() const {
    std::ostringstream ss;
    ss << "LPM GMLS TEST info:\n";
    ss << "\toutput_file = " << mfile_out << '\n';
    ss << "\tSource info (LPM):\n";
    ss << "\t\ttree_depth = " << lpm_tree_depth << '\n';
    ss << "\tTarget info (LL mesh):\n";
    ss << "\t\tnlon = " << nlon << '\n';
    ss << "\tGMLS info:\n";
    ss << "\t\tgmls_epsilon_multiplier = " << gmls_epsilon_multiplier << '\n';
    ss << "\t\tgmls_order = " << gmls_order << '\n';
    ss << "\t\tgmls_manifold_oder = " << gmls_manifold_order << '\n';
    ss << "\t\tgmls_weight_power = " << gmls_weight_power << '\n';
    return ss.str();
}
