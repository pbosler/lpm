#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSpherePoisson.hpp"
#include <iostream>
#include <sstream>
#ifdef HAVE_COMPADRE
#include "Compadre_GMLS.hpp"
#include "Compadre_Config.h"
#include "Compadre_Evaluator.hpp"
#include "Compadre_PointCloudSearch.hpp"
#endif
#include "Kokkos_Core.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
#ifdef HAVE_COMPADRE
ko::initialize(argc, argv);
{
    const int tree_depth = 3;
    
    Index nmaxverts, nmaxedges, nmaxfaces;
    std::ostringstream ss;

    MeshSeed<IcosTriSphereSeed> triseed;
    triseed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);

    SpherePoisson<TriFace> ic(nmaxverts, nmaxedges, nmaxfaces);
    ic.treeInit(tree_depth, triseed);
    ic.updateDevice();
    
    /// SETUP Target Mesh
    const int tgt_nlat = 91;
    const int tgt_nlon = 180;
    const int n_unif = tgt_nlat*tgt_nlon;
    const Real dlam = 2*PI/tgt_nlon;
    ko::View<Real[tgt_nlat]> lats("lats");
    ko::View<Real[tgt_nlon]> lons("lons");
    ko::parallel_for(tgt_nlat, KOKKOS_LAMBDA (int i) {
        lats(i) = -0.5*PI + i*dlam;
    });
    ko::parallel_for(tgt_nlon, KOKKOS_LAMBDA (int j) {
        lons(j) = j*dlam;
    }); 
    ko::View<Real[n_unif][3]> llmesh("llmesh");
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
    ko::View<Real[n_unif]> llharm54("sphharmonic54");
    ko::View<Real[n_unif][3]> llvel54("rh54_velocity");
    
    
    /// SETUP GMLS Neighborhoods
    const int gmls_order = 3;
    const int min_neighbors = Compadre::GMLS::getNP(gmls_order, 2);
    
    /// CURRENTLY: Neighborhoods constructed on host
    auto point_cloud_search(Compadre::CreatePointCloudSearch(llmesh));
    
    const Real eps_mult = 1.9;
    int estimated_upper_bound_number_neighbors = 
        point_cloud_search.getEstimatedNumberNeighborsUpperBound(min_neighbors, 3, eps_mult);
    ko::View<int**> neighbor_lists("neighbor_list", n_unif, estimated_upper_bound_number_neighbors);
    ko::View<int**>::HostMirror host_neighbors = ko::create_mirror_view(neighbor_lists);
    ko::View<Real*> neighborhood_radius("neighborhood_radius", n_unif);
    ko::View<Real*>::HostMirror host_radii = ko::create_mirror_view(neighborhood_radius);

    ko::View<Real**>::HostMirror target_coords = ko::create_mirror_view(llmesh);
    point_cloud_search.generateNeighborListsFromKNNSearch(target_coords, host_neighbors, host_radii, min_neighbors, 3, 
        eps_mult);
    
    ko::deep_copy(neighbor_lists, host_neighbors);
    ko::deep_copy(neighborhood_radius, host_radii);
    
    const std::string gmls_solver_name = "MANIFOLD";
    Compadre::GMLS gmls_scalar(gmls_order, gmls_solver_name.c_str(), gmls_order /*manifold order*/, 3);
    gmls_scalar.setProblemData(neighbor_lists, ic.getFaceCrds(), llmesh, neighborhood_radius);
    gmls_scalar.setReferenceOutwardNormalDirection(llmesh, true /* use to orient surface */);
}
ko::finalize();
return 0;
#else 
    return 1;
#endif
}
