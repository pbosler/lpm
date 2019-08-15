#include "LpmCompadre.hpp"

#ifdef HAVE_COMPADRE
#include "Compadre_PointCloudSearch.hpp"
#endif

namespace Lpm {

CompadreNeighborhoods::CompadreNeighborhoods(ko::View<const Real*[3], HostMem> host_src_crds, 
    ko::View<const Real*[3], HostMem> host_tgt_crds, const CompadreParams& params) {
    auto Compadre::point_cloud_search(Compadre::CreatePointCloudSearch(src_crds));
    const int est_upper_bound = point_cloud_search.getEstimatedNumberNeighborsUpperBound(params.min_neighbors,
        params.ambient_dim, params.gmls_eps_mult);
    
    neighbor_lists = ko::View<Index**>("neighbor_lists", host_tgt_crds.extent(0), est_upper_bound);
    neighborhood_radii = ko::View<Real*>("neighborhood_radii", host_tgt_crds.extent(0));
    
    // neighborhoods built on host
    auto host_neighbors = ko::create_mirror_view(neighbor_lists);
    auto host_radii = ko::create_mirror_view(neighborhood_radii);
    point_cloud_search.generateNeighborListsFromKNNSearch(host_tgt_crds, host_neighbors, host_radii,
        params.min_neighbors, params.ambient_dim, params.gmls_eps_mult);
    
    // copy to device
    ko::deep_copy(neighbor_lists, host_neighbors);
    ko::deep_copy(neighborhood_radii, host_radii);
}

Real CompadreNeighborhoods::minRadius() const {
    Real result;
    ko::parallel_reduce("MinReduce", neighborhood_radii.extent(0), KOKKOS_LAMBDA (int i, Real& r) {
        if (neighborhood_radii(i) < r) r = neighborhood_radii(i);
    }, result);
    return result;
}

Real CompadreNeighborhoods::maxRadius() const {
    Real result;
    ko::parallel_reduce("MaxReduce", neighborhood_radii.extent(0), KOKKOS_LAMBDA(int i, Real& r) {
        if (neighborhood_radii(i) > r) r = neighborhood_radii(i);
    }, result);
    return result;
}

Compadre::GMLS scalarGMLS(const ko::View<Real*[3],DevMem> src_crds, const ko::View<Real*[3],DevMem> tgt_crds,
    const CompadreNeighborhoods& nn, const CompadreParams& params, 
    const std::vector<Compadre::TargetOperation>& gmls_ops) {
    const std::string solver_name = "MANIFOLD";
    Compadre::GMLS result(params.gmls_order, solver_name.c_str(), params.gmls_manifold_order, params.ambient_dim);
    result.setProblemData(nn.neighbor_lists, src_crds, tgt_crds, nn.neighborhood_radii);
    result.setReferenceOutwardNormalDirection(tgt_crds);
    result.setCurvatureWeightingType(Compadre::WeightingFunctionType::Power);
    result.setCurvatureWeightingPower(params.gmls_manifold_weight_pow);
    result.setWeightingType(Compadre::WeightingFunctionType::Power);
    result.setWeightingPower(params.gmls_weight_pow);
    result.addTargets(gmls_ops);
    result.generateAlphas();
    return result;
}

}