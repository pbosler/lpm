#ifndef LPM_COMPADRE_HPP
#define LPM_COMPADRE_HPP

#ifdef HAVE_COMPADRE

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmCoords.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmKokkosUtil.hpp"
#include <vector>


#include "Compadre_GMLS.hpp"
#include "Compadre_Config.h"
#include "Compadre_Operators.hpp"
#include "Compadre_Evaluator.hpp"


namespace Lpm {

struct CompadreParams {
    Real gmls_eps_mult;
    Int gmls_order;
    Int gmls_manifold_order;
    Real gmls_weight_pow;
    Real gmls_manifold_weight_pow;
    Int ambient_dim;
    Int topo_dim;
    Int min_neighbors;
    
    std::string infoString() const;
    
    CompadreParams() : gmls_eps_mult(2), gmls_order(3), gmls_manifold_order(3), gmls_weight_pow(2), 
        gmls_manifold_weight_pow(2), ambient_dim(3), topo_dim(2), 
        min_neighbors(Compadre::GMLS::getNP(gmls_order, topo_dim) {}
};

struct CompadreNeighborhoods {
    ko::View<Index**> neighbor_lists;
    ko::View<Real*> neighborhood_radii;
    
    CompadreNeighborhoods(ko::View<const Real*[3], HostMem> host_src_crds, 
        ko::View<const Real*[3], HostMem> host_tgt_crds, const CompadreParams& params) ;
    
    Real minRadius() const;
    Real maxRadius() const;
};

Compadre::GMLS scalarGMLS(const ko::View<Real*[3],DevMem> src_crds, const ko::View<Real*[3],DevMem> tgt_crds,
    const CompadreNeighborhoods& nn, const CompadreParams& params, 
    const std::vector<Compadre::TargetOperation>& gmls_ops);

Compadre::GMLS vectorGMLS(const ko::View<Real*[3],DevMem> src_crds, const ko::View<Real*[3],DevMem> tgt_crds,
    const CompadreNeighborhoods& nn, const CompadreParams& params, 
    const std::vector<Compadre::TargetOperation>& gmls_ops);

}
#endif
#endif