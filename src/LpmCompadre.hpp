#ifndef LPM_COMPADRE_HPP
#define LPM_COMPADRE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmCoords.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmKokkosUtil.hpp"
#include <vector>
// #ifdef HAVE_COMPADRE
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
    
    std::string infoString(const int tab_level=0) const;
    
    CompadreParams() : gmls_eps_mult(2.0), gmls_order(3), gmls_manifold_order(3), gmls_weight_pow(2), 
        gmls_manifold_weight_pow(2), ambient_dim(3), topo_dim(2), 
        min_neighbors(Compadre::GMLS::getNP(gmls_order, topo_dim)) {}
    
    CompadreParams(const Int ord) : gmls_eps_mult((ord > 4 ? 3.0 : 2.0)), gmls_order(ord), gmls_manifold_order(ord),
    gmls_weight_pow(2), gmls_manifold_weight_pow(2), ambient_dim(3), topo_dim(2),
    min_neighbors(Compadre::GMLS::getNP(ord, topo_dim)) {}
};

struct CompadreNeighborhoods {
    ko::View<Index**> neighbor_lists;
    ko::View<Real*> neighborhood_radii;
    
//     ko::View<Index**,HostMem> host_neighbors;
//     ko::View<Real*, HostMem> host_radii;
    
    CompadreNeighborhoods(typename ko::View<Real*[3]>::HostMirror host_src_crds, 
        typename ko::View<Real*[3]>::HostMirror host_tgt_crds, const CompadreParams& params) ;
    
    Real minRadius() const;
    Real maxRadius() const;
    Int maxNeighbors() const;
    Int minNeighbors() const;
    
    std::string infoString(const int tab_level=0) const;
};

Compadre::GMLS scalarGMLS(const ko::View<Real*[3],DevMem> src_crds, const ko::View<Real*[3],DevMem> tgt_crds,
    const CompadreNeighborhoods& nn, const CompadreParams& params, 
    const std::vector<Compadre::TargetOperation>& gmls_ops);

Compadre::GMLS vectorGMLS(const ko::View<Real*[3],DevMem> src_crds, const ko::View<Real*[3],DevMem> tgt_crds,
    const CompadreNeighborhoods& nn, const CompadreParams& params, 
    const std::vector<Compadre::TargetOperation>& gmls_ops);

}
// #endif
#endif