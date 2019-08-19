#include "LpmParticleSet.hpp"
#include "LpmPolymesh2d.hpp"
#include "LpmCoords.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

template <typename Geo> template <typename SeedType>
Particles<Geo>::Particles(const MeshSeed<SeedType>& seed, const int tree_depth) {
    Index nmaxverts, nmaxedges, nmaxfaces;
    seed.setMaxAllocations(nmaxverts, nmaxedges, nmaxfaces, tree_depth);
    PolyMesh2d<Geo, typename MeshSeed<SeedType>::faceKind> pmesh(nmaxverts,
        nmaxedges, nmaxfaces);
    pmesh.treeInit(tree_depth, seed);

    const Index n = pmesh.nFacesHost();

    phys_crds = crd_view("phys_crds", n);
    lag_crds = crd_view("lag_crds", n);
    weights = scalar_view(weightName(SeedType::geo::ndim),n);
    
    _phys_crds = ko::create_mirror_view(phys_crds);
    _lag_crds = ko::create_mirror_view(lag_crds);
    _weights = ko::create_mirror_view(weights);
}

}
