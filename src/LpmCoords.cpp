#include "LpmCoords.hpp"
#include <random>

namespace Lpm {

template <typename Geo> 
std::string Coords<Geo>::infoString(const std::string& label) const {
    std::ostringstream oss;
    oss << "Coords " << label << " info: nh = (" << _nh(0) << ") of nmax = " << _nmax << " in memory" << std::endl; 
    for (Index i=0; i<_nmax; ++i) {
        if (i==_nh(0)) oss << "---------------------------------" << std::endl;
        oss << label << ": (" << i << ") : ";
        for (Int j=0; j<Geo::ndim; ++j) 
            oss << _host_crds(i,j) << " ";
        oss << std::endl;
    }
    return oss.str();
}

template <typename Geo> void Coords<Geo>::initRandom(const Real max_range, const Int ss) {
    unsigned seed = 0 + ss;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<Real> randDist(-max_range, max_range);
    for (Index i=0; i<_nmax; ++i) {
        Real cvec[Geo::ndim];
        for (Int j=0; j<Geo::ndim; ++j) {
            cvec[j] = randDist(generator);
        }
        insertHost(cvec);
    }
    updateDevice();
}

template <> void Coords<SphereGeometry>::initRandom(const Real max_range, const Int ss) {
    unsigned seed = 0 + ss;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<Real> randDist(-1.0, 1.0);
    for (Index i=0; i<_nmax; ++i) {
        Real uu = randDist(generator);
        Real vv = randDist(generator);
        while (uu*uu + vv*vv > 1.0) {
            uu = randDist(generator);
            vv = randDist(generator);
        }        
        const Real uv2 = uu*uu + vv*vv;
        const Real uvr = std::sqrt(1-uv2);
        const Real cvec[3] = {2*uu*uvr*max_range, 2*vv*uvr*max_range, (1-2*uv2)*max_range};
        insertHost(cvec);
    }
    updateDevice();
}

template <typename Geo> template <typename SeedType>
void Coords<Geo>::initBoundaryCrdsFromSeed(const MeshSeed<SeedType>& seed) {
    LPM_THROW_IF(_nmax < SeedType::nverts, "Coords::initBoundaryCrdsFromSeed error: not enough memory.");
    for (int i=0; i<SeedType::nverts; ++i) {
        for (int j=0; j<Geo::ndim; ++j) {
            _host_crds(i,j) = seed.scrds(i,j);
        }
    }
    _nh(0) = SeedType::nverts;
}

template <typename Geo> template <typename SeedType>
void Coords<Geo>::initInteriorCrdsFromSeed(const MeshSeed<SeedType>& seed) {
    LPM_THROW_IF(_nmax < SeedType::nfaces, "Coords::initInteriorCrdsFromSeed error: not enough memory.");
    for (int i=0; i<SeedType::nfaces; ++i) {
        for (int j=0; j<Geo::ndim; ++j) {
            _host_crds(i,j) = seed.scrds(SeedType::nverts + i, j);
        }
    }
    _nh(0) = SeedType::nfaces;
}

/// ETI
template class Coords<PlaneGeometry>;
template class Coords<SphereGeometry>;

template void Coords<PlaneGeometry>::initBoundaryCrdsFromSeed(const MeshSeed<TriHexSeed>& seed);
template void Coords<PlaneGeometry>::initInteriorCrdsFromSeed(const MeshSeed<TriHexSeed>& seed);
template void Coords<PlaneGeometry>::initBoundaryCrdsFromSeed(const MeshSeed<QuadRectSeed>& seed);
template void Coords<PlaneGeometry>::initInteriorCrdsFromSeed(const MeshSeed<QuadRectSeed>& seed);
template void Coords<SphereGeometry>::initBoundaryCrdsFromSeed(const MeshSeed<CubedSphereSeed>& seed);
template void Coords<SphereGeometry>::initInteriorCrdsFromSeed(const MeshSeed<CubedSphereSeed>& seed);
template void Coords<SphereGeometry>::initBoundaryCrdsFromSeed(const MeshSeed<IcosTriSphereSeed>& seed);
template void Coords<SphereGeometry>::initInteriorCrdsFromSeed(const MeshSeed<IcosTriSphereSeed>& seed);

}
