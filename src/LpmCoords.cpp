#include "LpmCoords.hpp"
#include <random>

namespace Lpm {

template <typename Geo> void Coords<Geo>::printcrds(const std::string& label) const {
    std::ostringstream oss;
    for (Index i=0; i<_nmax; ++i) {
        oss << label << ": (" << i << ") : ";
        std::cout << oss.str();
        for (Int j=0; j<Geo::ndim; ++j) 
            std::cout << _host_crds(i,j) << " ";
        std::cout << std::endl;
        oss.str("");
    }
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


/// ETI
template class Coords<PlaneGeometry>;
template class Coords<SphereGeometry>;

}
