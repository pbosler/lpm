#ifndef LPM_SEED_READER_HPP
#define LPM_SEED_READER_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

struct QuadRectSeed {
    static constexpr Int nverts = 9;
    static constexpr Int nfaces = 4;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    static constexpr Int faceverts = 4;
    static constexpr std::string = LPM_MESH_SEED_DIR + "quadRectSeed.dat";
};

struct TriHexSeed {
    static constexpr Int nverts = 7;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    static constexpr Int faceverts = 3;
};

struct CubedSphereSeed {
    static constexpr Int nverts = 8;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef SphereGeometry geo;
    static constexpr Int faceverts = 4;
};

struct IcosTriSeed {
    static constexpr Int nverts = 12;
    static constexpr Int nfaces = 20;
    static constexpr Int nedges = 30;
    typedef SphereGeometry geo;
    static constexpr Int faceverts = 3;
};

template <typename SeedType> struct SeedReader {
    static constexpr Int ncrds = SeedType::nverts + SeedType::nfaces;
    
    ko::View<Real[ncrds][SeedType::geo::ndim],Host> scrds;
    ko::View<Int[SeedType::nedges][4],Host> sedges;
    ko::View<Int[SeedType::nfaces][SeedType::faceverts],Host> sfaceverts;
    ko::View<Int[SeedType::nfaces][SeedType::faceverts],Host> sfaceedges;
};

}
#endif
