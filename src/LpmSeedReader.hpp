#ifndef LPM_SEED_READER_HPP
#define LPM_SEED_READER_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include <string>

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

struct QuadRectSeed {
    static constexpr Int nverts = 9;
    static constexpr Int nfaces = 4;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    static constexpr Int nfaceverts = 4;
    static std::string filename() {return "quadRectSeed.dat";}
    static std::string idString() {return "QuadRectSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
};

struct TriHexSeed {
    static constexpr Int nverts = 7;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    static constexpr Int nfaceverts = 3;
    static std::string filename() {return "triHexSeed.dat";}
    static std::string idString() {return "TriHexSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
};

struct CubedSphereSeed {
    static constexpr Int nverts = 8;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef SphereGeometry geo;
    static constexpr Int nfaceverts = 4;
    static std::string filename() {return "cubedSphereSeed.dat";}
    static std::string idString() {return "CubedSphereSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
};

struct IcosTriSphereSeed {
    static constexpr Int nverts = 12;
    static constexpr Int nfaces = 20;
    static constexpr Int nedges = 30;
    typedef SphereGeometry geo;
    static constexpr Int nfaceverts = 3;
    static std::string filename() {return "icosTriSphereSeed.dat";}
    static std::string idString() {return "IcosTriSphereSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
};

template <typename SeedType> struct SeedReader {
    static constexpr Int ncrds = SeedType::nverts + SeedType::nfaces;
    
    ko::View<Real[ncrds][SeedType::geo::ndim],Host> scrds;
    ko::View<Index[SeedType::nedges][4],Host> sedges;
    ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> sfaceverts;
    ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> sfaceedges;
    
    SeedReader() : scrds("seed coords"), sedges("seed edges"), sfaceverts("seed face vertices"), 
        sfaceedges("seed face edges") {readfile();}
    
    static std::string idString() {return SeedType::idString();}
    
    std::string fullFilename() const;
    
    void readfile();
    
    void setMaxAllocations(Index& nboundary, Index& nedges, Index& nfaces, const Int lev) const;
    
    std::string infoString() const;
};

}
#endif
