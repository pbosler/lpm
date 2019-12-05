#ifndef LPM_MESH_SEED_HPP
#define LPM_MESH_SEED_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include <string>

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

struct TriFace {
    static constexpr Int nverts = 3;
};
struct QuadFace {
    static constexpr Int nverts = 4;
};

struct VoronoiFace {
    static constexpr Int nverts = 10; // nverts = upper bound for this type
};

/**
    Each type of seed below is a initializer for the tree-based meshes employed by LPM.

    Required public typedefs:
        geo = Geometry type.  See LpmGeometry.hpp
    Required static members:
        nverts : number of vertices in the seed mesh (tree level 0)
        nfaces : number of faces in the seed mesh
        nedges : number of edges in the seed MeshSeed()
        nfaceverts : number of vertices per face
    Required static methods:
        filename : return the filename with seed data produced by meshSeeds.py
        idString : return the name of the seed (used for console output, debugging, etc.)
        nVerticesAtTreeLevel(const Int lev) : return the number of vertices in a mesh at tree depth = lev
        nFacesAtTreeLevel(const Int lev) : return the number of leaf faces in a mesh at tree depth = lev
        nEdgesAtTreeLevel(const Int nv, const Int nf): return the number of leaf edges in a mesh with nv vertices and nf faces
*/
struct QuadRectSeed {
    static constexpr Int nverts = 9;
    static constexpr Int nfaces = 4;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    typedef QuadFace faceKind;
    static constexpr Int nfaceverts = 4;
    static std::string filename() {return "quadRectSeed.dat";}
    static std::string idString() {return "QuadRectSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
    static constexpr bool isDual = false;
};

struct TriHexSeed {
    static constexpr Int nverts = 7;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef PlaneGeometry geo;
    typedef TriFace faceKind;
    static constexpr Int nfaceverts = 3;
    static std::string filename() {return "triHexSeed.dat";}
    static std::string idString() {return "TriHexSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
    static constexpr bool isDual = false;
};

struct CubedSphereSeed {
    static constexpr Int nverts = 8;
    static constexpr Int nfaces = 6;
    static constexpr Int nedges = 12;
    typedef SphereGeometry geo;
    typedef QuadFace faceKind;
    static constexpr Int nfaceverts = 4;
    static std::string filename() {return "cubedSphereSeed.dat";}
    static std::string idString() {return "CubedSphereSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
    static constexpr bool isDual = false;
};

struct IcosTriSphereSeed {
    static constexpr Int nverts = 12;
    static constexpr Int nfaces = 20;
    static constexpr Int nedges = 30;
    typedef SphereGeometry geo;
    typedef TriFace faceKind;
    static constexpr Int nfaceverts = 3;
    static std::string filename() {return "icosTriSphereSeed.dat";}
    static std::string idString() {return "IcosTriSphereSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
    static constexpr bool isDual = false;
};

struct IcosTriDualSeed {
    static constexpr Short nverts = 20;
    static constexpr Short nfaces = 12;
    static constexpr Short nedges = 30;
    typedef SphereGeometry geo;
    typedef VoronoiFace faceKind;
    static constexpr Short nfaceverts = 5;
    static std::string filename() {return "IcosTriDualSeed.dat";}
    static std::string idString() {return "IcosTriDualSeed";}
    static Index nFacesAtTreeLevel(const Int lev);
    static Index nVerticesAtTreeLevel(const Int lev);
    static Index nEdgesAtTreeLevel(const Index nv, const Index nf);
    static constexpr bool isDual = true;
};

/**
    The MeshSeed class is templated on the seed type (listed above).

    *** All of this class's methods execute on Host ***

    It provides run-time info based on the SeedType.

    MeshSeed is responsible for determining the memory required to construct a mesh.
*/
template <typename SeedType> struct MeshSeed {
    /// Number of coordinates (both vertices and faces) in the seed file.
    static constexpr Int ncrds = SeedType::nverts + SeedType::nfaces;

    /// Host views to load data read from file
    ko::View<Real[ncrds][SeedType::geo::ndim],Host> scrds;
    ko::View<Index[SeedType::nedges][8],Host> sedges;
    ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> sfaceverts;
    ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> sfaceedges;

    /// constructor.  Automatically reads data file.
    MeshSeed() : scrds("seed coords"), sedges("seed edges"), sfaceverts("seed face vertices"),
        sfaceedges("seed face edges") {readfile();}

    /// idString
    static std::string idString() {return SeedType::idString();}

    /// Concatenates directory info with SeedType::filename()
    std::string fullFilename() const;

    /** Return the required memory allocations for a mesh tree of depth = lev
        nboundary = n vertices in a typical mesh (only different for high-order meshes, which are not implemented yet)
        nedges = n edges required by the mesh tree
        nfaces = n faces required by the mesh tree
    */
    void setMaxAllocations(Index& nboundary, Index& nedges, Index& nfaces, const Int lev) const;

    /// Return runtime info about this object
    std::string infoString() const;

    /// Compute & return the area of the seed's initial faces.
    Real faceArea(const Int ind) const;

    protected:
        /// Read data file.
        void readfile();

    private:





};



}
#endif
