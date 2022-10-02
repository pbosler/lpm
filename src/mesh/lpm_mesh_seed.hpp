#ifndef LPM_MESH_SEED_HPP
#define LPM_MESH_SEED_HPP
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include <string>

#include "Kokkos_Core.hpp"

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

/*

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
    id_string : return the name of the seed (used for console output, debugging, etc.)
    n_vertices_at_tree_level(const Int lev) : return the number of vertices in a mesh at tree depth = lev
    n_faces_at_tree_level(const Int lev) : return the number of leaf faces in a mesh at tree depth = lev
    n_edges_at_tree_level(const Int nv, const Int nf): return the number of leaf edges in a mesh with nv vertices and nf faces
*/

/** @brief Seed for planar meshes of quadrilaterals, free boundary conditions.

  @image html quadRectSeed.pdf "QuadRectSeed"
*/
struct QuadRectSeed {
  static constexpr Int nverts = 9;
  static constexpr Int nfaces = 4;
  static constexpr Int nedges = 12;
  typedef PlaneGeometry geo;
  typedef QuadFace faceKind;
  static constexpr Int nfaceverts = 4;
  static constexpr Int vertex_degree = 4;
  static std::string filename() {return "quadRectSeed.dat";}
  static std::string id_string() {return "QuadRectSeed";}
  static std::string id() {return "qrect";}
  static std::string face_str() {return "quad";}
  static Index n_faces_at_tree_level(const Int lev);
  static Index n_vertices_at_tree_level(const Int lev);
  static Index n_edges_at_tree_level(const Index nv, const Index nf);
};

/** @brief Seed for planar meshes of triangular panels, free boundary conditions

   @image html triHexSeed.pdf "TriHexSeed"
*/
struct TriHexSeed {
  static constexpr Int nverts = 7;
  static constexpr Int nfaces = 6;
  static constexpr Int nedges = 12;
  typedef PlaneGeometry geo;
  typedef TriFace faceKind;
  static constexpr Int nfaceverts = 3;
  static constexpr Int vertex_degree = 6;
  static std::string filename() {return "triHexSeed.dat";}
  static std::string id_string() {return "TriHexSeed";}
  static std::string id() {return "trihex";}
  static std::string face_str() {return "tri";}
  static Index n_faces_at_tree_level(const Int lev);
  static Index n_vertices_at_tree_level(const Int lev);
  static Index n_edges_at_tree_level(const Index nv, const Index nf);
};

/** @brief Seed for planar meshes of the unit circle

  @image html unitDiskSeed.pdf "UnitDiskSeed"
*/
struct UnitDiskSeed {
  static constexpr Int nverts = 8;
  static constexpr Int nfaces = 5;
  static constexpr Int nedges = 12;
  typedef CircularPlaneGeometry geo;
  typedef QuadFace faceKind;
  static constexpr Int nfaceverts = 4;
  static constexpr Int vertex_degree = 4;
  static std::string filename() {return "unitDiskSeed.dat";}
  static std::string id_string() {return "UnitDiskSeed";}
  static std::string id() {return "disk";}
  static std::string face_str() {return "quad";}
  static Index n_faces_at_tree_level(const Int lev);
  static Index n_vertices_at_tree_level(const Int lev);
  static Index n_edges_at_tree_level(const Int nv, const Index nf);
};

/** @brief Seed for spherical quadrilateral meshes

   @image html cubedSphereSeed.pdf "CubedSphereSeed"
*/
struct CubedSphereSeed {
  static constexpr Int nverts = 8;
  static constexpr Int nfaces = 6;
  static constexpr Int nedges = 12;
  typedef SphereGeometry geo;
  typedef QuadFace faceKind;
  static constexpr Int nfaceverts = 4;
  static constexpr Int vertex_degree = 4;
  static std::string filename() {return "cubedSphereSeed.dat";}
  static std::string id_string() {return "CubedSphereSeed";}
  static std::string id() {return "cbsph";}
  static std::string face_str() {return "quad";}
  static Index n_faces_at_tree_level(const Int lev);
  static Index n_vertices_at_tree_level(const Int lev);
  static Index n_edges_at_tree_level(const Index nv, const Index nf);
};

/** @brief Seed for Icosahedral triangular meshes of the SphereGeometry

  @image html icosTriSphereSeed.pdf "IcosTriSphereSeed"
*/
struct IcosTriSphereSeed {
  static constexpr Int nverts = 12;
  static constexpr Int nfaces = 20;
  static constexpr Int nedges = 30;
  typedef SphereGeometry geo;
  typedef TriFace faceKind;
  static constexpr Int nfaceverts = 3;
  static constexpr Int vertex_degree = 6;
  static std::string filename() {return "icosTriSphereSeed.dat";}
  static std::string id_string() {return "IcosTriSphereSeed";}
  static std::string face_str() {return "tri";}
  static std::string id() {return "icosph";}
  static Index n_faces_at_tree_level(const Int lev);
  static Index n_vertices_at_tree_level(const Int lev);
  static Index n_edges_at_tree_level(const Index nv, const Index nf);
};

/** @brief A MeshSeed initializes a particle/panel mesh, and the Edges tree and Faces tree.

  The MeshSeed class is templated on the seed type (listed above).

  *** All of this class's methods execute on Host ***

  It provides run-time info based on the SeedType.

  MeshSeed is responsible for determining the memory required to construct a mesh, for use
  with Coords, Edges, and Faces constructors.
*/
template <typename SeedType> struct MeshSeed {
  /// Number of coordinates (both vertices and faces) in the seed file.
  static constexpr Int ncrds = SeedType::nverts + SeedType::nfaces;

  /// Host views to load data read from file
  ko::View<Real[ncrds][SeedType::geo::ndim],Host> seed_crds;
  ko::View<Index[SeedType::nedges][8],Host> seed_edges;
  ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> seed_face_verts;
  ko::View<Index[SeedType::nfaces][SeedType::nfaceverts],Host> seed_face_edges;
  ko::View<Index[SeedType::nverts][SeedType::vertex_degree],Host> seed_vert_edges;

  /// constructor.  Automatically reads data file.
  MeshSeed() : seed_crds("seed coords"), seed_edges("seed edges"), seed_face_verts("seed face vertices"),
    seed_face_edges("seed face edges"), seed_vert_edges("seed vertex edges") {read_file();}

  /// constructor. Multiplies radius of mesh.
  MeshSeed(const Real& maxr);

  /// id_string
  static std::string id_string() {return SeedType::id_string();}

  /// Concatenates directory info with SeedType::filename()
  std::string full_filename() const;

  /** Return the required memory allocations for a mesh tree of depth = lev
    nboundary = n vertices in a typical mesh (only different for high-order meshes, which are not implemented yet)
    nedges = n edges required by the mesh tree
    nfaces = n faces required by the mesh tree
  */
  void set_max_allocations(Index& nboundary, Index& nedges, Index& nfaces, const Int lev) const;

  /// Return runtime info about this object
  std::string info_string() const;

  /// Compute & return the area of the seed's initial faces.
  Real face_area(const Int ind) const;

  Real total_area() const;

  protected:
    /// Read data file.
    void read_file();

  private:

};



}
#endif
