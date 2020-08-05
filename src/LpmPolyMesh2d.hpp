#ifndef LPM_POLYMESH_2D_HPP
#define LPM_POLYMESH_2D_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmVtkIO.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

enum FieldKind {VertexField, EdgeField, FaceField};

#ifdef LPM_HAVE_NETCDF
  class PolyMeshReader; /// fwd. decl.
#endif

/** @brief Class for organizing a topologically 2D mesh of particles and panels

  Provides a single access point for a collection of:
  1. Vertices, represented by physical and Lagrangian Coords
  2. Edges
  3. Faces, and a coincident set of physical and Lagrangian Coords to represent face centers.
*/
template <typename SeedType> class PolyMesh2d {
  public:
  typedef typename SeedType::geo Geo;
  typedef typename SeedType::faceKind FaceType;

  /** @brief Constructor.  Allocates memory for a PolyMesh2d instance.

    @param nmaxverts Maximum number of vertices that will be allowed in memory
    @param nmaxedges Maximum number of edges allowed in memory
    @param nmaxfaces Maximum number of faces allowed in memory

    @see MeshSeed::setMaxAllocations()
  */
  PolyMesh2d(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces) :
    physVerts(nmaxverts), lagVerts(nmaxverts), edges(nmaxedges), faces(nmaxfaces),
    physFaces(nmaxfaces), lagFaces(nmaxfaces) {}

#ifdef LPM_HAVE_NETCDF
  PolyMesh2d(const PolyMeshReader& reader);
#endif

    /// Destructor
    virtual ~PolyMesh2d() {}

    /** @brief initial refinement level.  All panels in the MeshSeed will be refined to this level.

      Adaptive refinement adds to this level.
    */
    Int baseTreeDepth;

    /** @brief Return a subview of all initialized vertices' physical coordinates

      @device
    */
    typename Coords<Geo>::crd_view_type getVertCrds() const {
      return typename Coords<Geo>::crd_view_type(physVerts.crds, std::make_pair(0,physVerts.nh()), ko::ALL());}


    /** @brief Return a subview of all initialized face particles' physical coordinates

    @device
    */
    typename Coords<Geo>::crd_view_type getFaceCrds() const {
      return typename Coords<Geo>::crd_view_type(physFaces.crds, std::make_pair(0,faces.nh()), ko::ALL());}

    /** @brief Return a view face masks; leaves of the face tree are not masked. Internal nodes are masked.

      @device
    */
    mask_view_type getFacemask() const {
      return mask_view_type(faces.mask, std::make_pair(0,faces.nh()));}


    /** @brief Return a view of face masks on host.

     @todo This function seems useless without a deep copy

    @hostfn
    */
    typename mask_view_type::HostMirror getFacemaskHost() const {return ko::create_mirror_view(faces.mask);}


    /** @brief Return a subview of all initialized face areas

      @device
    */
    scalar_view_type getFaceArea() const {
      return scalar_view_type(faces.area, std::make_pair(0,faces.nh()));}


    /** @brief Number of initialized vertices

    @device
    */
    KOKKOS_INLINE_FUNCTION
    Index nverts() const {return physVerts.n();}

    /** @brief Number of initialized faces

    @device
    */
    KOKKOS_INLINE_FUNCTION
    Index nfaces() const {return faces.n();}


    /** @brief Number of initialized vertices

    @hostfn
    */
    Index nvertsHost() const {return physVerts.nh();}

    /** @brief Number of initialized edges

    @hostfn
    */
    Index nedgesHost() const {return edges.nh();}


    /** @brief Number of initialized faces

    @hostfn
    */
    Index nfacesHost() const {return faces.nh();}

    /// Physical coordinates of panel vertices
    Coords<Geo> physVerts;
    /// Lagrangian coordinates of panel vertices
    Coords<Geo> lagVerts;

    /// Panel Edges
    Edges edges;


    /// Panels (Faces)
    Faces<FaceType> faces;

    /// Physical coordinates of particles at face centers
    Coords<Geo> physFaces;

    /// Lagrangian coordinates of particles at face centers
    Coords<Geo> lagFaces;


    /** @brief Returns a pointer to the view of face coordinates

    @hostfn

    */
    typename Coords<Geo>::crd_view_type::HostMirror getFaceCrdsHost() {return physFaces.getHostCrdView();}


    /** @brief Starting with a MeshSeed, uniformly refines each panel until the desired level of initial refinement is reached.

    @hostfn

    @param initDepth Max depth of initially refined mesh
    @param seed Mesh seed used to initialize particles and panels
    */
    void treeInit(const Int initDepth, const MeshSeed<SeedType>& seed);


    /// @brief Construct relevant Vtk objects for visualization of a PolyMesh2d instance
    virtual void outputVtk(const std::string& fname) const;

    /// @brief Copies data from host to device
    virtual void updateDevice() const;

    /// @brief Copies data from device to host
    virtual void updateHost() const;


    /** @brief Writes basic info about a PolyMesh2d instance to a string.

    @hostfn
    */
    std::string infoString(const std::string& label="", const int& tab_level = 0, const bool& dump_all=false) const;

  protected:
    typedef FaceDivider<Geo,FaceType> divider;

    void seedInit(const MeshSeed<SeedType>& seed);


};


/** @brief Collects physical coordinates from all particles (vertices and faces)
  for use as a source of interpolation data.

  @todo rewrite to handle faces in parallel

  @param pm PolyMesh2d mesh used as data source
*/
template <typename SeedType>
ko::View<Real*[3]> sourceCoords(const PolyMesh2d<SeedType>& pm) {
  const Index nv = pm.nvertsHost();
  const Index nl = pm.faces.nLeavesHost();
  ko::View<Real*[3]> result("source_coords", nv + nl);
  ko::parallel_for(nv, KOKKOS_LAMBDA (int i) {
    for (int j=0; j<3; ++j) {
      result(i,j) = pm.physVerts.crds(i,j);
    }
  });
  ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
    Int offset = nv;
    for (int j=0; j<pm.nfaces(); ++j) {
      if (!pm.faces.mask(j)) {
        result(offset,0) = pm.physFaces.crds(j,0);
        result(offset,1) = pm.physFaces.crds(j,1);
        result(offset++,2) = pm.physFaces.crds(j,2);
      }
    }
  });
  return result;
}

}
#endif
