#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include <cassert>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

#ifdef LPM_USE_NETCDF
template <typename Geo>
class NcWriter;
template <typename Geo>
class UnstructuredNcReader;
class PolymeshReader;
#endif

#ifdef LPM_USE_VTK
template <typename SeedType>
class VtkPolymeshInterface;
#endif

/** \brief Coords class handles arrays of vectors in \f$\mathbb{R}^d\f$, where
\f$d=2,3\f$. Templated on Geometry Type (e.g., SphereGeometry, PlaneGeometry).

  All initialization is done on host.

  When used with adaptive refinement, this class will allocate more memory than
initialization requires, to save room.

*/
template <typename Geo>
class Coords {
 public:
  typedef Geo crds_geometry_type;
  typedef typename Geo::crd_view_type view_type;
  typedef typename Geo::crd_view_type
      crd_view_type;  ///< basic array type defined from Geometry type
  crd_view_type
      view;  ///< primary container --- a view of vectors in spatial coordinates
  n_view_type n;  ///< number of vectors currently intialized

#ifdef LPM_USE_VTK
  template <typename SeedType>
  friend class VtkPolymeshInterface;
#endif

#ifdef LPM_USE_NETCDF
  friend class NcWriter<Geo>;
  friend class UnstructuredNcReader<Geo>;
  friend class PolymeshReader;
#endif

  Coords() = default;

  /** @brief Constructor.

    @param nmax Maximum number of faces to allocate space.

    @see MeshSeed::setMaxAllocations()
  */
  explicit Coords(const Index nmax);

  /** @brief Constructor.

    Builds a new Coords wrapper around an existing view.

    @param cv [in] Existing view of coordinate vectors
  */
  explicit Coords(const ko::View<Real**> cv);

  /** @brief Constructor.

    Builds a new Coords object with its own views, using deep copy.
  */
  Coords(const Coords<Geo>& other);

  /**
    Copy data from host to device.
  */
  void update_device() const {
    ko::deep_copy(view, _hostview);
    ko::deep_copy(n, _nh);
  }

  /** Copy data from device to host.
   */
  void update_host() const {
    ko::deep_copy(_hostview, view);
    ko::deep_copy(_nh, n);
  }

  Real max_radius() const;

  /** \brief Number of coordinates initialized.

    nh() <= nMax()

    \hostfn
  */
  Index nh() const { return _nh(); }

  /// \brief maximum number of coordinate vectors allowed in memory
  KOKKOS_INLINE_FUNCTION
  Index n_max() const { return view.extent(0); }

  /** \brief Get specfied component (e.g., 0, 1, 2) from a particular coordinate
  vector

  \hostfn

  \param ind index of coordinate vector
  \param dim component of vector
  \return crds(ind,dim)
  */
  inline Real get_crd_component_host(const Index ind, const Int dim) const {
    LPM_ASSERT(ind < _nh());
    return _hostview(ind, dim);
  }

  /** \brief Inserts a new coordinate to the main data container

  \hostfn

  @param v position vector to add
  */
  template <typename CV>
  void insert_host(const CV v) {
    LPM_REQUIRE_MSG(_nmax >= _nh() + 1,
                    "Coords::insert error: not enough memory.");
    for (int i = 0; i < Geo::ndim; ++i) {
      _hostview(_nh(), i) = v[i];
    }
    ++_nh();
  }

  /** \brief overwrites a coordinate vector with new data

  \todo consider renaming to update or overwrite

  \hostfn

  \param ind index of vector to be overwritten
  \param v data to write
  */
  template <typename CV>
  void set_crds_host(const Index ind, const CV v) {
    LPM_ASSERT(ind < _nh());
    for (Short i = 0; i < Geo::ndim; ++i) {
      _hostview(ind, i) = v[i];
    }
  }

  /** \brief Writes basic info about a Coords instance to a string.

  \hostfn

  \param label name of instance
  */
  std::string info_string(const std::string& label="", const short& tab_level = 0,
                          const bool& dump_all = false) const;

  /** \brief Initializes a random set of coordinates.

  \hostfn
  */
  void init_random(const Real max_range = 1.0, const Int ss = 0);

  /**  \brief Initializes a coordinate on a panel edge (i.e., a vertex or a
  edge-interior point (high-order methods only)).

  \hostfn

  \param seed MeshSeed used for particle/panel initializaiton
  */
  template <typename SeedType>
  void init_vert_crds_from_seed(const MeshSeed<SeedType>& seed);

  /** \brief Initializes a coordinate on a panel interior

    Typically only used by high-order elements.

    \hostfn
  */
  template <typename SeedType>
  void init_interior_crds_from_seed(const MeshSeed<SeedType>& seed);

  /** \brief Output all data to a stream, writing it in matlab format.

  \hostfn

  \param os stream to write data (typically, an open .m file)
  \param name name for coordinate array
  */
  void write_matlab(std::ostream& os, const std::string& name) const;

  /** \brief return a reference to the primary data container's host mirror

  \hostfn
  */
  typename crd_view_type::HostMirror get_host_crd_view() { return _hostview; }

  typename crd_view_type::HostMirror get_const_host_crd_view() const {return _hostview; }

  Kokkos::MinMaxScalar<Real> min_max_extent(const int dim) const;

 protected:
  typename crd_view_type::HostMirror _hostview;  ///< host view of primary data
  Index _nmax;  ///< maximum number of coordinates allowed in memory
  typename n_view_type::HostMirror
      _nh;  ///< number of currently initialized coordinates
};

}  // namespace Lpm
#endif
