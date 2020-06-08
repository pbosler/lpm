#ifndef LPM_COORDS_HPP
#define LPM_COORDS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/** \brief Coords class handles arrays of vectors in \f$\mathbb{R}^d\f$, where \f$d=2,3\f$.
Templated on Geometry Type (e.g., SphereGeometry, PlaneGeometry).

  All initialization is done on host.

  When used with adaptive refinement, this class will allocate more memory than initialization requires, to save room.

*/
template <typename Geo> class Coords {
    public:
        typedef typename Geo::crd_view_type crd_view_type; ///< basic array type defined from Geometry type
        crd_view_type crds; ///< primary container --- a view of vectors
        n_view_type n; ///< number of vectors currently intialized

        Coords(const Index nmax) : crds("crds", nmax), _nmax(nmax), n("n") {
            _hostcrds = ko::create_mirror_view(crds);
            _nh = ko::create_mirror_view(n);
            _nh() = 0;
        };

        /**
          Copy data from host to device.
        */
        void updateDevice() const {
            ko::deep_copy(crds, _hostcrds);
            ko::deep_copy(n, _nh);
        }

        /** Copy data from device to host.
        */
        void updateHost() const {
            ko::deep_copy(_hostcrds, crds);
            ko::deep_copy(_nh, n);
        }


        /** \brief Number of coordinates initialized.

          nh() <= nMax()

          \hostfn
        */
        Index nh() const {return _nh();}

        /// \brief maximum number of coordinate vectors allowed in memory
        KOKKOS_INLINE_FUNCTION
        Index nMax() const { return crds.extent(0);}

        /** \brief Get specfied component (e.g., 0, 1, 2) from a particular coordinate vector

        \hostfn

        \param ind index of coordinate vector
        \param dim component of vector
        \return crds(ind,dim)
        */
        inline Real getCrdComponentHost(const Index ind, const Int dim) const {return _hostcrds(ind, dim);}

        /** \brief Inserts a new coordinate to the main data container

        \hostfn

        @param v position vector to add
        */
        template <typename CV>
        void insertHost(const CV v) {
            LPM_THROW_IF(_nmax < _nh() + 1, "Coords::insert error: not enough memory.");
            for (int i=0; i<Geo::ndim; ++i) {
                _hostcrds(_nh(), i) = v[i];
            }
            _nh() += 1;
        }

        /** \brief overwrites a coordinate vector with new data

        \todo consider renaming to update or overwrite

        \hostfn

        \param ind index of vector to be overwritten
        \param v data to write
        */
        void relocateHost(const Index ind, const ko::View<Real[Geo::ndim], Host> v) {
            LPM_THROW_IF(ind >= _nh(), "Coords::relocateHost error: index out of range.");
            for (int i=0; i<Geo::ndim; ++i) {
                _hostcrds(ind, i) = v(i);
            }
        }

        /** \brief Writes basic info about a Coords instance to a string.

        \hostfn

        \param label name of instance
        */
        std::string infoString(const std::string& label) const;

        /** \brief Initializes a random set of coordinates.

        \hostfn
        */
        void initRandom(const Real max_range=1.0, const Int ss=0);

        /**  \brief Initializes a coordinate on a panel boundary (i.e., a vertex).

        \todo consider renaming to avoid conflict with domain boundary

        \hostfn

        \param seed MeshSeed used for particle/panel initializaiton
        */
        template <typename SeedType>
        void initBoundaryCrdsFromSeed(const MeshSeed<SeedType>& seed);

        /** \brief Initializes a coordinate on a panel interior

          Typically only used by high-order elements.

          \hostfn
        */
        template <typename SeedType>
        void initInteriorCrdsFromSeed(const MeshSeed<SeedType>& seed);

        /** \brief Output all data to a stream, writing it in matlab format.

        \hostfn

        \param os stream to write data (typically, an open .m file)
        \param name name for coordinate array
        */
        void writeMatlab(std::ostream& os, const std::string& name) const;

        /** \brief return a reference to the primary data container's host mirror

        \hostfn
        */
        typename crd_view_type::HostMirror getHostCrdView() {return _hostcrds;}
    protected:
        typename crd_view_type::HostMirror _hostcrds; ///< host view of primary data
        Index _nmax; ///< maximum number of coordinates allowed in memory
        typename n_view_type::HostMirror _nh; ///< number of currently initialized coordinates

};


}
#endif
