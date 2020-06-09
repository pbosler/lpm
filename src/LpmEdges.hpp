#ifndef LPM_EDGES_HPP
#define LPM_EDGES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmMeshSeed.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/** @brief Edges of panels connect Coords to Faces

  Each edge has an origin vertex and a destination vertex (in Coords), a left face and a right face (in Faces).

  This class stores arrays that define a collection of edges.

  @warning  Modifications are only allowed on host.
    All device functions are read-only and const.
*/
class Edges {
    public:
        typedef ko::View<Index*> edge_view_type; ///< type to hold pointers (array indices) to other mesh objects
        typedef typename edge_view_type::HostMirror edge_host_type;
        typedef ko::View<Index*[2]> edge_tree_view; ///< Edge division results in a binary tree; this holds its data
        typedef typename edge_tree_view::HostMirror edge_tree_host;

        edge_view_type origs; ///< pointers to edge origin vertices
        edge_view_type dests; ///< pointers to edge destination vertices
        edge_view_type lefts; ///< pointers to edges' left faces
        edge_view_type rights; ///< pointers to edges' right faces
        edge_view_type parent; ///< pointers to parent edges
        edge_tree_view kids; ///< pointers to child edges
        n_view_type n; ///< number of initialized edges
        n_view_type nLeaves; ///< number of leaf edges



        /** @brief Constructor.

          @param nmax Maximum number of edges to allocate space.
          @see MeshSeed::setMaxAllocations()
        */
        Edges(const Index nmax) : origs("origs", nmax), dests("dests", nmax), lefts("lefts", nmax), rights("rights", nmax), parent("parent",nmax), kids("kids", nmax), n("n"), _nmax(nmax), nLeaves("nLeaves") {
            _nh = ko::create_mirror_view(n);
            _ho = ko::create_mirror_view(origs);
            _hd = ko::create_mirror_view(dests);
            _hl = ko::create_mirror_view(lefts);
            _hr = ko::create_mirror_view(rights);
            _hp = ko::create_mirror_view(parent);
            _hk = ko::create_mirror_view(kids);
            _hnLeaves = ko::create_mirror_view(nLeaves);
            _nh() = 0;
            _hnLeaves() = 0;
        }


        /** \brief Copies edge data from host to device.
        */
        void updateDevice() const {
            ko::deep_copy(origs, _ho);
            ko::deep_copy(dests, _hd);
            ko::deep_copy(lefts, _hl);
            ko::deep_copy(rights, _hr);
            ko::deep_copy(parent, _hp);
            ko::deep_copy(kids, _hk);
            ko::deep_copy(n, _nh);
            ko::deep_copy(nLeaves, _hnLeaves);
        }

        /** \brief Returns true if the edge is on the boundary of the domain

        */
        KOKKOS_INLINE_FUNCTION
        bool onBoundary(const Index ind) const {return lefts(ind) == NULL_IND ||
            rights(ind) == NULL_IND;}


        /** \brief Returns true if the edge has been divided.

        */
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < n() && kids(ind, 0) > 0;}

        /** \brief Maximum number of edges allowed in memory
        */
        KOKKOS_INLINE_FUNCTION
        inline Index nmax() const {return origs.extent(0);}

        /** Number of initialized edges.

        \hostfn
        */
        inline Index nh() const {return _nh();}

        /** Inserts a new edge into the data structure.

        \hostfn

        \param o index of origin vertex for new edge
        \param d index of destination vertex for new edge
        \param l index of left panel for new edge
        \param r index of right panel for new edge
        \param prt index of parent edge
        */
        void insertHost(const Index o, const Index d, const Index l, const Index r, const Index prt=NULL_IND);

        /** \brief Divides an edge, creating two children

        \hostfn


        Child edges have same left/right panels as parent edge.

        The first child, whose index will be n is the 0th index child of the parent, and shares its origin vertex.
        A new midpoint is added to both sets of Coords.
        The second child has index n+1 is the 1th index child of the parent, and shares its destination vertex.

        @todo assert(n==child(ind,0))
        @todo assert(n+1==child(ind,1))
        @todo assert(orig(n) == orig(ind))
        @todo assert(dest(n+1) == dest(ind))

        \param ind index of edge to be divided
        \param crds physical coordinates of edge vertices
        \param lagcrds Lagrangian coordinates of edge vertices
        */
        template <typename Geo>
        void divide(const Index ind, Coords<Geo>& crds, Coords<Geo>& lagcrds);

        /** \brief Overwrite the left panel of an edge

          \hostfn

          \param ind edge whose face needs updating
          \param newleft new index of the edge's left panel
        */
        inline void setLeft(const Index ind, const Index newleft) {
            _hl(ind) = newleft;
        }

        /** \brief Overwrite the right panel of an edge

          \hostfn

          \param ind edge whose face needs updating
          \param newright new index of the edge's right panel
        */
        inline void setRight(const Index ind, const Index newright) {
            _hr(ind) = newright;
        }

        /** Initialize a set of Edges from a MeshSeed

        \hostfn

        \param seed MeshSeed
        */
        template <typename SeedType>
        void initFromSeed(const MeshSeed<SeedType>& seed);

        /** Return the requested child (0 or 1) of an edge.

        \hostfn

        \param ind index of edge whose child is needed
        \param child either 0 or 1, to represent the first or second child
        \see divide()
        */
        inline Index getEdgeKidHost(const Index ind, const Int child) const {return _hk(ind, child);}

        /// Host function
        std::string infoString(const std::string& label) const;

        /// Host functions
        inline Index getOrigHost(const Index ind) const {return _ho(ind);}
        inline Index getDestHost(const Index ind) const {return _hd(ind);}
        inline Index getLeftHost(const Index ind) const {return _hl(ind);}
        inline Index getRightHost(const Index ind) const {return _hr(ind);}

        /// Host function
        inline bool onBoundaryHost(const Index ind) const {return _hl(ind) == NULL_IND || _hr(ind) == NULL_IND;}

        /// Host function
        inline bool hasKidsHost(const Index ind) const {return ind < _nh() && _hk(ind, 0) > 0;}

    protected:
        edge_host_type _ho;
        edge_host_type _hd;
        edge_host_type _hl;
        edge_host_type _hr;
        edge_host_type _hp;
        edge_tree_host _hk;
        ko::View<Index>::HostMirror _nh;
        ko::View<Index>::HostMirror _hnLeaves;
        Index _nmax;
};

}
#endif
