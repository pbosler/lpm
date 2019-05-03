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

/**
    Modifications are only allowed on host.
    
    All device functions are read-only and const.
*/
class Edges {
    public:
        typedef ko::View<Index*> edge_view_type;
        typedef typename edge_view_type::HostMirror edge_host_type;
        typedef ko::View<Index*[2]> edge_tree_view;
        typedef typename edge_tree_view::HostMirror edge_tree_host;
        
        edge_view_type origs;
        edge_view_type dests;
        edge_view_type lefts;
        edge_view_type rights;
        edge_view_type parent;
        edge_tree_view kids;
        n_view_type n;
        n_view_type nLeaves;
    
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

        KOKKOS_INLINE_FUNCTION
        bool onBoundary(const Index ind) const {return lefts(ind) == NULL_IND || 
            rights(ind) == NULL_IND;}
        
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < n() && kids(ind, 0) > 0;}
        
/*/////  HOST FUNCTIONS ONLY BELOW THIS LINE         
    
        todo: make them protected, not public    
        
*/           
        /// Host function 
        inline Index nmax() const {return _nmax;}
        
        /// Host function
        inline Index nh() const {return _nh();}
        
        /// Host function
        void insertHost(const Index o, const Index d, const Index l, const Index r, const Index prt=NULL_IND);
        
        /// Host function
        template <typename Geo>
        void divide(const Index ind, Coords<Geo>& crds, Coords<Geo>& lagcrds);
        
        /// Host function
        inline void setLeft(const Index ind, const Index newleft) {
            _hl(ind) = newleft;
        }
    
        /// Host function
        inline void setRight(const Index ind, const Index newright) {
            _hr(ind) = newright;
        }
        
        /// Host function
        template <typename SeedType>
        void initFromSeed(const MeshSeed<SeedType>& seed);
        
        /// Host function
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
