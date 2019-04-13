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
    
        Edges(const Index nmax) : _origs("origs", nmax), _dests("dests", nmax), _lefts("lefts", nmax), _rights("rights", nmax), _parent("parent",nmax), _kids("kids", nmax), _n("n"), _nmax(nmax), _nActive("nactive") {
            _nh = ko::create_mirror_view(_n);
            _ho = ko::create_mirror_view(_origs);
            _hd = ko::create_mirror_view(_dests);
            _hl = ko::create_mirror_view(_lefts);
            _hr = ko::create_mirror_view(_rights);
            _hp = ko::create_mirror_view(_parent);
            _hk = ko::create_mirror_view(_kids);
            _hnActive = ko::create_mirror_view(_nActive);
            _nh(0) = 0;
            _hnActive(0) = 0;
        }
        
        /// Host function 
        inline Index nmax() const {return _nmax;}
        
        KOKKOS_INLINE_FUNCTION
        Index n() const {return _n(0);}
        
        /// Host function
        inline Index nh() const {return _nh(0);}
        
        void updateDevice() {
            ko::deep_copy(_origs, _ho);
            ko::deep_copy(_dests, _hd);
            ko::deep_copy(_lefts, _hl);
            ko::deep_copy(_rights, _hr);
            ko::deep_copy(_parent, _hp);
            ko::deep_copy(_kids, _hk);
            ko::deep_copy(_n, _nh);
            ko::deep_copy(_nActive, _hnActive);
        }
        
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
            
        KOKKOS_INLINE_FUNCTION Index getOrig(const Index ind) const {return _origs(ind);}
        KOKKOS_INLINE_FUNCTION Index getDest(const Index ind) const {return _dests(ind);}
        KOKKOS_INLINE_FUNCTION Index getLeft(const Index ind) const {return _lefts(ind);}
        KOKKOS_INLINE_FUNCTION Index getRight(const Index ind) const {return _rights(ind);}
        KOKKOS_INLINE_FUNCTION
        bool onBoundary(const Index ind) const {return _lefts(ind) == NULL_IND || 
            _rights(ind) == NULL_IND;}
        
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < _n(0) && _kids(ind, 0) >= 0;}
        
        
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
        inline bool hasKidsHost(const Index ind) const {return ind < _nh(0) && _hk(ind, 0) >= 0;}
    
    protected:
        edge_view_type _origs;
        edge_view_type _dests;
        edge_view_type _lefts;
        edge_view_type _rights;
        edge_view_type _parent;
        edge_tree_view _kids;
        ko::View<Index> _n;
        ko::View<Index> _nActive;
        
        edge_host_type _ho;
        edge_host_type _hd;
        edge_host_type _hl;
        edge_host_type _hr;
        edge_host_type _hp;
        edge_tree_host _hk; 
        ko::View<Index>::HostMirror _nh;    
        ko::View<Index>::HostMirror _hnActive;    
        Index _nmax;
};

}
#endif
