#ifndef LPM_EDGES_HPP
#define LPM_EDGES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"

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
    
        Edges(const Index nmax) : _origs("origs", nmax), _dests("dests", nmax), _lefts("lefts", nmax), _rights("rights", nmax), _parent("parent",nmax), _kids("kids", nmax), _n("n"), _nmax(nmax) {
            _nh = ko::create_mirror_view(_n);
            _ho = ko::create_mirror_view(_origs);
            _hd = ko::create_mirror_view(_dests);
            _hl = ko::create_mirror_view(_lefts);
            _hr = ko::create_mirror_view(_rights);
            _hp = ko::create_mirror_view(_parent);
            _hk = ko::create_mirror_view(_kids);
            _nh(0) = 0;
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
        }
        
        /// Host function
        void insertHost(const Index o, const Index d, const Index l, const Index r, const Index prt=NULL_IND) {
            const Index ins_pt = _nh(0);
            _ho(ins_pt) = o;
            _hd(ins_pt) = d;
            _hl(ins_pt) = l;
            _hr(ins_pt) = r;
            _hp(ins_pt) = prt;
            _hk(ins_pt, 0) = NULL_IND;
            _hk(ins_pt, 1) = NULL_IND;
            _nh(0) += 1;
        }
        
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
            
        KOKKOS_INLINE_FUNCTION Index getOrig(const Index ind) const {return _ho(ind);}
        KOKKOS_INLINE_FUNCTION Index getDest(const Index ind) const {return _hd(ind);}
        KOKKOS_INLINE_FUNCTION Index getLeft(const Index ind) const {return _hl(ind);}
        KOKKOS_INLINE_FUNCTION Index getRight(const Index ind) const {return _hr(ind);}
        
        template <typename V>
        inline void getKidsHost(V& v, const Index ind) const {
            v[0] = _hk(ind,0);
            v[1] = _hk(ind,1);
        }
        
        /// Host function
        void printedges(const std::string& label) const;
        
        KOKKOS_INLINE_FUNCTION
        bool onBoundary(const Index ind) const {return _lefts(ind) == NULL_IND || _rights(ind) == NULL_IND;}
        
        KOKKOS_INLINE_FUNCTION
        bool hasKids(const Index ind) const {return ind < _n(0) && _kids(ind, 0) >= 0;}
        
        inline bool onBoundaryHost(const Index ind) const {return _hl(ind) == NULL_IND || _hr(ind) == NULL_IND;}
        inline bool hasKidsHost(const Index ind) const {return ind < _nh(0) && _hk(ind, 0) >= 0;}
    
    protected:
        edge_view_type _origs;
        edge_view_type _dests;
        edge_view_type _lefts;
        edge_view_type _rights;
        edge_view_type _parent;
        edge_tree_view _kids;
        ko::View<Index> _n;
        
        edge_host_type _ho;
        edge_host_type _hd;
        edge_host_type _hl;
        edge_host_type _hr;
        edge_host_type _hp;
        edge_tree_host _hk; 
        ko::View<Index>::HostMirror _nh;        
        Index _nmax;
};

}
#endif
