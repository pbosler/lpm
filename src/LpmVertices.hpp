#ifndef LPM_VERTICES_HPP
#define LPM_VERTICES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"

#include "LpmMeshSeed.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

template <typename SeedType> class Vertices {
    public:
        ko::View<Index*[SeedType::vertex_degree]> vertexEdges;
        n_view_type n;

        Vertices(const Index& nmax) : vertexEdges("vertex_edges", nmax), _nmax(nmax) {
            _hostVertEdges = ko::create_mirror_view(vertexEdges);
            _nh = ko::create_mirror_view(n);
            _nh() = 0;
        }

    protected:
        typename ko::View<Index*[SeedType::vertex_degree]>::HostMirror _hostVertEdges;
        typename n_view_type::HostMirror _nh;
        Index _nmax;

};

}
#endif
