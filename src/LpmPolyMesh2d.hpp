#ifndef LPM_POLYMESH_2D
#define LPM_POLYMESH_2D

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

template <typename FaceType, typename SeedType> class PolyMesh2d {
    public:
        typedef typename SeedType::geo geo;
        typedef ko::View<Real[geo::ndim], Dev> crd_view;
        typedef ko::View<Real[geo::ndim], Host> host_crd_view;
        typedef ko::View<const Real[geo::ndim],Dev> const_crd_view;
        typedef ko::View<const Real[geo::ndim],Host> host_const_crd_view;
        
        KOKKOS_INLINE_FUNCTION
        Index faceContainingPoint(const const_crd_view crd) const;
        
        /// Host function
        Index faceContainingPointHost(const host_const_crd_view crd) const;
    
    protected:
        Coords<geo> _vertcrds;
        Coords<geo> _vertlagcrds;
        Coords<geo> _facecrds;
        Coords<geo> _facelagcrds;
        
        Edges _edges;
        
        Faces<FaceType> _faces;
};

}
#endif
