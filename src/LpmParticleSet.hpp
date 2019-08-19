#ifndef LPM_PARTICLE_SET_HPP
#define LPM_PARTICLE_SET_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmMeshSeed.hpp"

template <typename Geo> class Particles {
    public:
        typedef typename Geo::crd_view_type crd_view;
        typedef typename scalar_view_type scalar_field;
        typedef typename crd_view vector_field;
        typedef typename vector_field::HostMirror host_vector;
        typedef typename scalar_field::HostMirror host_scalar;
        typedef typename crd_view::HostMirror host_crd;
        crd_view phys_crds;
        crd_view lag_crds;
        scalar_view_type weights;
        quad_tree_view kids;
        index_view_type parents;
        ko::View<Int*> level;
        mask_view_type mask;
        n_view_type nLeaves;
        
        template <typename SeedType> 
        Particles(const MeshSeed<SeedType>& seed, const int tree_depth);
        
        virtual ~ParticleSet() {}
        
        virtual void updateDevice();
        virtual void updateHost();
    
    protected:
        host_crd _phys_crds;
        host_crd _lag_crds;
        host_scalar _weights;
};

#endif