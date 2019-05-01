#ifndef LPM_KOKKOS_UTIL
#define LPM_KOKKOS_UTIL

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"

/**
Kokkos-related utilities
*/
namespace Kokkos {

/** Tuple type for || reduction.

    T is a plain old data type
    ndim is the number of T's in the tuple.
    
*/
template <typename T, int ndim> struct Tuple {
    T data[ndim];
    KOKKOS_FORCEINLINE_FUNCTION Tuple() {
        for (int i=0; i<ndim; ++i) 
            data[i] = 0;
    }
    KOKKOS_FORCEINLINE_FUNCTION Tuple(const T n) {
        for (int i=0; i<ndim; ++i) 
            data[i] = n;
    }
    KOKKOS_FORCEINLINE_FUNCTION Tuple operator += (const Tuple<T,ndim>& o) const {
        Tuple<T,ndim> result;
        for (int i=0; i<ndim; ++i) {
            result.data[i] = data[i] + o.data[i];
        }
        return result;
    }
    KOKKOS_FORCEINLINE_FUNCTION Tuple operator *= (const Tuple<T,ndim>& o) const {
        Tuple<T,ndim> result;
        for (int i=0; i<ndim; ++i) {
            result.data[i] = data[i] * o.data[i];
        }
        return result;
    }
    KOKKOS_FORCEINLINE_FUNCTION T& operator [] (const int i) {return data[i];}
    KOKKOS_FORCEINLINE_FUNCTION const T& operator [] (const int i) const {return data[i];}
};

template <> 
struct reduction_identity<Tuple<Lpm::Real,3>> {
    KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,3> sum() {return Tuple<Lpm::Real,3>();}
    KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,3> prod() {return Tuple<Lpm::Real,3>(1);}
};

}

namespace Lpm {
/**
    ExeSpaceUtils is a TeamPolicy factory.  Defines thread layout : number of teams, threads per team.
    
    CPU: 1 thread per team.
*/
template <typename ExeSpace=ko::DefaultExecutionSpace>
struct ExeSpaceUtils {
    using TeamPolicy = ko::TeamPolicy<ExeSpace>;
    
    static TeamPolicy get_default_team_policy(Int ni, Int nk) {
#ifdef MIMIC_GPU
        const int max_threads = ExeSpace::concurrency();
        const int team_size = max_threads < 7 ? max_threads : 7;
        return TeamPolicy(ni, team_size);
#else
        return TeamPolicy(ni, 1);
#endif
    }
};

/**
    Specialized policy for Cuda.
    
    GPU: <= 128 threads per team.
*/
#ifdef HAVE_CUDA
template <>
struct ExeSpaceUtils<ko::Cuda> {
    using TeamPolicy = ko::TeamPolicy<ko::Cuda>;
    
    static TeamPolicy get_default_team_policy(Int ni, Int nk) {
        return TeamPolicy(ni, std::min(128, 32*((nk+31)/32)));
    }
};
#endif

/**
    TeamUtils provide concurrency info for thread teams
*/
// template <typename ExeSpace=ko::DefaultExecutionSpace>
// class TeamUtils {
//     protected:
//         int _team_size;
//         int _num_teams;
//     
//     public:
//         template <typename TeamPolicy>
//         TeamUtils(const TeamPolicy& policy) : _team_size(0) {
//             const int max_threads = ExeSpace::concurrency();
//             const int team_size = policy.team_size;
//         }
// };

}
#endif
