#ifndef LPM_KOKKOS_UTIL
#define LPM_KOKKOS_UTIL

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"

/**
Kokkos-related utilities
*/
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
