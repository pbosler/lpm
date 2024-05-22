#ifndef DFS_RHS_HPP
#define DFS_RHS_HPP

#include "dfs_config.hpp"
#include "dfs_fft.hpp"

namespace SpherePoisson{
template<typename srcType, typename ViewType>
class DFS_RHS{
    private:
        Int nrows;
        Int ncols;
        ViewType sdrhs;
        ViewType rhse;
        ViewType rhso;
        view_1d<Int> ie;
        view_1d<Int> io;
        DFS_FFTProcessor<srcType, ViewTYpe> FFTP;

        // private methods
        void indices_split();
        void splitrhs();
        void scalerhs();
        
    public:
        // constructor
        DFS_RHS(const Int nrow_, const Int ncols_);
        // Default destructor
        ~DFS_RHS(){}

        // public methods
        void poisson_rhs(srcType f);
        view_1d<Int> get_io();
        view_1d<Int> get_ie();
        ViewType get_rhse();
        ViewType get_rhso();

        // public member
         ViewType drhs;
         
};

}
#endif
