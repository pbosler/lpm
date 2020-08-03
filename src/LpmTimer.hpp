#ifndef LPM_TIMER_HPP
#define LPM_TIMER_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include <mpi.h>
#include <string>

namespace Lpm {

class Timer {

  public:
    Timer(const std::string& name="") : _name(name), _start(0), _end(0) {}

    inline void start() {_start = MPI_Wtime();}

    inline void stop() {_end = MPI_Wtime(); _elap = _end - _start;}

    inline Real elapsed() const {return _elap;}

    std::string infoString() const;

  protected:
    Real _start;
    Real _end;
    Real _elap;
    std::string _name;

};

}
#endif
