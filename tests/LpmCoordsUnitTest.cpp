#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <exception>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    Coords<PlaneGeometry> pc(20);
    std::cout << "pc.nMax() = " << pc.nMax() << ", pc.n() = " << pc.n() << std::endl;
    
    
    Coords<SphereGeometry> sc(20);
    std::cout << "sc.nMax() = " << sc.nMax() << ", sc.n() = " << sc.n() << std::endl;
    
    }
    Kokkos::finalize();
return 0;
}