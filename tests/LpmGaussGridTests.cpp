#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "SPGaussGrid.hpp"
#include <iostream>
#include "Kokkos_Core.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    GaussGrid<> gg1(1);
    std::cout << gg1.infoString();
    GaussGrid<> gg2(2);
    std::cout << gg2.infoString();
    GaussGrid<> gg3(3);
    std::cout << gg3.infoString();
    GaussGrid<> gg4(4);
    std::cout << gg4.infoString();
}
ko::finalize();
return 0;
}
