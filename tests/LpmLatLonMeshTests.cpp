#include "LpmConfig.h"
#include "LpmLatLonMesh.hpp"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include "LpmRossbyWaves.hpp"
#include "LpmMatlabIO.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    const std::vector<Int> nlats = {91, 181, 361, 721};
    const std::vector<Int> nlons = {180, 360, 720, 1440};
    for (int i=0; i<nlats.size(); ++i) {
        const Int nlat = nlats[i];
    	const Int nlon = nlons[i];
    
        LatLonMesh ll(nlat, nlon);
        const Index npts = nlat*nlon;
    
        Real surf_area;
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& a) {
            a += ll.wts(i);
        }, surf_area);
        const Real area_err = std::abs(surf_area - 4*PI);
        std::cout << "surf_area = " << std::setprecision(20) << surf_area << " abs(area_err) = " <<
            std::setprecision(20) << area_err << "\n";

        if (i==0) {
            std::ofstream mfile("latlon_test.m");
            ll.writeLatLonMeshgrid(mfile);
            mfile.close();
        }
        
        ko::View<Real*> onesA("ones", npts);
        ko::View<Real*> onesB("ones", npts);
        ko::View<Real*> onesError("error", npts);
        ko::parallel_for(npts, KOKKOS_LAMBDA (const Int& i) {
            onesA(i) = 1.0;
            onesB(i) = 1.001;
        });
        
        ll.computeScalarError(onesError, onesA, onesB);
        ErrNorms errs = ll.scalarErrorNorms(onesError, onesA);
        std::cout << errs.infoString("approx. 1 :: ");
    }
}
ko::finalize();
return 0;
}
