#include "LpmConfig.h"
#include "LpmLatLonMesh.hpp"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include "LpmRossbyWaves.hpp"
#include "LpmMatlabIO.hpp"
#include "LpmErrorNorms.hpp"
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
        std::cout << "Mesh ready.\n";
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
        
        std::cout << "ones arrays ready.\n";
        ll.computeScalarError(onesError, onesA, onesB);
        std::cout << "errors ready.\n";
        
//        ko::View<Real[6],Dev> calc_err_view("error_intermediate");
//         ko::View<ko::Tuple<Real,6>> tup_view("tuple_view");
//         ko::Tuple<Real,6> err_val;
//         ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, ko::Tuple<Real,6>& er) {
//             er[0] += abs(onesError(i))*ll.wts(i);
//             er[1] += abs(onesA(i))*ll.wts(i);
//             er[2] += square(onesError(i))*ll.wts(i);
//             er[3] += square(onesA(i))*ll.wts(i);
//             er[4] = max(abs(onesError(i)), er[4]);
//             er[5] = max(abs(onesA(i)), er[5]);
//         }, ErrReducer<Dev>(tup_view));
//         auto host_err = ko::create_mirror_view(calc_err_view);
//         auto host_err = ko::create_mirror_view(tup_view);
//         ko::deep_copy(host_err, calc_err_view);
//         ko::deep_copy(host_err, tup_view);
//         ErrNorms<> errs(*host_err.data());
        ErrNorms<> errs(onesError, onesA, ll.wts);
        
        /**
        Real l1, l1denom;
        Real l2, l2denom;
        Real linf, linfdenom;
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& er) {
            er += abs(onesError(i))*ll.wts(i);
        }, l1);
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& ex) {
            ex += abs(onesA(i))*ll.wts(i);
        }, l1denom);
        l1 /= l1denom;
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& er) {
            er += square(onesError(i))*ll.wts(i);
        }, l2);
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& ex) {
            ex += square(onesA(i))*ll.wts(i);
        }, l2denom);
        l2 = std::sqrt(l2/l2denom);
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& er) {
            if (abs(onesError(i)) > er) er = abs(onesError(i));
        }, ko::Max<Real>(linf));
        ko::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& ex) {
            if (abs(onesA(i)) > ex) ex = abs(onesA(i));
        }, ko::Max<Real>(linfdenom));
        
        ErrNorms errs(l1,l2,linf);
        //ErrNorms errs = ll.scalarErrorNorms(onesError, onesA);
        */
        
        
        std::cout << errs.infoString("approx. 1 :: ");
    }
}
ko::finalize();
return 0;
}
