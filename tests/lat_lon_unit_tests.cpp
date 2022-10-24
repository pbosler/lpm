#include "lpm_lat_lon_pts.hpp"
#include "lpm_constants.hpp"
#include "catch.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"

#include <vector>
#include <fstream>

using namespace Lpm;

TEST_CASE("lat_lon_unit_tests", "") {
  Comm comm;
  Logger<> logger("lat_lon_tests", Log::level::debug, comm);

  const std::vector<Int> nlats = {91, 181, 361, 721};
  const std::vector<Int> nlons = {180, 360, 720, 1440};

  REQUIRE(nlats.size() == nlons.size());

  for (int i=0; i<nlats.size(); ++i) {
    const Int nlat = nlats[i];
    const Int nlon = nlons[i];
    const Int npts = nlat*nlon;

    LatLonPts ll(nlat, nlon);

    CHECK(ll.dlambda == Approx(ll.dtheta));

    Real surf_area = 0;
    Kokkos::parallel_reduce(npts, KOKKOS_LAMBDA (const Index& i, Real& a) {
      a += ll.wts(i);
    }, surf_area);

    logger.info("ll resolution {} surface area error = {}",
      ll.dlambda * constants::RAD2DEG, std::abs(4*constants::PI - surf_area) / (4*constants::PI));

    scalar_view_type ones("ones", npts);
    Kokkos::deep_copy(ones, 1);
    auto h_ones = Kokkos::create_mirror_view(ones);
    Kokkos::deep_copy(h_ones, ones);

    if (i==0) {
      std::ofstream mfile("latlon_test.m");
      ll.write_grid_matlab(mfile, "ll");
      ll.write_scalar_field_matlab(mfile, "ones", h_ones);
      mfile.close();
    }

  }

}
