#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dfs/lpm_dfs_grid.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#ifdef LPM_USE_VTK
#include "vtkNew.h"
#include "vtkXMLStructuredGridWriter.h"
#endif

using namespace Lpm;

TEST_CASE("dfs_grid_tests", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_grid_tests", Log::level::info, comm);
  constexpr int nlon = 40;

  DFS::DFSGrid dfs_grid(nlon);
  const int nlat = dfs_grid.nlat;
  logger.info(dfs_grid.info_string());

  const int last_lat_idx = nlat - 1;
  const int last_lon_idx = nlon - 1;
  CHECK(dfs_grid.dtheta == Catch::Approx(dfs_grid.dlambda));
  CHECK((nlat - 1) * dfs_grid.dtheta == Catch::Approx(constants::PI));
  CHECK(nlon * dfs_grid.dlambda == Catch::Approx(2 * constants::PI));
  CHECK(dfs_grid.colatitude(0) == 0);
  CHECK(dfs_grid.colatitude(last_lat_idx) == constants::PI);
  CHECK(dfs_grid.longitude(0) == 0);
  CHECK(dfs_grid.longitude(last_lon_idx) ==
        2 * constants::PI * (1 - 1.0 / nlon));

  Real surf_area;
  const auto md_policy =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nlat, nlon});
  Kokkos::parallel_reduce(
      "spherical area", md_policy,
      KOKKOS_LAMBDA(const Int i, const Int j, Real& a) {
        a += dfs_grid.area_weight(i, j);
      },
      surf_area);

  CHECK(surf_area == Catch::Approx(4 * constants::PI));

  // quadrature weights integrate low-degree polynomials exactly:
  // int 1 dA = 4 pi, int (x^4 + z^6) dA = 4 pi (1/5 + 1/7)
  Real quad_area;
  Real quad_poly;
  Kokkos::parallel_reduce(
      "quadrature area", md_policy,
      KOKKOS_LAMBDA(const Int i, const Int j, Real& a) {
        a += dfs_grid.quadrature_weight(i, j);
      },
      quad_area);
  Kokkos::parallel_reduce(
      "quadrature polynomial", md_policy,
      KOKKOS_LAMBDA(const Int i, const Int j, Real& a) {
        Real xyz[3];
        dfs_grid.sph2xyz(xyz, i, j);
        a += (pow(xyz[0], 4) + pow(xyz[2], 6)) * dfs_grid.quadrature_weight(i, j);
      },
      quad_poly);

  CHECK(quad_area == Catch::Approx(4 * constants::PI).epsilon(1e-13));
  CHECK(quad_poly ==
        Catch::Approx(4 * constants::PI * (1.0 / 5 + 1.0 / 7)).epsilon(1e-13));

#ifdef LPM_USE_VTK
  auto vtk_grid = dfs_grid.vtk_grid();

  const std::string vtk_fname = "dfs_grid_test.vts";
  vtkNew<vtkXMLStructuredGridWriter> writer;
  writer->SetInputData(vtk_grid);
  writer->SetFileName(vtk_fname.c_str());
  writer->Write();
#endif

  logger.info("tests pass.");
}
