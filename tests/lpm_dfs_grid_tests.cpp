#include "lpm_dfs_grid.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#include <catch2/catch_test_macros.hpp>
#ifdef LPM_USE_VTK
#include "vtkNew.h"
#include "vtkXMLStructuredGridWriter.h"
#endif

using namespace Lpm;

TEST_CASE("dfs_grid_tests", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_grid_tests", Log::level::info, comm);
  constexpr int nlon = 180;

  DFS::DFSGrid dfs_grid(nlon);
  const int nlat = dfs_grid.nlat;
  logger.info("DFSGrid with nlon = {}, nlat = {} created.", nlon, nlat);

  const int last_lat_idx = nlat-1;
  const int last_lon_idx = nlon-1;
  CHECK(dfs_grid.colatitude(0) == 0);
  CHECK(dfs_grid.colatitude(last_lat_idx) == constants::PI);
  CHECK(dfs_grid.longitude(0) == 0);
  CHECK(dfs_grid.longitude(last_lon_idx) == 2*constants::PI*(1 - 1.0/nlon));

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
