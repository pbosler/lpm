#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "catch.hpp"

using namespace Lpm;

std::string mem_header() {
  std::ostringstream ss;
  ss << std::setw(20) << "tree level" << std::setw(20) << "nverts" << std::setw(20) \
        << "nedges" << std::setw(20) << "nfaces\n";
  return ss.str();
}

std::string mem_line(const int lev, const int nv, const int ne, const int nf) {
  std::ostringstream ss;
  ss << std::setw(20) << lev << std::setw(20) << nv << std::setw(20) \
            << ne << std::setw(20) << nf << "\n";
  return ss.str();
}

template <typename SeedType>
bool seed_test(const MeshSeed<SeedType>& seed, const Comm& comm) {
  bool result_pass = true;

  const Int maxlev = 9;
  Index nmax_verts;
  Index nmax_faces;
  Index nmax_edges;

  Logger<> logger("mesh_seed_test_"+seed.id_string(), Log::level::info, comm);

  logger.info(seed.info_string());
  logger.info("seed memory requirements");
  logger.info(mem_header());
  for (int i=0; i<maxlev; ++i) {
    seed.set_max_allocations(nmax_verts, nmax_edges, nmax_faces, i);
    logger.info(mem_line(i, nmax_verts, nmax_edges, nmax_faces));
  }

  return result_pass;
}


TEST_CASE( "mesh_seed", "") {

    Comm comm;

    MeshSeed<QuadRectSeed> qrseed;
    bool pass = seed_test(qrseed, comm);
    REQUIRE( pass );

    MeshSeed<TriHexSeed> thseed;
    pass = seed_test(thseed, comm);
    REQUIRE( pass );

    MeshSeed<CubedSphereSeed> csseed;
    pass = seed_test(csseed, comm);
    REQUIRE (pass);

    MeshSeed<UnitDiskSeed> udseed;
    pass = seed_test(udseed, comm);
    REQUIRE(pass);
    Real sa = 0.0;
    for (Int i=0; i<UnitDiskSeed::nfaces; ++i) {
      sa += udseed.face_area(i);
    }
    REQUIRE(FloatingPoint<Real>::equiv(sa, constants::PI));

    MeshSeed<IcosTriSphereSeed> icseed;
    pass = seed_test(icseed, comm);
    REQUIRE(pass);


}

