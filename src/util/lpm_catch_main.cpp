#define CATCH_CONFIG_RUNNER

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_test_utils.hpp"

#include "catch.hpp"

#include "Kokkos_Core.hpp"

#include <mpi.h>


int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  // Read LPM-specific args
  /*
    We expect a comma-separated list of key=value pairs, e.g.,
    key1=val1,key2=val2...

    This lambda reads them...
  */
  auto const readCommaSeparatedParams = [] (const std::string& cmd_line_arg) {
    if (cmd_line_arg == "") {
      return;
    }
    auto& ts = Lpm::TestSession::get();
    std::stringstream input(cmd_line_arg);
    std::string option;
    while (getline(input, option, ',')) {
      auto eq_pos = option.find('=');
      LPM_REQUIRE_MSG(eq_pos != std::string::npos,
        "Error: incorrect format for command line options\n");
      std::string key = option.substr(0,eq_pos);
      std::string val = option.substr(eq_pos+1);
      LPM_REQUIRE_MSG(key != "", "Error: empty key found in command line options.\n");
      LPM_REQUIRE_MSG(val != "", "Error: empty value found in command line options.\n");
      ts.params[key] = val;
    }
  };

  // Catch requires exactly one instance of its session
  Catch::Session catch_session;

  // build a command line parser from catch2
  auto cli = catch_session.cli();
  cli |= Catch::clara::Opt(readCommaSeparatedParams, "key1=val1[,key2=val2[,...]]")
    ["--lpm-test-params"] ("list of parameters to forward to the test");
  catch_session.cli(cli);

  LPM_REQUIRE_MSG(catch_session.applyCommandLine(argc, argv)==0,
    "Error: something went wrong while parsing command line.\n");

  Kokkos::initialize(argc, argv);

  const int num_failed = catch_session.run(argc, argv);


  Kokkos::finalize();

  MPI_Finalize();


return (num_failed == 0 ? 0 : 1);
}
