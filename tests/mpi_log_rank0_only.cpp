#include "lpm_logger.hpp"
#include "lpm_log_file.hpp"
#include "lpm_comm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>

using namespace Lpm;

TEST_CASE("log console: rank 0 only, log file: all ranks", "[logging]") {

  Comm comm(MPI_COMM_WORLD);

  Logger<LogBasicFile<Log::level::debug>> mylog("console_rank0_only", Log::level::debug, comm);
  mylog.console_output_rank0_only(comm);
  const std::string logfilename = "console_rank0_only_rank" + std::to_string(comm.rank())
     + "_logfile.txt";
  mylog.warn("if you see this in the console from any rank except 0, something is wrong.\n It will show up in every file.");
  mylog.info("that previous message covered multiple lines.");

  // verify that all ranks produce a file
  std::ifstream lf(logfilename);
  REQUIRE( lf.is_open() );

  REQUIRE( mylog.logfile_name() == logfilename);
}

TEST_CASE("log console: rank 0 only; log file: rank 0 only", "[logging]") {
  Comm comm(MPI_COMM_WORLD);

  Logger<LogBasicFile<Log::level::info>> mylog("all_output_rank0_only", Log::level::info, comm);
  mylog.console_output_rank0_only(comm);
  mylog.file_output_rank0_only(comm);

  mylog.warn("if you see this in the console or a file from any rank except 0, something is wrong.");

  const std::string logfilename = "all_output_rank0_only_rank" + std::to_string(comm.rank())
     + "_logfile.txt";
  std::ifstream lf(logfilename);
  if (comm.i_am_root()) {
    REQUIRE(mylog.logfile_name() == logfilename);
    REQUIRE(lf.good());
  }
  else {
    REQUIRE(mylog.logfile_name() == "null");
    REQUIRE(!lf.good());
  }
}
