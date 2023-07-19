#include "lpm_logger.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <fstream>

using namespace Lpm;

TEST_CASE("basic logger", "[logging]") {

  Comm comm;

  SECTION("console only, default comm") {
    // setup a console-only logger
    Logger<LogNoFile> mylog("ekat_log_test_console_only", Log::level::debug, comm);

    mylog.info("This is a console-only message, with level = info");
    mylog.error("Here is an error message.");
    mylog.error("Here is an error message with a number: {}", 4);

    // check that this log did not produce a file
    const std::string logfilename = "ekat_log_test_console_only_logfile.txt";
    std::ifstream lf(logfilename);
    REQUIRE( !lf.is_open() );

    REQUIRE(mylog.logfile_name() == "null");
  }

  SECTION("console and file logging, with mpi rank info") {

      Logger<LogBasicFile<Log::level::trace>>
        mylog("combined_console_file_mpi", Log::level::debug, comm);

      mylog.debug("here is a debug message that will also show up in this rank's log file.");

      // the file level is trace, but the log level is debug (debug > trace); trace messages will be skipped.
      mylog.trace("this message won't show up anywhere.");
      REQUIRE( !mylog.should_log(Log::level::trace) );

      // verify that this log did produce a file
      const std::string logfilename = "combined_console_file_mpi_rank0_logfile.txt";
      std::ifstream lf(logfilename);
      REQUIRE( lf.is_open() );

      REQUIRE( mylog.logfile_name() == logfilename);
    }

    SECTION("Debug excludes Trace") {
      Logger<LogBasicFile<Log::level::debug>>
        mylog("debug_excludes_trace", Log::level::debug, comm);

      mylog.debug("this debug message will show up in both the console and the log file.");
      mylog.trace("Entered 'debug excludes tracer' section; this message will not show up anywhere.");
    }

    SECTION("log file check") {
      // The log file is not necessarily written as you write log messages.
      // For performance reasons, log messages may first be collated into a buffer; that
      // buffer is then written according to spdlog's internal logic.
      // This means that in order to check the previous section's log file, we have to
      // wait for its log to go out of scope (which guarantees that the file will be written).

      const std::string lfname = "debug_excludes_trace_rank0_logfile.txt";
      std::ifstream lf(lfname);
      REQUIRE(lf.is_open());
      std::stringstream ss;
      ss << lf.rdbuf();

      REQUIRE( ss.str().find("[trace]") == std::string::npos );
      REQUIRE( ss.str().find("[debug]") != std::string::npos );
    }

    SECTION("create your own sinks") {
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"


      using file_policy = LogNoFile;
      const std::string log_name = "external_sink_log";
      auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      auto file = file_policy::get_file_sink(log_name);

      Logger<file_policy> mylog(log_name, Log::level::warn, console, file, comm);

      mylog.warn("This is a test.  Here is a warning message.  This is only a test.");
    }
}

