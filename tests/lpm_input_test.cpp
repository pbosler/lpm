#include "lpm_input.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_string_util.hpp"

using namespace Lpm;
using namespace Lpm::user;
using Catch::Approx;

TEST_CASE("input", "") {
  Comm comm;

  Logger<> logger("input_unit_tests", Log::level::debug, comm);

  SECTION("Option tests") {
    const Int int_val         = 42;
    const Real real_val       = 3.1415926;
    const std::string str_val = "i_am_a_new_string";

    Option o1("int_opt", "-i", "--integer", "an integer", 2);
    Option o2("real_opt", "-r", "--real", "a real number", 2.71);
    Option o3("str_opt", "-s", "--string", "a string",
              std::string("i am a string"));
    Option o4("bool_opt", "-t", "--true", "a bool", false);
    const std::set<std::string> str_vals({"val1", "val2"});
    Option o5("str_opts", "-ss", "--string-set", "string_set",
              std::string("val1"), str_vals);

    logger.info(o1.info_string());
    logger.info(o2.info_string());
    logger.info(o3.info_string());
    logger.info(o4.info_string());
    logger.info(o5.info_string());

    auto iv  = o1.get_int();
    auto rv  = o2.get_real();
    auto sv  = o3.get_str();
    auto bv  = o4.get_bool();
    auto sv1 = o5.get_str();

    REQUIRE(iv == 2);
    REQUIRE(rv == Approx(2.71));
    REQUIRE(sv == "i am a string");
    REQUIRE(!bv);

    logger.debug("setting new values");

    o1.set_value(int_val);
    o2.set_value(real_val);
    o3.set_value(str_val);
    o4.set_value(true);
    REQUIRE_THROWS(o5.set_value(std::string("not_valid")));
    o5.set_value(std::string("val2"));

    iv  = o1.get_int();
    rv  = o2.get_real();
    sv  = o3.get_str();
    bv  = o4.get_bool();
    sv1 = o5.get_str();

    logger.info(o1.info_string());
    logger.info(o2.info_string());
    logger.info(o3.info_string());
    logger.info(o4.info_string());
    logger.info(o5.info_string());

    REQUIRE(iv == int_val);
    REQUIRE(rv == Approx(real_val));
    REQUIRE(sv == str_val);
    REQUIRE(bv);
    REQUIRE(sv1 == "val2");
    REQUIRE_THROWS(o1.get_real());
  }

  SECTION("Input tests") {
    Option input_file_option("input_file", "-i", "--input", "input file name",
                             std::string("test_file_in.txt"));
    Option output_file_option("output_file", "-o", "--output",
                              "output file name",
                              std::string("test_file_out.txt"));
    Option tfinal_option("tfinal", "-tf", "--time_final", "time final", 0.1);
    Option nsteps_option("nsteps", "-n", "--nsteps", "number of steps", 10);
    Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree depth",
                             3);

    Input input("input_unit_test");
    input.add_option(input_file_option);
    input.add_option(output_file_option);
    input.add_option(tfinal_option);
    input.add_option(nsteps_option);
    input.add_option(tree_depth_option);

    logger.info(input.info_string());
    REQUIRE(input.get_option("tfinal").get_real() == Approx(0.1));
    REQUIRE(input.get_option("nsteps").get_int() == 10);
    REQUIRE(input.get_option("tree_depth").get_int() == 3);

    const int argc = 7;
    char* argv[]   = {"prog_name", "-tf", "1.0", "--nsteps", "200", "-d", "5"};

    input.parse_args(argc, argv);
    logger.info(input.info_string());
    REQUIRE(input.get_option("tfinal").get_real() == Approx(1.0));
    REQUIRE(input.get_option("nsteps").get_int() == 200);
    REQUIRE(input.get_option("tree_depth").get_int() == 5);

    Option duplicate_short_flag_option("duplicate_short_flag", "-i",
                                       "--duplicate-input", "duplicate input",
                                       std::string("duplicate_input.txt"));
    REQUIRE_THROWS(input.add_option(duplicate_short_flag_option));
    Option duplicate_long_flag_option("duplicate_long_flag", "-oo", "--output",
                                      "duplicate output",
                                      std::string("duplicate_out.txt"));
    REQUIRE_THROWS(input.add_option(duplicate_long_flag_option));
  }
}
