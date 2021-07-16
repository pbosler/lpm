#include "lpm_field.hpp"
#include "util/lpm_string_util.hpp"
#include <sstream>

namespace Lpm {

std::string field_loc_string(const FieldLocation& floc) {
  std::string result;
  switch (floc) {
    case (ParticleField) : {
      result = "particle_field";
      break;
    }
    case (VertexField) : {
      result = "vertex_field";
      break;
    }
    case (EdgeField) : {
      result = "edge_field";
      break;
    }
    case (FaceField) : {
      result = "face_field";
      break;
    }
  }
  return result;
}

template <FieldLocation FL>
std::string ScalarField<FL>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "ScalarField info:\n";
  tabstr += "\t";
  for (auto& md : metadata) {
    ss << tabstr << md.first << ": " << md.second << "\n";
  }
  return ss.str();
};

template <typename Geo, FieldLocation FL>
std::string VectorField<Geo, FL>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "VectorField info:\n";
  tabstr += "\t";
  for (auto& md : metadata) {
    ss << tabstr << md.first << ": " << md.second << "\n";
  }
  return ss.str();
};


}
