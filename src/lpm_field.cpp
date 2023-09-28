#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"

namespace Lpm {

std::string field_loc_string(const FieldLocation& floc) {
  std::string result;
  switch (floc) {
    case (ParticleField): {
      result = "particle_field";
      break;
    }
    case (VertexField): {
      result = "vertex_field";
      break;
    }
    case (EdgeField): {
      result = "edge_field";
      break;
    }
    case (FaceField): {
      result = "face_field";
      break;
    }
  }
  return result;
}


}  // namespace Lpm
