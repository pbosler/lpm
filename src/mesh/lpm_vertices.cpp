#include "lpm_vertices.hpp"

#include "lpm_vertices_impl.hpp"

namespace Lpm {

// ETI
template class Vertices<Coords<PlaneGeometry>>;
template class Vertices<Coords<SphereGeometry>>;
// template class Vertices<Coords<CircularPlaneGeometry>>;

}  // namespace Lpm
