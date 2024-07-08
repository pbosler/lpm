#include "mesh/lpm_bivar_remesh.hpp"
#include "mesh/lpm_bivar_remesh_impl.hpp"

namespace Lpm {

template class BivarRemesh<QuadRectSeed>;
template class BivarRemesh<TriHexSeed>;

}
