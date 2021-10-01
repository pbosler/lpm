#ifndef LPM_HEADER_HPP
#define LPM_HEADER_HPP

#include "LpmConfig.h"

#define BOX_PADDING 1e-5
#define MAX_OCTREE_DEPTH 10
#define MAX_QUADTREE_DEPTH 16
#define WORD_MASK 0xFFFFFFFF
#define NULL_IDX -1

namespace Lpm {

typedef uint_fast32_t id_type;
typedef uint32_t key_type;
typedef uint_fast64_t code_type;

namespace quadtree {


}

namespace octree {

}



} // namespace Lpm

#endif
