#include "LpmErrorNorms.hpp"
#include <sstream>

namespace Lpm {

std::string ErrNorms::infoString(const std::string& label, const int tab_level) const {
    std::ostringstream ss;
    std::string tabstr;
    for (int i=0; i<tab_level; ++i) tabstr += "\t";
    ss << tabstr << label << " ErrNorms: l1 = " << l1 << " l2 = " << l2 << " linf = " << linf << "\n";
    return ss.str();
}

}
