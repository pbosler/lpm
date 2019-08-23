#include "LpmMatlabIO.hpp"

namespace Lpm {

void writeVectorMatlab(std::ostream& os, const std::string name, const ko::View<const Real*,HostMem> v) {
    os << name << " = [";
    for (Index i=0; i<v.extent(0)-1; ++i) 
        os << v(i) << ",";
    os << v(v.extent(0)-1) << "];\n";
}

void writeArrayMatlab(std::ostream& os, const std::string name, const ko::View<const Real**,HostMem> a) {
    os << name << " = [";
    const Index nrow = a.extent(0);
    const Int ncol = a.extent(1);
    for (Index i=0; i<nrow; ++i) {
        for (Int j=0; j<ncol; ++j) {
            os << a(i,j) << (i<nrow-1 ? (j<ncol-1 ? "," : ";") : (j<ncol-1 ? "," : "];\n"));
        }
    }
}


}
