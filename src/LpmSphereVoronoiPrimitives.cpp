#include <iostream>
#include <iomanip>
#include <sstream>

#include "LpmSphereVoronoiPrimitives.hpp"

namespace Lpm {
namespace Voronoi {

std::string Cell::infoString(const Short tab_level, const Short id) const {
    std::stringstream ss;
    std::string tabstr;
    for (Short i=0; i<tab_level; ++i) {
        tabstr += "\t";
    }
    ss << tabstr << "Cell " << (id>=0 ? std::to_string(id) : "") << " info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") edgeAroundCell = " << edgeAroundCell << "\n";
    return ss.str();
}

std::string Vertex::infoString(const Short tab_level, const Short id) const {
    std::string tabstr;
    for (Short i=0; i<tab_level; ++i) {
        tabstr += "\t";
    }
    std::ostringstream ss;
    ss << tabstr <<  "Vertex " << (id>=0 ? std::to_string(id) : "") << " info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") edgeAtVertex = " << edgeAtVertex << "\n";
    return ss.str();
}

std::string WingedEdge::infoString(const Short tab_level, const Short id) const {
    std::ostringstream ss;
    std::string tabstr;
    for (Short i=0; i<tab_level; ++i) {
        tabstr += "\t";
    }
    ss << tabstr << "WingedEdge " << (id>=0 ? std::to_string(id) : "") << " info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") orig_vertex = " << orig_vertex << " dest_vertex = " << dest_vertex
       << " cw_orig = " << cw_orig << " ccw_orig = " << ccw_orig << " cw_dest = " << cw_dest
       << " ccw_dest = " << ccw_dest << "\n";
    return ss.str();
}

Index nfacesAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev) {
    Index nf = nrootfaces;
    Index nv = 2*nf-4;
    for (Short i=1; i<=lev; ++i) {
        nf = nf + nv;
        nv = 2*nf - 4;
    }
    return nf;
}

Index nvertsAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev) {
    Index nf = nrootfaces;
    Index nv = 2*nf-4;
    for (Short i=1; i<=lev; ++i) {
        nf = nf + nv;
        nv = 2*nf-4;
    }
    return nv;
}


}}
