#ifndef LPM_SPHERE_VORONOI_PRIMITIVES_HPP
#define LPM_SPHERE_VORONOI_PRIMITIVES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include <string>

namespace Lpm {
namespace Voronoi {

/**
		ccw_orig									   cw_dest
			 \											 /
  			  \					left_cell				/
   			   \									   /
	orig_vertex O---->>>>--- this edge --->>>---------O  dest_vertex
  			   /									   \
			  /					right_cell				\
			 /											 \
		   cw_orig									   ccw_dest
*/


struct Cell {
	Real xyz[3];
	Real latlon[2];
	Index edgeAroundCell;  // Index of an edge (any edge) around *this cell

    Cell(const Real* xx, const Index& edge) : edgeAroundCell(edge) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    const Real lat = SphereGeometry::latitude(xyz);
	    const Real lon = SphereGeometry::longitude(xyz);
	    latlon[0] = lat;
	    latlon[1] = lon;
    }
	std::string infoString(const Short tab_level=0, const Short id=-1) const;
};

struct Vertex {
	Real xyz[3];
	Real latlon[2];
	Index edgeAtVertex; // Index of an edge (any edge) incident to *this vertex

	Vertex(const Real* xx, const Index& edge) : edgeAtVertex(edge) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    const Real lat = SphereGeometry::latitude(xyz);
	    const Real lon = SphereGeometry::longitude(xyz);
	    latlon[0] = lat;
	    latlon[1] = lon;
	}

	std::string infoString(const Short tab_level=0, const Short id=-1) const;
};

struct WingedEdge {
    Real xyz[3];
    Real latlon[2];

    Index orig_vertex;
	Index dest_vertex;
	Index right_cell;
	Index left_cell;
	Index cw_orig; // from the origin vertex, this is the next edge (after *this) at that vertex in clockwise order
	Index ccw_orig; // from the origin vertex, this is the next edge (after *this) at that vertex in counter-clockwise order
	Index cw_dest; //  the destination vertex, this is the next edge (after *this) at that vertex in clockwise order
	Index ccw_dest; // the destination vertex, this is the next edge (after *this) at that vertex in counter-clockwise order

	WingedEdge(const Real* xx, const Index& orig, const Index& dest, const Index& left, const Index& right,
	    const Index& cwo, const Index& ccwo, const Index& cwd, const Index& ccwd) : orig_vertex(orig), dest_vertex(dest),
	    right_cell(right), left_cell(left), cw_orig(cwo), ccw_orig(ccwo), cw_dest(cwd), ccw_dest(ccwd) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    latlon[0] = SphereGeometry::latitude(xyz);
	    latlon[1] = SphereGeometry::longitude(xyz);
	}

	WingedEdge(const Index& orig, const Index& dest, const Index& left, const Index& right,
	    const Index& cwo, const Index& ccwo, const Index& cwd, const Index& ccwd, const std::vector<Vertex>& vertices) :
	    orig_vertex(orig), dest_vertex(dest), left_cell(left), right_cell(right), cw_orig(cwo), ccw_orig(ccwo),
	    cw_dest(cwd), ccw_dest(ccwd) {
        SphereGeometry::midpoint(xyz, vertices[orig_vertex].xyz, vertices[dest_vertex].xyz);
        latlon[0] = SphereGeometry::latitude(xyz);
        latlon[1] = SphereGeometry::longitude(xyz);
	}

	std::string infoString(const Short tab_level=0, const Short id=-1) const;
};

Index nfacesAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev);

Index nvertsAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev);

}
}
#endif
