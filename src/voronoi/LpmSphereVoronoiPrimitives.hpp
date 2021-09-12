#ifndef LPM_SPHERE_VORONOI_PRIMITIVES_HPP
#define LPM_SPHERE_VORONOI_PRIMITIVES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include <string>
#include <vector>

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
	Index id;
	Real area;

    Cell(const Real* xx, const Index& edge) : edgeAroundCell(edge), id(-1), area(0) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    const Real lat = SphereGeometry::latitude(xyz);
	    const Real lon = SphereGeometry::longitude(xyz);
	    latlon[0] = lat;
	    latlon[1] = lon;
    }

    Cell(const Cell& other) : edgeAroundCell(other.edgeAroundCell), id(other.id), area(other.area) {
        for (Short i=0; i<3; ++i) {
            xyz[i] = other.xyz[i];
        }
        for (Short i=0; i<2; ++i) {
            latlon[i] = other.latlon[i];
        }
    }

    Cell& operator = (const Cell& other) {
        this->id = other.id;
        this->edgeAroundCell = other.edgeAroundCell;
        this->area = other.area;
        for (Short i=0; i<3; ++i) {
            this->xyz[i] = other.xyz[i];
        }
        for (Short i=0; i<2; ++i) {
            this->latlon[i] = other.latlon[i];
        }
        return *this;
    }

	std::string infoString(const Short tab_level=0) const;
};

struct Vertex {
	Real xyz[3];
	Real latlon[2];
	Index edgeAtVertex; // Index of an edge (any edge) incident to *this vertex
	Real circumradius;
	Index id;

	Vertex() : edgeAtVertex(-1), circumradius(0), id(-1) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = 0;
	    }
	    for (Short i=0; i<2; ++i) {
	        latlon[i] = 0;
	    }
	}

	Vertex(const Real* xx, const Index& edge) : edgeAtVertex(edge), circumradius(0), id(-1) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    const Real lat = SphereGeometry::latitude(xyz);
	    const Real lon = SphereGeometry::longitude(xyz);
	    latlon[0] = lat;
	    latlon[1] = lon;
	}

    Vertex(const Vertex& other) : edgeAtVertex(other.edgeAtVertex), circumradius(other.circumradius), id(other.id) {
        for (Short i=0; i<3; ++i) {
            xyz[i] = other.xyz[i];
        }
        for (Short i=0; i<2; ++i) {
            latlon[i] = other.latlon[i];
        }
    }

    Vertex& operator = (const Vertex& other) {
        this->id = other.id;
        this->edgeAtVertex = other.edgeAtVertex;
        this->circumradius = other.circumradius;
        for (Short i=0; i<3; ++i) {
            this->xyz[i] = other.xyz[i];
        }
        for (Short i=0; i<2; ++i) {
            this->latlon[i] = other.latlon[i];
        }
        return *this;
    }

	std::string infoString(const Short tab_level=0) const;
};

struct WingedEdge {
    Real xyz[3];
    Real latlon[2];
    Real len;
    Index id;

    Index orig_vertex;
	Index dest_vertex;
	Index right_cell;
	Index left_cell;
	Index cw_orig; // from the origin vertex, this is the next edge (after *this) at that vertex in clockwise order
	Index ccw_orig; // from the origin vertex, this is the next edge (after *this) at that vertex in counter-clockwise order
	Index cw_dest; //  the destination vertex, this is the next edge (after *this) at that vertex in clockwise order
	Index ccw_dest; // the destination vertex, this is the next edge (after *this) at that vertex in counter-clockwise order

	WingedEdge() : id(-1), orig_vertex(-1), dest_vertex(-1), left_cell(-1), right_cell(-1), cw_orig(-1), ccw_orig(-1),
	    cw_dest(-1), ccw_dest(-1), len(0) {
	    for (Short i=0; i<2; ++i) {
	        xyz[i] = 0;
	        latlon[i] = 0;
	    }
	    xyz[2] = 0;
    }

	WingedEdge(const Real* xx, const Index& orig, const Index& dest, const Index& left, const Index& right,
	    const Index& cwo, const Index& ccwo, const Index& cwd, const Index& ccwd) : id(-1), orig_vertex(orig), dest_vertex(dest),
	    right_cell(right), left_cell(left), cw_orig(cwo), ccw_orig(ccwo), cw_dest(cwd), ccw_dest(ccwd), len(0) {
	    for (Short i=0; i<3; ++i) {
	        xyz[i] = xx[i];
	    }
	    latlon[0] = SphereGeometry::latitude(xyz);
	    latlon[1] = SphereGeometry::longitude(xyz);
	}

	WingedEdge(const Index& orig, const Index& dest, const Index& left, const Index& right,
	    const Index& cwo, const Index& ccwo, const Index& cwd, const Index& ccwd, const std::vector<Vertex>& vertices) :
	    orig_vertex(orig), dest_vertex(dest), left_cell(left), right_cell(right), cw_orig(cwo), ccw_orig(ccwo),
	    cw_dest(cwd), ccw_dest(ccwd), id(-1) {
        SphereGeometry::midpoint(xyz, vertices[orig_vertex].xyz, vertices[dest_vertex].xyz);
        latlon[0] = SphereGeometry::latitude(xyz);
        latlon[1] = SphereGeometry::longitude(xyz);
        len = SphereGeometry::sqEuclideanDistance(vertices[dest_vertex].xyz, vertices[orig_vertex].xyz);
	}

    WingedEdge& operator = (const WingedEdge& other) {
        for (Short i=0; i<2; ++i) {
            this->xyz[i] = other.xyz[i];
            this->latlon[i] = other.latlon[i];
        }
        this->id = other.id;
        this->xyz[2] = other.xyz[2];
        this->orig_vertex = other.orig_vertex;
        this->dest_vertex = other.dest_vertex;
        this->right_cell = other.right_cell;
        this->left_cell = other.left_cell;
        this->cw_orig = other.cw_orig;
        this->ccw_orig = other.ccw_orig;
        this->cw_dest = other.cw_dest;
        this->ccw_dest = other.ccw_dest;
        this->len = other.len;
        return *this;
    }

    void setLength(const std::vector<Vertex>& verts);

	std::string infoString(const Short tab_level=0) const;
};

Index nfacesAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev);

Index nvertsAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev);

struct EdgeCellIndexPair {
    Index edge_index;
    Index cell_index;

    EdgeCellIndexPair(const Index& e, const Index& c) : edge_index(e), cell_index(c) {}
};

bool operator < (const EdgeCellIndexPair& left, const EdgeCellIndexPair& right);

struct CCWVertexGenerators {
    Index cell_a;
    Index cell_b;
    Index cell_c;

    CCWVertexGenerators(const Index& a, const Index& b, const Index& c) : cell_a(a), cell_b(b), cell_c(c) {}
};

std::ostream& operator << (std::ostream& os, const CCWVertexGenerators& gen);

}
}
#endif
