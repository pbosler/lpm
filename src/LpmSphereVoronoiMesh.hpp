#ifndef LPM_SPHERE_VORONOI_MESH_HPP
#define LPM_SPHERE_VORONOI_MESH_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include <vector>

namespace Lpm {
namespace Voronoi {

template <typename SeedType>
struct VoronoiMesh {
    static constexpr Short MAX_POLYGON_SIDES = 10;
    static constexpr Short MAX_VERTEX_DEGREE = 10;

    std::vector<WingedEdge> edges;
    std::vector<Vertex> vertices;
    std::vector<Cell> cells;

    VoronoiMesh(const MeshSeed<SeedType>& seed, const Short& init_refinement_level);

    inline Index nverts() const {return vertices.size();}
    inline Index nedges() const {return edges.size();}
    inline Index ncells() const {return cells.size();}

    /**
		Okabe et. al. algorithm 4.2.1
		Retrieve the edges and cells incident to a Vertex

		Output:
			edge_inds: list of all edge indices at the vertex
			cell_inds: list of all cell indices at the vertex
		Input:
			vert_ind: index of vertex

	*/
	void getEdgesAndCellsAtVertex(std::vector<Index>& edge_inds, std::vector<Index>& cell_inds,
		const Index& vert_ind) const;

	/**
		Okabe et. al. algorithm 4.2.2
		Retrieve the edges and vertices surrounding a cell

		Output:
			edge_inds: list of all edges around Cell
			vert_inds: list of all vertices in cell
		Input:
			cell_ind: index of a cell
	*/
	void getEdgesAndVerticesInCell(std::vector<Index>& edge_inds, std::vector<Index>& vert_inds,
		const Index& cell_ind) const;

	std::string infoString(const bool& verbose=false) const;

	protected:
	    void seedInit(const MeshSeed<SeedType>& seed);
};

}}
#endif
