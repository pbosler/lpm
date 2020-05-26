#ifndef LPM_SPHERE_VORONOI_MESH_HPP
#define LPM_SPHERE_VORONOI_MESH_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmMeshSeed.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include <vector>
#include <set>

namespace Lpm {
namespace Voronoi {

template <typename SeedType>
struct VoronoiMesh {
    static constexpr Short MAX_POLYGON_SIDES = 20;
    static constexpr Short MAX_VERTEX_DEGREE = 10;

    std::vector<WingedEdge> edges;
    std::vector<Vertex> vertices;
    std::vector<Cell> cells;

    Real surfarea;

    VoronoiMesh(const MeshSeed<SeedType>& seed, const Short& init_refinement_level);

    inline Index nverts() const {return vertices.size();}
    inline Index nedges() const {return edges.size();}
    inline Index ncells() const {return cells.size();}

    Real cellArea(const Index& cell_ind);

    void updateSurfArea();

    /**
        Returns the index of the cell whose generating point is closest to xyz.

        Performs a walk search, worst case performance is O(sqrt(N)).
    */
    Index cellContainingPoint(const Real* xyz, const Index start_search) const;

    /**
        Finds the corner of cells[cell_ind] that is closest to xyz.

        Note: If cells[cell_ind] is the cell that contains xyz, this finds the absolute closest corner in the whole mesh.
    */
    Index nearestCornerToPoint(const Real* xyz, const Index cell_ind) const;

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

    std::vector<Index> getAdjacentCells(const Index& cell_ind) const;

    /// returns true if the new_xyz point "breaks" corner vert_ind
    bool brokenCorner(const Real* new_xyz, const Index& vert_ind) const;

	std::string infoString(const bool& verbose=false) const;

	void insertCellAtPoint(const Real* xyz, const Index& first_guess_cell_ind=0);

	protected:
	    std::vector<bool> cell_flags;  // cell_flags[i] = true if cell i is incident to a broken vertex
	    std::vector<bool> vertex_flags; // vertex_flags[i] = true if vertex i is broken

	    void seedInit(const MeshSeed<SeedType>& seed);

	    std::vector<Index> buildBrokenCornerList(const Real* xyz, const Index& first_bc, const bool& verbose=false) const;

	    void pointInsertionBookkeeping(std::vector<Index>& edges_to_delete, std::vector<Index>& edges_to_update_ccw,
	        std::vector<Index>& new_vert_inds, std::vector<CCWVertexGenerators>& gens, std::vector<Index>& new_edge_inds,
	        const std::vector<Index>& broken_corners);

	    void makeNewVertices(std::vector<Vertex>& newverts, const std::vector<CCWVertexGenerators>& gens,
	        const Real* xyz, const std::vector<Index>& edges_to_update_ccw) const;

};

}}
#endif
