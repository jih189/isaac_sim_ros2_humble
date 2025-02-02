#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <vector>
#include <iostream>
#include <stdexcept>

// Boost headers
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

// add cuda header
#include <cuda_runtime.h>
#include <memory>
#include <util.h>

namespace CUDAMPLib
{

    class Graph {
    public:

        // Add a set of states (configurations) to the graph.
        void add_states(const std::vector<std::vector<float>>& qs);

        // Returns true if the configuration q is in the graph.
        bool contain(const std::vector<float>& q) const;

        // Connect two states (configurations) by adding an edge between them.
        // The edge weight is the Euclidean distance between the configurations.
        void connect(const std::vector<float>& q1, const std::vector<float>& q2);

        // Check if two states are directly connected by an edge.
        bool is_connect(const std::vector<float>& q1, const std::vector<float>& q2) const;

        // Find the shortest path from configuration q1 (start) to q2 (goal)
        // using Dijkstra's algorithm. Returns the path as a vector of configurations.
        std::vector<std::vector<float>> find_path(const std::vector<float>& q1, const std::vector<float>& q2) const;

        // Compute the Euclidean distance from configuration q to all other configurations in the graph.
        std::vector<float> compute_distance(const std::vector<float> &q) const;

        // Constructor.
        Graph(int dim) : dim(dim), has_d_graph(false) {}

        // Destructor.
        ~Graph();

    private:
        // Internal Boost Graph definitions.

        // Vertex property: stores a configuration.
        struct VertexProperties {
            std::vector<float> configuration;
        };

        // Edge property: stores the weight (Euclidean distance between configurations).
        struct EdgeProperties {
            double weight;
        };

        // Define the Boost graph type.
        // We use vecS for vertices and edges (so vertices get automatic indices),
        // and an undirected graph.
        using BoostGraph = boost::adjacency_list<
            boost::vecS,         // OutEdgeList
            boost::vecS,         // VertexList
            boost::undirectedS,  // Undirected graph
            VertexProperties,    // Vertex properties
            EdgeProperties       // Edge properties
        >;

        // Convenience typedef for a vertex descriptor.
        using Vertex = boost::graph_traits<BoostGraph>::vertex_descriptor;

        // The underlying Boost graph instance.
        BoostGraph g;

        // Helper: find a vertex by comparing its stored configuration.
        // Returns the vertex descriptor if found, or null_vertex() if not.
        Vertex find_vertex_by_config(const std::vector<float>& q) const;

        // add state to the graph
        void add_state(const std::vector<float>& q);

        // dimension of the configuration
        size_t dim;

        // d_graph
        float* d_graph;
        float* d_distance_map;
        bool has_d_graph;
    };
} // namespace CUDAMPLib
