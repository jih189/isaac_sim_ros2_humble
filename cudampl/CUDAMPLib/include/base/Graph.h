#pragma once

// Boost headers
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include <base/States.h>

namespace CUDAMPLib
{
    class BaseGraph {
        public:
            BaseGraph(int dim) : dim(dim) {}

            virtual ~BaseGraph() {}

            // Add a states to the graph.
            virtual void add_states(const BaseStatesPtr & states) = 0;

            // Returns true if the configuration q is in the graph.
            virtual void contain(const BaseStatesPtr & states, int * d_result) const = 0;

            // Connect two states by adding an edge between them.
            // The edge weight is the Euclidean distance between the configurations.
            virtual void connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, float * d_weight) = 0;

            // Check if two states are directly connected by an edge.
            virtual void is_connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, int * d_result) const = 0;

        private:

            // Vertex property: stores a location of the state in device memory.
            struct VertexProperties {
                int location;
            };

            // Edge property: stores the weight
            struct EdgeProperties {
                float weight;
            };

            // Internal Boost Graph definitions.
            using BoostGraph = boost::adjacency_list<
                boost::vecS,         // OutEdgeList
                boost::vecS,         // VertexList
                boost::undirectedS,  // Undirected graph
                VertexProperties,    // Vertex properties
                EdgeProperties       // Edge properties
            >;

            // Convenience typedef for a vertex descriptor.
            using BoostVertex = boost::graph_traits<BoostGraph>::vertex_descriptor;

            // The Boost graph.
            BoostGraph graph;

            size_t dim;
    };

    typedef std::shared_ptr<BaseGraph> BaseGraphPtr;

} // namespace CUDAMPLib