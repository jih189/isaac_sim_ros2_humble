#pragma once

// Boost headers
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include "base/Graph.h"
#include "states/SingleArmStates.h"

namespace CUDAMPLib
{
    class SingleArmGraph : public BaseGraph
    {
        public:
            SingleArmGraph(int num_of_joints);

            ~SingleArmGraph();

            // Add a states to the graph.
            void add_states(const BaseStatesPtr & states) override;

            // Add start states to the graph.
            void add_start_states(const BaseStatesPtr & states) override;

            // Add goal states to the graph.
            void add_goal_states(const BaseStatesPtr & states) override;

            // Get motions to k nearest neighbors.
            BaseMotionsPtr get_motions_to_k_nearest_neighbors(const BaseStatesPtr & states, int k, std::vector<StateIndexPair> pairs) override;

            // Returns true if the configuration q is in the graph.
            void contain(const BaseStatesPtr & states, int * d_result) const override;

            // Connect two states by adding an edge between them.
            // The edge weight is the Euclidean distance between the configurations.
            void connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, float * d_weight) override;

            // Check if two states are directly connected by an edge.
            void is_connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, int * d_result) const override;

            // debug print
            void print() const override;

        private:

            int num_of_joints;

            float * d_states_in_graph; // states in graph

            // Vertex property: stores a location of the state in device memory.
            struct VertexProperties {
                int location;
                int group; // group of the state in the graph. Start, goal, or others.
                           // 0: others, 1: start, 2: goal
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

            // get size of the graph
            int get_size() const{
                return boost::num_vertices(graph);
            }

            // helper function to add states to the graph
            void add_states_helper(const BaseStatesPtr & states, int group);
    };

    typedef std::shared_ptr<SingleArmGraph> SingleArmGraphPtr;

} // namespace CUDAMPLib