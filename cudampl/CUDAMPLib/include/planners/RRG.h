#pragma once

#include <base/Planner.h>

// Boost headers
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

namespace CUDAMPLib
{
    class RRG : public BasePlanner
    {
        public:
            RRG(BaseSpacePtr space);
            ~RRG() override;

            void setMotionTask(BaseTaskPtr task) override;
            void solve() override;
        private:
            BaseStateManagerPtr state_manager;

            // parameters
            int sample_attempts;
            int k;

            // graph
            struct VertexProperties {
                int index_in_manager; // index of the state in the state manager
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

            // start and goal nodes vector
            std::vector<BoostVertex> start_nodes;
            BoostVertex start_node;
            std::vector<BoostVertex> goal_nodes;
            BoostVertex goal_node;
    };

    typedef std::shared_ptr<RRG> RRGPtr;
} // namespace CUDAMPLibs