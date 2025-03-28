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
        /**
            @brief Rapidly-exploring Random Graph (RRG) planner.
            The graph is constructed by adding vertices and edges. The vertices are the states and the edges are the motions between the states.
            The actual states are stored in the state manager, and you can access them by passing the node index in the graph.
         */
        public:
            RRG(BaseSpacePtr space);
            ~RRG() override;

            void setMotionTask(BaseTaskPtr task, bool get_full_path=true) override;

            void solve(BaseTerminationPtr termination_condition) override;
            
            /**
                @brief Get the state manager which stores the states of the graph.
             */
            BaseStateManagerPtr getStateManager()
            {
                return state_manager;
            }
            
            /**
                @brief Get the start and goal group states.
                @param start_group_states The start group states as output.
                @param goal_group_states The goal group states as output.
             */
            void getStartAndGoalGroupStates(
                BaseStatesPtr & start_group_states,
                BaseStatesPtr & goal_group_states
            );

            /**
                @brief Set the maximum travel distance.
                @param max_travel_distance The maximum travel distance.
             */
            void setMaxTravelDistance(float max_travel_distance)
            {
                max_travel_distance_ = max_travel_distance;
            }

            /**
                @brief Set the sample attempts in each iteration.
                @param sample_attempts_in_each_iteration The number of sampled state in each iteration.
             */
            void setSampleAttemptsInEachIteration(int sample_attempts_in_each_iteration)
            {
                sample_attempts_in_each_iteration_ = sample_attempts_in_each_iteration;
            }
        private:
            BaseStateManagerPtr state_manager;

            // parameters
            int sample_attempts_in_each_iteration_; // the number of sampled state in each iteration
            float max_travel_distance_; // the maximum distance to expand the graph.

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

            // motion task.
            BaseTaskPtr task_;

            /**
                @brief Get start and goal indexs in the state manager.
                @param start_group_indexs The indexs of the start group as output.
                @param goal_group_indexs The indexs of the goal group as output.
             */
            void getStartAndGoalGroupIndexs(
                std::vector<int> & start_group_indexs,
                std::vector<int> & goal_group_indexs
            );

            /**
                @brief Get all conbinations of two indices from two groups. This is used
                to check the connection in the graph between each pair of start and goal states.
                @param start_group_indexs The indexs of the start group.
                @param goal_group_indexs The indexs of the goal group.
                @param left_index_of_pair The left index of the pair.
                @param right_index_of_pair The right index of the pair.
                @return True if the function is successful, otherwise false.
             */
            bool getAllCombinations(
                const std::vector<int> & start_group_indexs,
                const std::vector<int> & goal_group_indexs,
                std::vector<int> & left_index_of_pair,
                std::vector<int> & right_index_of_pair
            );
    };

    typedef std::shared_ptr<RRG> RRGPtr;
} // namespace CUDAMPLibs