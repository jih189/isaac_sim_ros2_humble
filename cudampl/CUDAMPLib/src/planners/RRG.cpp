#include <planners/RRG.h>
#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    // Constructor
    RRG::RRG(BaseSpacePtr space)
        : BasePlanner(space)
    {
        state_manager = space->createStateManager();

        // set the parameters
        sample_attempts = 3;
        k = 3;
    }

    // Destructor
    RRG::~RRG()
    {
        // Cleanup code here, if needed
    }

    // Set the motion task
    void RRG::setMotionTask(BaseTaskPtr task)
    {
        // get start states
        auto start_states = task->getStartStates(space_);
        start_states->update();
        space_->checkStates(start_states);
        // add start states to the state manager
        std::vector<int> start_node_indexs = state_manager->add_states(start_states);

        // reset the graph and add start nodes and goal nodes
        graph.clear();
        start_nodes.clear();
        goal_nodes.clear();

        // create a start node
        start_node = boost::add_vertex(graph);
        graph[start_node].group = 1; // Start group
        graph[start_node].index_in_manager = -1;

        // Add start states to the graph
        for (size_t i = 0; i < start_node_indexs.size(); i++)
        {
            // Add a vertex to the graph
            BoostVertex v = boost::add_vertex(graph);
            graph[v].index_in_manager = start_node_indexs[i];
            graph[v].group = 1; // Start group
            start_nodes.push_back((int)v);

            // add an edge between the start node and the start state with zero weight
            boost::add_edge(v, start_node, EdgeProperties{0.0f}, graph);
        }

        // get goal states
        auto goal_states = task->getGoalStates(space_);
        goal_states->update();
        space_->checkStates(goal_states);
        // add goal states to the state manager
        std::vector<int> goal_node_indexs = state_manager->add_states(goal_states);

        // create a goal node
        goal_node = boost::add_vertex(graph);
        graph[goal_node].group = 2; // Goal group
        graph[goal_node].index_in_manager = -1;

        // Add goal states to the graph
        for (size_t i = 0; i < goal_node_indexs.size(); i++)
        {
            // Add a vertex to the graph
            BoostVertex v = boost::add_vertex(graph);
            graph[v].index_in_manager = goal_node_indexs[i];
            graph[v].group = 2; // Goal group
            goal_nodes.push_back((int)v);

            // add an edge between the goal node and the goal state with zero weight
            boost::add_edge(v, goal_node, EdgeProperties{0.0f}, graph);
        }
    }

    /**
        * @brief Solve the task.

        The pseudocode of the RRG algorithm in loop is as follows:
        1. Sample 10 configurations.
        2. Evaluate the feasibility of the states.
        3. Filter out the infeasible states.
        4. Find k nearest neighbors for each state.
        5. Generate a set of states pairs
        6. Check the feasibility of motion between the states pairs.
        7. Add the states to the graph.
     */
    void RRG::solve()
    {
        bool has_solution = false;

        //TODO: We should check if direct motion between start and goal states is feasible.

        // main loop
        for(int t = 0 ; t < 1; t++)
        {
            // sample states
            auto states = space_->sample(6);
            states->update();

            // evaluate the feasibility of the states
            std::vector<bool> state_feasibility;
            space_->checkStates(states, state_feasibility);

            // remove the infeasible states
            states->filterStates(state_feasibility);

            // check if the sampled states are all infeasible, then continue
            if (states->getNumOfStates() == 0)
            {
                continue;
            }

            // find k nearest neighbors for each state
            std::vector<std::vector<int>> neighbors_index;
            state_manager->find_k_nearest_neighbors(k, states, neighbors_index);

            // if k is larger than the number of states, we need to adjust k
            int actual_k = neighbors_index[0].size();

            // validate the motion from the sampled states to their neighbors.
            // prepare the motion states 1
            std::vector<BaseStatesPtr> states_list;
            for (int i = 0; i < actual_k; i++)
                states_list.push_back(states);

            auto motion_states_1 = state_manager->concatinate_states(states_list);

            // prepare the motion states 2
            std::vector<int> indexs_in_manager;
            for (int i = 0; i < actual_k; i++)
            {
                for (size_t j = 0; j < neighbors_index.size(); j++)
                {
                    indexs_in_manager.push_back(neighbors_index[j][i]);
                }
            }

            auto motion_states_2 = state_manager->get_states(indexs_in_manager);

            // calculate costs and check the feasibility of motion between the states pairs.
            std::vector<bool> motion_feasibility;
            std::vector<float> motion_costs;
            space_->checkMotions(motion_states_1, motion_states_2, motion_feasibility, motion_costs);

            /*
                Assume we have three sampled states and k = 2, and S is the sampled state, N is the neighbor of S.
                motion state 1:
                    S 1, S 2, S 3, S 1, S 2, S 3
                motion state 2:
                    N 1 of S 1, N 1 of S 2, N 1 of S 3, N 2 of S 1, N 2 of S 2, N 2 of S 3
            */

            // determine which sampled states can be added to the graph.
            std::vector<bool> can_connect(states->getNumOfStates(), false);
            std::vector<std::vector<int>> feasible_neighbors_indexs_of_added_states;
            std::vector<std::vector<float>> feasible_neighbors_costs_of_added_states;
            for(int i = 0; i < states->getNumOfStates(); i++)
            {
                std::vector<int> feasible_neighbors_indexs_of_added_states_i;
                std::vector<float> feasible_neighbors_costs_of_added_states_i;
                // check if this state is connected to any of its neighbors
                for(int j = 0; j < actual_k; j++)
                {
                    if(motion_feasibility[j * states->getNumOfStates() + i])
                    {
                        can_connect[i] = true;
                        feasible_neighbors_indexs_of_added_states_i.push_back(indexs_in_manager[j * states->getNumOfStates() + i]);
                        feasible_neighbors_costs_of_added_states_i.push_back(motion_costs[j * states->getNumOfStates() + i]);
                    }
                }

                if(can_connect[i])
                {
                    // if the state is connected to any of its neighbors, we add the indexs of its neighbors to the feasible_neighbors_indexs_of_added_states
                    feasible_neighbors_indexs_of_added_states.push_back(feasible_neighbors_indexs_of_added_states_i);
                    // if the state is connected to any of its neighbors, we add the costs of its neighbors to the feasible_neighbors_costs_of_added_states
                    feasible_neighbors_costs_of_added_states.push_back(feasible_neighbors_costs_of_added_states_i);
                }
            }

            // remove the states can not connect to any neighbors
            states->filterStates(can_connect);

            // add the states to the manager and get their indexs in the manager
            std::vector<int> indexs_of_new_state_in_manager = state_manager->add_states(states);

            if(indexs_of_new_state_in_manager.size() != feasible_neighbors_indexs_of_added_states.size())
            {
                throw std::runtime_error("indexs_of_new_state_in_manager.size() != neighbors_index_actual.size()");
            }

            // add the states to the graph
            for(size_t i = 0; i < indexs_of_new_state_in_manager.size(); i++)
            {
                // printf("Create node %d\n", indexs_of_new_state_in_manager[i]);
                // Add a vertex to the graph
                BoostVertex v = boost::add_vertex(graph);
                graph[v].index_in_manager = indexs_of_new_state_in_manager[i];
                graph[v].group = 0; // Normal group

                bool has_connect_to_start = false;
                bool has_connect_to_goal = false;
                // add edges between the new state and its neighbors
                for(size_t j = 0; j < feasible_neighbors_indexs_of_added_states[i].size(); j++)
                {
                    // printf("connect node %d and node %d\n", indexs_of_new_state_in_manager[i], feasible_neighbors_indexs_of_added_states[i][j]);
                    // find the vertex in the graph where its index in the state manager is feasible_neighbors_indexs_of_added_states[i][j]
                    for(auto v2 : boost::make_iterator_range(boost::vertices(graph)))
                    {
                        if(graph[v2].index_in_manager == feasible_neighbors_indexs_of_added_states[i][j])
                        {
                            if (graph[v2].group == 1)
                            {
                                has_connect_to_start = true;
                            }else if (graph[v2].group == 2)
                            {
                                has_connect_to_goal = true;
                            }
                            else
                            {
                                // print in red color
                                printf("\033[1;31m Something is wrong \033[0m \n");
                            }

                            // add an edge between v and v2 with the cost motion_costs[i * actual_k + j]
                            boost::add_edge(v, v2, EdgeProperties{feasible_neighbors_costs_of_added_states[i][j]}, graph);
                            break;
                        }
                    }
                }

                if(has_connect_to_start && ! has_connect_to_goal)
                {
                    graph[v].group = 1;
                }
                else if(! has_connect_to_start && has_connect_to_goal)
                {
                    graph[v].group = 2;
                }
                else if(has_connect_to_start && has_connect_to_goal)
                {
                    // has path from start to goal
                    has_solution = true;
                }
                else
                {
                    // error
                    throw std::runtime_error("Error: has_connect_to_start && has_connect_to_goal are both false");
                }
            }

            if(has_solution)
                break;
        }

        if(has_solution)
        {
            printf("Find a solution\n");

            const auto num_vertices = boost::num_vertices(graph);
            std::vector<float> distances(num_vertices, std::numeric_limits<float>::max());
            std::vector<BoostVertex> predecessors(num_vertices, boost::graph_traits<BoostGraph>::null_vertex());

            auto indexMap = boost::get(boost::vertex_index, graph);
            // Get the weight map explicitly from the bundled edge property.
            auto weight_map = boost::get(&EdgeProperties::weight, graph);

            // Run Dijkstra's algorithm from the source.
            boost::dijkstra_shortest_paths(graph, start_node,
                boost::weight_map(weight_map)
                .distance_map(boost::make_iterator_property_map(distances.begin(), indexMap))
                .predecessor_map(boost::make_iterator_property_map(predecessors.begin(), indexMap))
            );

            // Extract the shortest path to the goal node.
            std::vector<BoostVertex> path;
            for (BoostVertex v = goal_node; v != start_node; v = predecessors[v])
            {
                path.push_back(v);
            }

            path.push_back(start_node);

            // Print the shortest path.
            printf("Shortest path from start to goal:\n");
            for (auto it = path.rbegin(); it != path.rend(); ++it)
            {
                printf("%d ", graph[*it].index_in_manager);
            }
            printf("\n");

            // // print all weights of edges in the graph for debugging
            // for (auto e : boost::make_iterator_range(boost::edges(graph)))
            // {
            //     printf("Edge (%d, %d) has weight: %f\n", graph[boost::source(e, graph)].index_in_manager, graph[boost::target(e, graph)].index_in_manager, graph[e].weight);
            //     if(graph[e].weight > 0.0)
            //     {
            //         // print the joint values of both states
            //         auto debug_states = state_manager->get_states(
            //             {graph[boost::source(e, graph)].index_in_manager, graph[boost::target(e, graph)].index_in_manager});
                    
            //         // static cast the states to SingleArmStatesPtr
            //         SingleArmStatesPtr debug_states_single_arm = std::static_pointer_cast<SingleArmStates>(debug_states);
            //         std::vector<std::vector<float>> debug_states_joint_values = debug_states_single_arm->getJointStatesHost();

            //         // calculate the distance between the two states
            //         float distance = 0.0;
            //         for (size_t i = 0; i < debug_states_joint_values[0].size(); i++)
            //         {
            //             distance += std::pow(debug_states_joint_values[0][i] - debug_states_joint_values[1][i], 2);
            //         }
            //         distance = std::sqrt(distance);
            //         printf("Distance between the two states: %f\n", distance);

            //         if (distance != graph[e].weight)
            //         {
            //             // print in red color
            //             printf("\033[1;31m Error: distance != graph[e].weight \033[0m \n");
            //         }
            //     }
            // }

        }
        else
        {
            printf("No solution\n");
        }
    }
} // namespace CUDAMPLib