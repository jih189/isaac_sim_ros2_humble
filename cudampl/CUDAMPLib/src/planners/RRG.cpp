#include <planners/RRG.h>

// include for time
#include <chrono>

namespace CUDAMPLib
{
    // Constructor
    RRG::RRG(BaseSpacePtr space)
        : BasePlanner(space)
    {
        state_manager = space->createStateManager();

        // set the parameters as default values
        sample_attempts_in_each_iteration_ = 30;
        max_travel_distance_ = 0.5;
    }

    // Destructor
    RRG::~RRG()
    {
        // clear the state manager by calling destructor
        state_manager.reset();

        // clear the graph
        graph.clear();
    }

    // Set the motion task
    void RRG::setMotionTask(BaseTaskPtr task, bool get_full_path)
    {
        // set the get full path flag
        get_full_path_ = get_full_path;

        // get start states
        auto start_states = task->getStartStates(space_);
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

        task_ = task;

        // clear start and goal states
        start_states.reset();
        goal_states.reset();
    }

    bool RRG::getAllCombinations(
        const std::vector<int> & start_group_indexs,
        const std::vector<int> & goal_group_indexs,
        std::vector<int> & left_index_of_pair,
        std::vector<int> & right_index_of_pair
    )
    {
        if (start_group_indexs.size() == 0 || goal_group_indexs.size() == 0)
        {
            // print in red color
            printf("\033[1;31m start_group_indexs or goal_group_indexs is empty \033[0m \n");
            return false;
        }

        // initialize the left and right index of the pair with size. The size of the left and right 
        // index of the pair is the product of the size of the start and goal group indexs.
        left_index_of_pair.clear();
        right_index_of_pair.clear();

        for (size_t i = 0; i < start_group_indexs.size(); i++)
        {
            for (size_t j = 0; j < goal_group_indexs.size(); j++)
            {
                left_index_of_pair.push_back(start_group_indexs[i]);
                right_index_of_pair.push_back(goal_group_indexs[j]);
            }
        }
        return true;
    }

    void RRG::getStartAndGoalGroupIndexs(
        std::vector<int> & start_group_indexs,
        std::vector<int> & goal_group_indexs
    )
    {
        for(auto v : boost::make_iterator_range(boost::vertices(graph)))
        {
            if(graph[v].group == 1 && graph[v].index_in_manager != -1)
            {
                start_group_indexs.push_back(graph[v].index_in_manager);
            }
            else if(graph[v].group == 2 && graph[v].index_in_manager != -1)
            {
                goal_group_indexs.push_back(graph[v].index_in_manager);
            }
        }
    }

    void RRG::solve(BaseTerminationPtr termination_condition)
    {
        bool has_solution = false;

        // need to check if start and goal states are feasible
        std::vector<int> start_state_indexs_in_manager;
        for (auto start_node : start_nodes)
        {
            start_state_indexs_in_manager.push_back(graph[start_node].index_in_manager);
        }
        std::vector<int> goal_state_indexs_in_manager;
        for (auto goal_node : goal_nodes)
        {
            goal_state_indexs_in_manager.push_back(graph[goal_node].index_in_manager);
        }

        auto start_states_in_cuda = state_manager->get_states(start_state_indexs_in_manager);
        auto goal_states_in_cuda = state_manager->get_states(goal_state_indexs_in_manager);

        start_states_in_cuda->update();
        goal_states_in_cuda->update();

        std::vector<bool> start_state_feasibility;
        space_->checkStates(start_states_in_cuda, start_state_feasibility);
        std::vector<bool> goal_state_feasibility;
        space_->checkStates(goal_states_in_cuda, goal_state_feasibility);

        // deallocate the start and goal states in cuda
        start_states_in_cuda.reset();
        goal_states_in_cuda.reset();

        // check if any start state and goal states is feasible
        if(std::find(start_state_feasibility.begin(), start_state_feasibility.end(), true) == start_state_feasibility.end()
            || std::find(goal_state_feasibility.begin(), goal_state_feasibility.end(), true) == goal_state_feasibility.end())
        {
            // print in red color
            // printf("\033[1;31m No feasible start state or goal state \033[0m \n");
            task_->setFailureReason("InvalidInput");
            return;
        }

        // check if any pair of start and goal states is connected.
        std::vector<int> left_index_of_pair;
        std::vector<int> right_index_of_pair;
        if(! getAllCombinations(start_state_indexs_in_manager, goal_state_indexs_in_manager, left_index_of_pair, right_index_of_pair))
        {
            task_->setFailureReason("InvalidInput");
            return;
        }
        
        auto states_1_in_cuda = state_manager->get_states(left_index_of_pair);
        auto states_2_in_cuda = state_manager->get_states(right_index_of_pair);

        // check motions
        std::vector<bool> init_motion_feasibility;
        std::vector<float> init_motion_costs;
        space_->checkMotions(states_1_in_cuda, states_2_in_cuda, init_motion_feasibility, init_motion_costs);

        // deallocate the states in cuda
        states_1_in_cuda.reset();
        states_2_in_cuda.reset();

        int feasible_start_index = -1;
        int feasible_goal_index = -1;
        float current_cost = std::numeric_limits<float>::max();

        for(size_t i = 0; i < init_motion_feasibility.size(); i++)
        {
            if(init_motion_feasibility[i])
            {
                has_solution = true;
                if (init_motion_costs[i] < current_cost)
                {
                    // try to return the motion with the minimum cost
                    current_cost = init_motion_costs[i];
                    feasible_start_index = left_index_of_pair[i];
                    feasible_goal_index = right_index_of_pair[i];
                }
            }
        }

        if (has_solution)
        {
            // there exists a solution between the start and goal states directly.
            auto waypoints = state_manager->get_states({feasible_start_index, feasible_goal_index});
            auto solution = space_->getPathFromWaypoints(waypoints);
            task_->setSolution(solution, space_);
            waypoints.reset();
            solution.reset();

            return;
        }

        // reset the termination condition
        termination_condition->reset();

        float start_group_max_travel_distance = max_travel_distance_;
        float goal_group_max_travel_distance = max_travel_distance_;
        
        // main loop
        for(int t = 0 ; t < 1000000; t++)
        {
            // check the termination condition
            if(termination_condition->checkTerminationCondition())
            {
                termination_condition->printTerminationReason();
                task_->setFailureReason("MeetTerminationCondition");
                break;
            }

            // sample states
            auto states = space_->sample(sample_attempts_in_each_iteration_);

            // get state indexes of both start and goal group in the graph.
            std::vector<int> start_group_indexs;
            std::vector<int> goal_group_indexs;
            getStartAndGoalGroupIndexs(start_group_indexs, goal_group_indexs);

            // find the nearest neighbors of the states
            std::vector<std::vector<int>> nearest_neighbors_index;
            if (t % 2 == 0)
            {
                // find the nearest neighbors of the states in the start group
                state_manager->find_the_nearest_neighbors(states, {start_group_indexs}, nearest_neighbors_index);
            }
            else
            {
                // find the nearest neighbors of the states in the goal group
                state_manager->find_the_nearest_neighbors(states, {goal_group_indexs}, nearest_neighbors_index);
            }

            // get the nearest state index for each sampled state
            std::vector<int> nearest_neighbors_index_for_each_sampled_state;
            for(auto i : nearest_neighbors_index)
            {
                nearest_neighbors_index_for_each_sampled_state.push_back(i[0]);
            }

            // constraint the sampled states to be within the max travel distance of their nearest neighbors
            if(t % 2 == 0)
            {
                // constraint the sampled states to be within the max travel distance of their nearest neighbors in the start group
                state_manager->interpolateToStates(states, nearest_neighbors_index_for_each_sampled_state, start_group_max_travel_distance);
            }
            else
            {
                // constraint the sampled states to be within the max travel distance of their nearest neighbors in the goal group
                state_manager->interpolateToStates(states, nearest_neighbors_index_for_each_sampled_state, goal_group_max_travel_distance);
            }

            // update the states
            states->update();

            // evaluate the feasibility of the sampled states
            std::vector<bool> state_feasibility;
            space_->checkStates(states, state_feasibility);

            // remove the infeasible sampled states
            states->filterStates(state_feasibility);

            // if the sampled states are all infeasible, then continue
            if (states->getNumOfStates() == 0)
            {
                // clear the states
                states.reset();
                continue;
            }

            // find the nearest neighbor from start and goal group for each state
            std::vector<std::vector<int>> neighbors_index;
            int group_number = 2; // for start and goal group, we only need 1 nearest neighbor for each group.
            state_manager->find_the_nearest_neighbors(states, {start_group_indexs, goal_group_indexs}, neighbors_index);

            // validate the motion from the sampled states to their neighbors.
            // prepare the motion states 1
            std::vector<BaseStatesPtr> states_list;
            for (int i = 0; i < group_number; i++)
                states_list.push_back(states);

            auto motion_states_1 = state_manager->concatinate_states(states_list);

            // prepare the motion states 2
            std::vector<int> indexs_in_manager;
            for (int i = 0; i < group_number; i++)
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

            // clear the states
            motion_states_1.reset();
            motion_states_2.reset();

            // determine which sampled states can be added to the graph.
            std::vector<bool> can_connect(states->getNumOfStates(), false); // indicate if the state can connect to any of its neighbors
            std::vector<std::vector<int>> feasible_neighbors_indexs_of_added_states; // the indexs of the neighbors of the each added states
            std::vector<std::vector<float>> feasible_neighbors_costs_of_added_states; // the costs of the neighbors of the each added states
            std::vector<int> num_of_connection_to_groups(group_number, 0); // the number of connections to each group
            for(int i = 0; i < states->getNumOfStates(); i++)
            {
                std::vector<int> feasible_neighbors_indexs_of_added_states_i;
                std::vector<float> feasible_neighbors_costs_of_added_states_i;
                // check if this state is connected to any of its neighbors
                for(int j = 0; j < group_number; j++)
                {
                    if(motion_feasibility[j * states->getNumOfStates() + i])
                    {
                        can_connect[i] = true;
                        feasible_neighbors_indexs_of_added_states_i.push_back(indexs_in_manager[j * states->getNumOfStates() + i]);
                        feasible_neighbors_costs_of_added_states_i.push_back(motion_costs[j * states->getNumOfStates() + i]);
                        num_of_connection_to_groups[j]++;
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
;
            if (t % 2 == 0)
            {
                if (num_of_connection_to_groups[0] == 0)
                {
                    // if there is no connection to the start group, we reduce the max travel distance for the next iteration
                    start_group_max_travel_distance = start_group_max_travel_distance * 0.5 > 0.1 ? start_group_max_travel_distance * 0.5 : 0.1;
                    t++;
                    continue;
                }
                else{
                    // if there is connection to the start group, we reset the max travel distance for the next iteration
                    start_group_max_travel_distance = max_travel_distance_;
                }
            }
            else
            {
                if (num_of_connection_to_groups[1] == 0)
                {
                    // if there is no connection to the goal group, we reduce the max travel distance for the next iteration
                    goal_group_max_travel_distance = goal_group_max_travel_distance * 0.5 > 0.1 ? goal_group_max_travel_distance * 0.5 : 0.1;
                    t++;
                    continue;
                }
                else{
                    // if there is connection to the goal group, we reset the max travel distance for the next iteration
                    goal_group_max_travel_distance = max_travel_distance_;
                }
            }


            // remove the states can not connect to any neighbors
            states->filterStates(can_connect);

            // add the states to the manager and get their indexs in the manager
            std::vector<int> indexs_of_new_state_in_manager = state_manager->add_states(states);

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

                            // add an edge between v and v2 with the cost motion_costs[i * group_number + j]
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

            // clear states
            states.reset();

            if(has_solution)
                break;
        }

        if(has_solution)
        {
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

            std::vector<BoostVertex> path;
            for (BoostVertex v = goal_node; v != start_node; v = predecessors[v])
            {
                path.push_back(v);
            }

            path.push_back(start_node);

            // Print the shortest path. 
            // printf("Shortest path from start to goal:\n");
            std::vector<int> path_indexs_in_manager;
            for (auto it = path.rbegin(); it != path.rend(); ++it)
            {
                if (graph[*it].index_in_manager != -1)
                {
                    path_indexs_in_manager.push_back(graph[*it].index_in_manager);
                    // printf("%d ", graph[*it].index_in_manager);
                }
            }
            // printf("\n");
            if(get_full_path_)
            {
                // get the full path
                auto waypoints = state_manager->get_states(path_indexs_in_manager);
                auto solution = space_->getPathFromWaypoints(waypoints);
                task_->setSolution(solution, space_);
                waypoints.reset();
                solution.reset();
            }
            else
            {
                // get the waypoints
                auto waypoints = state_manager->get_states(path_indexs_in_manager);
                task_->setSolution(waypoints, space_);
                waypoints.reset();
            }
            
        }
    }

    void RRG::getStartAndGoalGroupStates(
        BaseStatesPtr & start_group_states,
        BaseStatesPtr & goal_group_states
    )
    {
        std::vector<int> start_group_indexs;
        std::vector<int> goal_group_indexs;
        getStartAndGoalGroupIndexs(start_group_indexs, goal_group_indexs);

        start_group_states = state_manager->get_states(start_group_indexs);
        goal_group_states = state_manager->get_states(goal_group_indexs);
    }
} // namespace CUDAMPLib