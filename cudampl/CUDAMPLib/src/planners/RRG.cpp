#include <planners/RRG.h>

namespace CUDAMPLib
{
    // Constructor
    RRG::RRG(BaseSpacePtr space)
        : BasePlanner(space)
    {
        // generate the graph based on the space
        graph = space->createGraph();

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
        graph->add_start_states(start_states);

        state_manager->add_states(start_states);

        // get goal states
        auto goal_states = task->getGoalStates(space_);
        goal_states->update();
        space_->checkStates(goal_states);
        graph->add_goal_states(goal_states);

        state_manager->add_states(goal_states);
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
        // sample k configurations
        auto states = space_->sample(7);
        states->update();
        // evaluate the feasibility of the states
        std::vector<bool> state_feasibility;
        space_->checkStates(states, state_feasibility);

        // filter out the infeasible states
        states->filterStates(state_feasibility);

        graph->add_states(states);

        state_manager->add_states(states);

        auto new_states = space_->sample(2);

        new_states->update();
        // evaluate the feasibility of the states
        state_feasibility.clear();
        space_->checkStates(new_states, state_feasibility);

        // filter out the infeasible states
        new_states->filterStates(state_feasibility);

        // find the motions to k nearest neighbor
        std::vector<StateIndexPair> connect_pairs;
        auto possible_motions = graph->get_motions_to_k_nearest_neighbors(new_states, k, connect_pairs);

        std::vector<bool> motion_feasibility;
        std::vector<float> motion_costs;
        space_->checkMotions(possible_motions, motion_feasibility, motion_costs);

        // print motion pairs
        for (const auto &pair : connect_pairs)
        {
            printf("pair: %d %d\n", pair.index_in_states, pair.index_in_graph);
        }

        // print the motion feasibility
        for (bool feasible : motion_feasibility)
        {
            printf("feasibility %d ", feasible);
        }
        printf("\n");

        // print the motion costs
        for (float cost : motion_costs)
        {
            printf("cost %f ", cost);
        }
        printf("\n");

        std::vector<std::vector<int>> neighbors_index;

        state_manager->find_k_nearest_neighbors(k, new_states, neighbors_index);

        printf("print result from state manager\n");

        std::vector<BaseStatesPtr> all_neighbor_states;

        // print the neighbors index
        for (const auto &index : neighbors_index)
        {
            for (int i : index)
            {
                printf("index of neighbor state %d \n", i);
            }

            auto neighbor_states = state_manager->get_states(index);
            neighbor_states->print();
            printf("\n");

            all_neighbor_states.push_back(neighbor_states);
        }

        
        auto sum_states = state_manager->concatinate_states(all_neighbor_states);
        printf("sum states\n");
        sum_states->print();

        // // add the states to the graph
        // graph->add_states(states);

        // print the graph
        // graph->print();
    }
} // namespace CUDAMPLib