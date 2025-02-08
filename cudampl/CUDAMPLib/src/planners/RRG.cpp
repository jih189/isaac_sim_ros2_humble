#include <planners/RRG.h>

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
        state_manager->add_states(start_states);

        // get goal states
        auto goal_states = task->getGoalStates(space_);
        goal_states->update();
        space_->checkStates(goal_states);
        // add goal states to the state manager
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
        auto states = space_->sample(9);
        states->update();
        // evaluate the feasibility of the states
        std::vector<bool> state_feasibility;
        space_->checkStates(states, state_feasibility);

        // filter out the infeasible states
        states->filterStates(state_feasibility);

        state_manager->add_states(states);

        auto new_states = space_->sample(2);

        std::vector<std::vector<int>> neighbors_index;
        state_manager->find_k_nearest_neighbors(k, new_states, neighbors_index);

        // prepare the motion states 1
        std::vector<BaseStatesPtr> states_list;
        for (int i = 0; i < k; i++)
        {
            states_list.push_back(new_states);
        }
        auto motion_states_1 = state_manager->concatinate_states(states_list);

        // prepare the motion states 2
        std::vector<int> states_2_index;
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < neighbors_index.size(); j++)
            {
                states_2_index.push_back(neighbors_index[j][i]);
            }
        }
        auto motion_states_2 = state_manager->get_states(states_2_index);

        // printf("motion_states_1:\n");
        // motion_states_1->print();

        // printf("motion_states_2:\n");
        // motion_states_2->print();

        // check the feasibility of motion between the states pairs.
        std::vector<bool> motion_feasibility;
        std::vector<float> motion_costs;
        space_->checkMotions(motion_states_1, motion_states_2, motion_feasibility, motion_costs);

        // print the motion feasibility and costs
        for (int i = 0; i < motion_feasibility.size(); i++)
        {
            printf("motion %d: feasibility = %s, cost = %f\n", 
            i, 
            motion_feasibility[i] ? "True" : "False", 
            motion_costs[i]);
        }
    }
} // namespace CUDAMPLib