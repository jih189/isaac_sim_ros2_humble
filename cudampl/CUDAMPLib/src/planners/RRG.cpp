#include <planners/RRG.h>

namespace CUDAMPLib
{
    // Constructor
    RRG::RRG(BaseSpacePtr space)
        : BasePlanner(space)
    {
        // generate the graph based on the space
        graph = space->createGraph();

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

        // get goal states
        auto goal_states = task->getGoalStates(space_);
        goal_states->update();
        space_->checkStates(goal_states);
        graph->add_goal_states(goal_states);
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

        auto new_states = space_->sample(2);

        new_states->update();
        // evaluate the feasibility of the states
        state_feasibility.clear();
        space_->checkStates(new_states, state_feasibility);

        // filter out the infeasible states
        new_states->filterStates(state_feasibility);

        // print graph
        printf("Graph======================\n");
        graph->print();

        printf("States======================\n");
        new_states->print();

        // find the motions to k nearest neighbor
        std::vector<StateIndexPair> connect_pairs;
        graph->get_motions_to_k_nearest_neighbors(new_states, k, connect_pairs);

        // // add the states to the graph
        // graph->add_states(states);

        // print the graph
        // graph->print();
    }
} // namespace CUDAMPLib