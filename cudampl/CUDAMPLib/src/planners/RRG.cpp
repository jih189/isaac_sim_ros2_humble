#include <planners/RRG.h>

namespace CUDAMPLib
{
    // Constructor
    RRG::RRG(BaseSpacePtr space)
        : BasePlanner(space)
    {
        // generate the graph based on the space
        graph = space->createGraph();
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
        graph->add_start_states(start_states);

        // get goal states
        auto goal_states = task->getGoalStates(space_);
        graph->add_goal_states(goal_states);
    }

    // Solve method
    void RRG::solve()
    {
        // Implement the logic for solving the planning problem
        // Example: Expand the graph, evaluate paths, etc.
    }
} // namespace CUDAMPLib