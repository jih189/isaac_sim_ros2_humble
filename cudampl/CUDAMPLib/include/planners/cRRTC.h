#pragma once

#include <base/Planner.h>
#include <NvrtcUtil.h>

namespace CUDAMPLib
{
    class cRRTC : public BasePlanner
    {
        /**
            @brief Rapidly-exploring Random Graph (RRG) planner.
            The graph is constructed by adding vertices and edges. The vertices are the states and the edges are the motions between the states.
            The actual states are stored in the state manager, and you can access them by passing the node index in the graph.
         */
        public:
            cRRTC(BaseSpacePtr space);
            ~cRRTC() override;

            void setMotionTask(BaseTaskPtr task, bool get_full_path=true) override;
            void solve(BaseTerminationPtr termination_condition) override;
        private:

            std::vector<std::vector<float>> backtraceTree(const std::vector<float>& tree_configurations, const std::vector<int>& tree_parent_indexs, int dim, int start_index);
            std::vector<std::vector<float>> constructFinalPath(int dim,
                const std::vector<float>& start_tree_configurations,
                const std::vector<int>& start_tree_parent_indexs,
                const std::vector<float>& goal_tree_configurations,
                const std::vector<int>& goal_tree_parent_indexs,
                int connection_index_start, // index in the start tree where connection occurred
                int connection_index_goal);  // index in the goal tree where connection occurred

            std::vector<std::vector<float>> start_states_vector_;
            std::vector<std::vector<float>> goal_states_vector_;

            // motion task.
            BaseTaskPtr task_;

            int max_interations_;
            int dim_;
            int num_of_threads_per_motion_;
            int num_of_thread_blocks_;
            float step_resolution_;
            int max_step_;
            std::string forward_kinematics_kernel_source_code_;
            std::string robot_collision_model_kernel_source_code_;
            std::string constraint_functions_kernel_source_code_;
            std::string launch_check_constraint_kernel_source_code_;

            // device memory
            float* d_start_tree_configurations_;
            int* d_start_tree_parent_indexs_;
            float* d_goal_tree_configurations_;
            int* d_goal_tree_parent_indexs_;
            int * connected_tree_node_pair_;

            float * d_sampled_configurations_;

            KernelFunctionPtr cRRTCKernelPtr_;

            std::string generateSourceCode();
    };

    typedef std::shared_ptr<cRRTC> cRRTCPtr;
} // namespace CUDAMPLibs