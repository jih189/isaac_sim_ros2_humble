#include "RobotSolver.h"
#include <algorithm>  // for std::count

namespace CPRRTC
{

    RobotSolver::RobotSolver(
        std::string robot_name,
        size_t dim,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_parent_link_maps,
        const std::vector<int>& self_collision_spheres_to_link_map,
        const std::vector<Eigen::Vector3d>& self_collision_spheres_pos_in_link,
        const std::vector<float>& self_collision_spheres_radius,
        const std::vector<bool>& active_joint_map,
        const std::vector<float>& lower,
        const std::vector<float>& upper,
        const std::vector<float>& default_joint_values,
        const std::vector<std::string>& link_names,
        const std::vector<std::vector<bool>>& self_collision_enables_map,
        float resolution
    )
        : robot_name_(std::move(robot_name))
        , dim_(dim)
        , joint_types_(joint_types)
        , joint_poses_(joint_poses)
        , joint_axes_(joint_axes)
        , link_parent_link_maps_(link_parent_link_maps)
        , self_collision_spheres_to_link_map_(self_collision_spheres_to_link_map)
        , self_collision_spheres_pos_in_link_(self_collision_spheres_pos_in_link)
        , self_collision_spheres_radius_(self_collision_spheres_radius)
        , active_joint_map_(active_joint_map)
        , lower_bound_(lower)
        , upper_bound_(upper)
        , default_joint_values_(default_joint_values)
        , link_names_(link_names)
        , self_collision_enables_map_(self_collision_enables_map)
        , resolution_(resolution)
        , num_of_joints_(static_cast<int>(joint_types_.size()))
        , num_of_links_(static_cast<int>(link_names_.size()))
        , num_of_self_collision_spheres_(static_cast<int>(self_collision_spheres_radius_.size()))
        , num_of_active_joints_(static_cast<int>(std::count(active_joint_map_.begin(), active_joint_map_.end(), true)))
    {
        max_iterations_ = 100;
        num_of_threads_per_motion_ = max_step_ = 32;
        num_of_thread_blocks_ = 1;

        // Initialize self-collision check parameters
        self_collision_sphere_indices_1_.clear();
        self_collision_sphere_indices_2_.clear();
        self_collision_distance_thresholds_.clear();
        num_of_self_collision_check_ = 0;

        for (int i = 0; i < num_of_self_collision_spheres_; i++){
            for (int j = i + 1; j < num_of_self_collision_spheres_; j++){
                // check if the two spheres are not in the same link and self-collision is enabled between the two links
                if (self_collision_spheres_to_link_map_[i] != self_collision_spheres_to_link_map_[j] && self_collision_enables_map_[self_collision_spheres_to_link_map_[i]][self_collision_spheres_to_link_map_[j]]){
                    self_collision_sphere_indices_1_.push_back(i);
                    self_collision_sphere_indices_2_.push_back(j);
                    self_collision_distance_thresholds_.push_back((self_collision_spheres_radius_[i] + self_collision_spheres_radius_[j]) * (self_collision_spheres_radius_[i] + self_collision_spheres_radius_[j])); // squared distance threshold
                    num_of_self_collision_check_++;
                }
            }
        }

        std::string source_code = generateKernelSourceCode();

        // save the source code to a file
        std::string file_name = "/home/ros/ros2_ws/src/cRRTCKernel.cu";
        std::ofstream source_file(file_name);

        if (source_file.is_open())
        {
            // first clear the file
            source_file.clear();

            source_file << source_code;
            source_file.close();
        }
        else
        {
            std::cerr << "Unable to open file: " << file_name << std::endl;
        }

        const char *source_code_c_str = source_code.c_str();

        cRRTCKernelPtr_ = KernelFunction::create(source_code_c_str, "CPRRTCKernel");

        if (! cRRTCKernelPtr_ || ! cRRTCKernelPtr_->function) {
            std::cerr << "\033[31m" << "Kernel function 'cRRTCKernel' compilation failed." << "\033[0m" << std::endl;
        }
    }

    RobotSolver::~RobotSolver()
    {
        // TODO: release CUDA resources, unload PTX, etc.
    }

    void RobotSolver::setEnvObstacleCache(int num_of_spheres, int num_of_cuboids, int num_of_cylinders)
    {
        // TODO: allocate or resize GPU buffers for spheres, cuboids, cylinders
    }

    void RobotSolver::updateEnvObstacle(
        std::vector<Sphere>& spheres,
        std::vector<Cuboid>& cuboids,
        std::vector<Cylinder>& cylinders
    )
    {
        // TODO: upload obstacle data to GPU
    }

    std::vector<std::vector<float>> RobotSolver::solve(
        std::vector<float>& start,
        std::vector<float>& goal
    )
    {
        // TODO:
        //  1. Set kernel arguments (start, goal, joint limits, etc.)
        //  2. Launch NVRTC‐compiled kernel
        //  3. Retrieve and decode result path
        //  4. Return as vector of joint‐value sequences
        return {};
    }

    std::string RobotSolver::generateKernelSourceCode()
    {
        std::string kernel_code;
        kernel_code += R"(
#ifndef FLT_MAX
#define FLT_MAX __int_as_float(0x7f7fffff)    // 3.40282347e+38f
#endif

constexpr float UNWRITTEN_VAL = -9999.0f;

extern "C" {
    __device__ int startTreeCounter = 0;
    __device__ int goalTreeCounter = 0;
    __device__ int sampledCounter = 0;
    __device__ int foundSolution = 0;
}
)";

        kernel_code += generateFKKernelSourceCode();
        kernel_code += generateSelfCollisionCheckSourceCode();

        kernel_code += "__device__ __forceinline__ bool check_partially_written(float *node) {\n";
        kernel_code += "    #pragma unroll\n";
        kernel_code += "    for (int i = 0; i < " + std::to_string(dim_) + "; i++) {\n";
        kernel_code += "        if (node[i] == UNWRITTEN_VAL) return true;\n";
        kernel_code += "    }\n";
        kernel_code += "    return false;\n";
        kernel_code += "}\n";

        kernel_code += "extern \"C\" __global__ void CPRRTCKernel(float* d_start_tree_configurations, float* d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations, int * connected_tree_node_pair, int num_of_sphere_obstacles, float * d_sphere_obstacles, int num_of_cuboid_obstacles, float * d_cuboid_obstacles, int num_of_cylinder_obstacles, float * d_cylinder_obstacles){\n";
        kernel_code += "    __shared__ float * target_tree;\n";
        kernel_code += "    __shared__ int * target_tree_counter;\n";
        kernel_code += "    __shared__ int * target_tree_parent_indexs;\n";
        kernel_code += "    __shared__ float * other_tree;\n";
        kernel_code += "    __shared__ int * other_tree_counter;\n";
        kernel_code += "    __shared__ int localSampledCounter;\n";
        kernel_code += "    __shared__ float partial_distance_cost_from_nn[" + std::to_string(num_of_threads_per_motion_) + "];\n";
        kernel_code += "    __shared__ int partial_nn_index[" + std::to_string(num_of_threads_per_motion_) + "];\n";
        kernel_code += "    __shared__ float local_to_configuration[" + std::to_string(dim_) + "];\n";
        kernel_code += "    __shared__ float local_from_configuration[" + std::to_string(dim_) + "];\n";
        kernel_code += "    __shared__ float local_delta_motion[" + std::to_string(dim_) + "];\n";
        kernel_code += "    __shared__ int local_parent_index;\n";
        kernel_code += "    __shared__ float local_nearest_neighbor_distance;\n";
        kernel_code += "    __shared__ float local_motion_configurations[" + std::to_string(dim_ * max_step_) + "]; \n";
        kernel_code += "    __shared__ int motion_step;\n";
        kernel_code += "    __shared__ bool should_skip;\n";
        kernel_code += "    __shared__ int new_node_index;\n";
        kernel_code += "    __shared__ int connected_node_in_target_tree;\n";
        kernel_code += "    __shared__ int connected_node_in_other_tree;\n";
        kernel_code += "    __shared__ int connected_index_in_other_tree;\n";
        kernel_code += "    const int tid = threadIdx.x;\n";
        kernel_code += "    float self_collision_spheres_pos_in_base[" + std::to_string(num_of_self_collision_spheres_ * 3) + "];\n\n";
        kernel_code += "    for (int i = 0; i < " + std::to_string(max_iterations_) + "; i++) {\n";
        kernel_code += "        // Need to decide which tree to grow\n";
        kernel_code += "        if (i == 0) {\n";
        kernel_code += "            should_skip = false;\n";
        kernel_code += "            localSampledCounter =  atomicAdd(&sampledCounter, 1);\n\n";
        kernel_code += "            if (startTreeCounter > goalTreeCounter) {\n";
        kernel_code += "                target_tree = d_goal_tree_configurations;\n";
        kernel_code += "                target_tree_counter = &goalTreeCounter;\n";
        kernel_code += "                target_tree_parent_indexs = d_goal_tree_parent_indexs;\n";
        kernel_code += "                other_tree = d_start_tree_configurations;\n";
        kernel_code += "                other_tree_counter = &startTreeCounter;\n";
        kernel_code += "                connected_node_in_target_tree = blockIdx.x * 2 + 1;\n";
        kernel_code += "                connected_node_in_other_tree = blockIdx.x * 2;\n";
        kernel_code += "            }\n";
        kernel_code += "            else {\n";
        kernel_code += "                target_tree = d_start_tree_configurations;\n";
        kernel_code += "                target_tree_counter = &startTreeCounter;\n";
        kernel_code += "                target_tree_parent_indexs = d_start_tree_parent_indexs;\n";
        kernel_code += "                other_tree = d_goal_tree_configurations;\n";
        kernel_code += "                other_tree_counter = &goalTreeCounter;\n";
        kernel_code += "                connected_node_in_target_tree = blockIdx.x * 2;\n";
        kernel_code += "                connected_node_in_other_tree = blockIdx.x * 2 + 1;\n";
        kernel_code += "            }\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // If meet the max iterations, skip the rest of the iterations\n";
        kernel_code += "        if (localSampledCounter >= " + std::to_string(max_iterations_) + ")\n";
        kernel_code += "            return;\n";
        kernel_code += "        // Sample a random configuration by loading it from global memory\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            local_to_configuration[tid] = d_sampled_configurations[localSampledCounter * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Find the nearest neighbor in the target tree\n";
        kernel_code += "        float best_distance = FLT_MAX;\n";
        kernel_code += "        int best_index = -1;\n";
        kernel_code += "        for (int i = tid; i < *target_tree_counter; i += blockDim.x) {\n";
        kernel_code += "            if (check_partially_written(&target_tree[i * " + std::to_string(dim_) + "])) break;\n";
        kernel_code += "            float distance = 0.0f;\n";
        kernel_code += "            float difference = 0.0f;\n";
        kernel_code += "            #pragma unroll\n";
        kernel_code += "            for (int j = 0; j < " + std::to_string(dim_) + "; j++) {\n";
        kernel_code += "                difference = local_to_configuration[j] - target_tree[i * " + std::to_string(dim_) + " + j];\n";
        kernel_code += "                distance += difference * difference;\n";
        kernel_code += "            }\n";
        kernel_code += "            if (distance < best_distance) {\n";
        kernel_code += "                best_distance = distance;\n";
        kernel_code += "                best_index = i;\n";
        kernel_code += "            }\n";
        kernel_code += "        }\n";
        kernel_code += "        // Write the local best distance and index to shared memory\n";
        kernel_code += "        partial_distance_cost_from_nn[tid] = best_distance;\n";
        kernel_code += "        partial_nn_index[tid] = best_index;\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Perform reduction to find the best distance and index\n";
        kernel_code += "        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {\n";
        kernel_code += "            if (tid < stride) {\n";
        kernel_code += "                if (partial_distance_cost_from_nn[tid] > partial_distance_cost_from_nn[tid + stride]) {\n";
        kernel_code += "                    partial_distance_cost_from_nn[tid] = partial_distance_cost_from_nn[tid + stride];\n";
        kernel_code += "                    partial_nn_index[tid] = partial_nn_index[tid + stride];\n";
        kernel_code += "                }\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n";
        kernel_code += "        }\n\n";
        kernel_code += "        // Write the best distance and index to shared memory\n";
        kernel_code += "        if (tid == 0) {\n";
        kernel_code += "            local_nearest_neighbor_distance = sqrtf(partial_distance_cost_from_nn[0]);\n";
        kernel_code += "            local_parent_index = partial_nn_index[0];\n";
        kernel_code += "            motion_step = min((int)(local_nearest_neighbor_distance / " + std::to_string(resolution_) + "), " + std::to_string(max_step_) + ");\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Calculate the delta motion from the nearest neighbor in the target tree to sampled configuration\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            local_from_configuration[tid] = target_tree[local_parent_index * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "            local_delta_motion[tid] = (local_to_configuration[tid] - local_from_configuration[tid]) / local_nearest_neighbor_distance * " + std::to_string(resolution_) + ";\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Interpolate the new configuration between the nearest neighbor in the target tree and sampled configuration\n";
        kernel_code += "        for (int j = tid; j < " + std::to_string(dim_) + " * motion_step; j += blockDim.x) {\n";
        kernel_code += "            int state_ind_in_motion = j / " + std::to_string(dim_) + ";\n";
        kernel_code += "            int joint_ind_in_state = j % " + std::to_string(dim_) + ";\n";
        kernel_code += "            local_motion_configurations[j] = local_from_configuration[joint_ind_in_state] + local_delta_motion[joint_ind_in_state] * state_ind_in_motion;\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Call forward kinematics function to calculate self-collision spheres positions in base frame\n";
        kernel_code += "        if (tid < motion_step) {\n";
        kernel_code += "            kin_forward(&(local_motion_configurations[tid]), self_collision_spheres_pos_in_base);\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Check for self-collision\n";
        kernel_code += "        if (tid < motion_step) {\n";
        kernel_code += "            should_skip = checkSelfCollisionConstraint(self_collision_spheres_pos_in_base);\n";
        kernel_code += "        }\n";
        // kernel_code += "        __syncthreads();\n\n";
        // kernel_code += "        if (! should_skip) {\n";
        // kernel_code += "            // Check for collision with environment obstacles\n";
        // kernel_code += "            // TODO: implement collision check with environment obstacles\n";
        // kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        if (should_skip) {\n";
        kernel_code += "            continue;\n";
        kernel_code += "        }\n\n";
        kernel_code += "        __syncthreads();\n";
        kernel_code += "        // Add the new node to the target tree\n";
        kernel_code += "        if (tid == 0) {\n";
        kernel_code += "            new_node_index = atomicAdd(target_tree_counter, 1);\n";
        kernel_code += "            target_tree_parent_indexs[new_node_index] = local_parent_index;\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Write the new node to the target tree\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            target_tree[new_node_index * " + std::to_string(dim_) + " + tid] = local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "            local_from_configuration[tid] = local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // find the nearest neighbor in the other tree\n";
        kernel_code += "        best_distance = FLT_MAX;\n";
        kernel_code += "        best_index = -1;\n";
        kernel_code += "        for (int i = tid; i < *other_tree_counter; i += blockDim.x) {\n";
        kernel_code += "            if (check_partially_written(&other_tree[i * " + std::to_string(dim_) + "])) break;\n";
        kernel_code += "            float distance = 0.0f;\n";
        kernel_code += "            float difference = 0.0f;\n";
        kernel_code += "            #pragma unroll\n";
        kernel_code += "            for (int j = 0; j < " + std::to_string(dim_) + "; j++) {\n";
        kernel_code += "                difference = other_tree[i * " + std::to_string(dim_) + " + j] - local_from_configuration[j];\n";
        kernel_code += "                distance += difference * difference;\n";
        kernel_code += "            }\n";
        kernel_code += "            if (distance < best_distance) {\n";
        kernel_code += "                best_distance = distance;\n";
        kernel_code += "                best_index = i;\n";
        kernel_code += "            }\n";
        kernel_code += "        }\n";
        kernel_code += "        // Write the local best distance and index to shared memory\n";
        kernel_code += "        partial_distance_cost_from_nn[tid] = best_distance;\n";
        kernel_code += "        partial_nn_index[tid] = best_index;\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Perform reduction to find the best distance and index\n";
        kernel_code += "        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {\n";
        kernel_code += "            if (tid < stride) {\n";
        kernel_code += "                if (partial_distance_cost_from_nn[tid] > partial_distance_cost_from_nn[tid + stride]) {\n";
        kernel_code += "                    partial_distance_cost_from_nn[tid] = partial_distance_cost_from_nn[tid + stride];\n";
        kernel_code += "                    partial_nn_index[tid] = partial_nn_index[tid + stride];\n";
        kernel_code += "                }\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n";
        kernel_code += "        }\n\n";
        kernel_code += "        // Write the best distance and index to shared memory\n";
        kernel_code += "        if (tid == 0) {\n";
        kernel_code += "            local_nearest_neighbor_distance = sqrtf(partial_distance_cost_from_nn[0]);\n";
        kernel_code += "            connected_index_in_other_tree = partial_nn_index[0];\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // get the configuration of the nearest neighbor in the other tree\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            local_to_configuration[tid] = other_tree[connected_index_in_other_tree * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        while (!should_skip) {\n";
        kernel_code += "            if (tid == 0) {\n";
        kernel_code += "                motion_step = min((int)(local_nearest_neighbor_distance / " + std::to_string(resolution_) + "), " + std::to_string(max_step_) + ");\n";
        kernel_code += "                local_parent_index = new_node_index;\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Calculate the delta motion from the new node to the nearest neighbor in the other tree\n";
        kernel_code += "            if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "                local_delta_motion[tid] = (local_to_configuration[tid] - local_from_configuration[tid]) / local_nearest_neighbor_distance * " + std::to_string(resolution_) + ";\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Calculate the intermediate configurations of next motion segment\n";
        kernel_code += "            for (int j = tid; j < " + std::to_string(dim_) + " * motion_step; j += blockDim.x) {\n";
        kernel_code += "                int state_ind_in_motion = j / " + std::to_string(dim_) + ";\n";
        kernel_code += "                int joint_ind_in_state = j % " + std::to_string(dim_) + ";\n";
        kernel_code += "                local_motion_configurations[j] = local_from_configuration[joint_ind_in_state] + local_delta_motion[joint_ind_in_state] * state_ind_in_motion;\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Call forward kinematics function to calculate self-collision spheres positions in base frame\n";
        kernel_code += "            if (tid < motion_step) {\n";
        kernel_code += "                kin_forward(&(local_motion_configurations[tid]), self_collision_spheres_pos_in_base);\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Check for self-collision\n";
        kernel_code += "            if (tid < motion_step) {\n";
        kernel_code += "                should_skip = checkSelfCollisionConstraint(self_collision_spheres_pos_in_base);\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            if (should_skip) {\n";
        kernel_code += "                break;\n";
        kernel_code += "            }\n\n";
        // kernel_code += "            // Check for collision with environment obstacles\n";
        // kernel_code += "            // TODO: implement collision check with environment obstacles\n";
        // kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Add the new node to the target tree\n";
        kernel_code += "            if (tid == 0) {\n";
        kernel_code += "                new_node_index = atomicAdd(target_tree_counter, 1);\n";
        kernel_code += "                target_tree_parent_indexs[new_node_index] = local_parent_index;\n";
        kernel_code += "                // Calculate the distance from the new node to the nearest configuration in the other tree\n";
        kernel_code += "                float squared_distance = 0.0f;\n";
        kernel_code += "                #pragma unroll\n";
        kernel_code += "                for (int j = 0; j < " + std::to_string(dim_) + "; j++) {\n";
        kernel_code += "                    float diff = local_to_configuration[j] - local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + j];\n";
        kernel_code += "                    squared_distance += diff * diff;\n";
        kernel_code += "                }\n";
        kernel_code += "                local_nearest_neighbor_distance = sqrtf(squared_distance);\n";
        kernel_code += "                // The new node is close enough to the nearest configuration in the other tree\n";
        kernel_code += "                if (local_nearest_neighbor_distance < " + std::to_string(resolution_ * 2) + ") {\n";
        kernel_code += "                    should_skip = true;\n";
        kernel_code += "                    connected_tree_node_pair[connected_node_in_target_tree] = new_node_index;\n";
        kernel_code += "                    connected_tree_node_pair[connected_node_in_other_tree] = connected_index_in_other_tree;\n";
        kernel_code += "                    foundSolution = 1;\n";
        kernel_code += "                }\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Write the new node to the target tree\n";
        kernel_code += "            if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "                target_tree[new_node_index * " + std::to_string(dim_) + " + tid] = local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "                local_from_configuration[tid] = local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "        }\n";
        kernel_code += "        // check if the connection is found\n";
        kernel_code += "        if (foundSolution != 0) {\n";
        kernel_code += "            return;\n";
        kernel_code += "        }\n";
        kernel_code += "    }\n";
        kernel_code += "};\n";
        return kernel_code; 
    }

    std::string RobotSolver::generateSelfCollisionCheckSourceCode()
    {
        std::string source_code;
        source_code += "__constant__ int self_collision_check_pairs[" + std::to_string(num_of_self_collision_check_) + "][2] = \n";
        source_code += "{\n";
        for (int i = 0; i < num_of_self_collision_check_; i++)
        {
            if (i == num_of_self_collision_check_ - 1)
                source_code += "{" + std::to_string(self_collision_sphere_indices_1_[i]) + ", " + std::to_string(self_collision_sphere_indices_2_[i]) + "}\n";
            else
                source_code += "{" + std::to_string(self_collision_sphere_indices_1_[i]) + ", " + std::to_string(self_collision_sphere_indices_2_[i]) + "}, ";
        }
        source_code += "};\n\n";
        source_code += "__constant__ float self_collision_distance_threshold[" + std::to_string(num_of_self_collision_check_) + "] = \n";
        source_code += "{\n";
        for (int i = 0; i < num_of_self_collision_check_; i++)
        {
            if (i == num_of_self_collision_check_ - 1)
                source_code += std::to_string(self_collision_distance_thresholds_[i]) + "\n";
            else
                source_code += std::to_string(self_collision_distance_thresholds_[i]) + ", ";
        }
        source_code += "};\n\n";
        source_code += "// SelfCollisionConstraint check function\n";
        source_code += "__device__ __forceinline__ bool checkSelfCollisionConstraint(float * self_collision_sphere_pos){\n";
        source_code += "    float dx = 0.0f;\n";
        source_code += "    float dy = 0.0f;\n";
        source_code += "    float dz = 0.0f;\n";
        source_code += "    float squared_distance = 0.0f;\n";
        source_code += "    for (int i = 0; i < " + std::to_string(num_of_self_collision_check_) + "; i++){\n";
        source_code += "        dx = self_collision_sphere_pos[3 * self_collision_check_pairs[i][0]] - self_collision_sphere_pos[3 * self_collision_check_pairs[i][1]];\n";
        source_code += "        dy = self_collision_sphere_pos[3 * self_collision_check_pairs[i][0] + 1] - self_collision_sphere_pos[3 * self_collision_check_pairs[i][1] + 1];\n";
        source_code += "        dz = self_collision_sphere_pos[3 * self_collision_check_pairs[i][0] + 2] - self_collision_sphere_pos[3 * self_collision_check_pairs[i][1] + 2];\n";
        source_code += "        squared_distance = dx * dx + dy * dy + dz * dz;\n";
        source_code += "        if (squared_distance < self_collision_distance_threshold[i]){\n";
        source_code += "            return true;\n";
        source_code += "        }\n";
        source_code += "    }\n";
        source_code += "    return false;\n";

        source_code += "}\n";
        return source_code;
    }

    std::string RobotSolver::generateFKKernelSourceCode(){
        std::string kernel_source_code;

        // create constants in device memory
        // set for joint_poses
        kernel_source_code += "__constant__ float joint_poses[" + std::to_string(num_of_links_ * 4 * 4) + "] = \n";
        kernel_source_code += "{\n";
        for (size_t i = 0; i < joint_poses_.size(); i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                for (size_t k = 0; k < 4; k++)
                {
                    if ( i == joint_poses_.size() - 1 && j == 3 && k == 3)
                    {
                        kernel_source_code += std::to_string(joint_poses_[i].matrix()(j, k));
                    }
                    else
                    {
                        kernel_source_code += std::to_string(joint_poses_[i].matrix()(j, k)) + ", ";
                    }
                }
            }
            kernel_source_code += "\n";
        }
        kernel_source_code += "};\n";

        // set for joint_axes
        kernel_source_code += "__constant__ float joint_axes[" + std::to_string(num_of_links_ * 3) + "] = \n";
        kernel_source_code += "{\n";
        for (size_t i = 0; i < joint_axes_.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                if ( i == joint_axes_.size() - 1 && j == 2)
                {
                    kernel_source_code += std::to_string(joint_axes_[i](j));
                }
                else
                {
                    kernel_source_code += std::to_string(joint_axes_[i](j)) + ", ";
                }
            }
            kernel_source_code += "\n";
        }
        kernel_source_code += "};\n";

        // set self collision spheres pos in link
        kernel_source_code += "__constant__ float self_collision_spheres_pos_in_link[" + std::to_string(num_of_self_collision_spheres_ * 3) + "] = \n";
        kernel_source_code += "{\n";
        for (size_t i = 0; i < self_collision_spheres_pos_in_link_.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                if ( i == self_collision_spheres_pos_in_link_.size() - 1 && j == 2)
                {
                    kernel_source_code += std::to_string(self_collision_spheres_pos_in_link_[i][j]);
                }
                else
                {
                    kernel_source_code += std::to_string(self_collision_spheres_pos_in_link_[i][j]) + ", ";
                }
            }
            kernel_source_code += "\n";
        }
        kernel_source_code += "};\n";

        // set self collision spheres to link map
        kernel_source_code += "__constant__ int self_collision_spheres_to_link_map[" + std::to_string(num_of_self_collision_spheres_) + "] = \n";
        kernel_source_code += "{\n";
        for (size_t i = 0; i < self_collision_spheres_to_link_map_.size(); i++)
        {
            if ( i == self_collision_spheres_to_link_map_.size() - 1)
            {
                kernel_source_code += std::to_string(self_collision_spheres_to_link_map_[i]);
            }
            else
            {
                kernel_source_code += std::to_string(self_collision_spheres_to_link_map_[i]) + ", ";
            }
        }
        kernel_source_code += "\n};\n";

        kernel_source_code += R"(
 // Multiply two 4x4 matrices (row-major order)
__device__ __forceinline__ void multiply4x4(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C)
{
    C[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8]  + A[3] * B[12];
    C[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9]  + A[3] * B[13];
    C[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
    C[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];

    C[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8]  + A[7] * B[12];
    C[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9]  + A[7] * B[13];
    C[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
    C[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];

    C[8]  = A[8]  * B[0] + A[9]  * B[4] + A[10] * B[8]  + A[11] * B[12];
    C[9]  = A[8]  * B[1] + A[9]  * B[5] + A[10] * B[9]  + A[11] * B[13];
    C[10] = A[8]  * B[2] + A[9]  * B[6] + A[10] * B[10] + A[11] * B[14];
    C[11] = A[8]  * B[3] + A[9]  * B[7] + A[10] * B[11] + A[11] * B[15];

    // Last row is fixed as [0, 0, 0, 1]
    C[12] = 0.f; C[13] = 0.f; C[14] = 0.f; C[15] = 1.f;
}

// Fixed joint: multiply parent's pose with joint's fixed pose.
__device__ __forceinline__ void fixed_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    float* link_pose)
{
    multiply4x4(parent_link_pose, joint_pose, link_pose);
}

// Create a rotation matrix from an axis-angle representation.
__device__ __forceinline__ void make_rotation_axis_angle(float angle, float x, float y, float z, float* R)
{
    float length = sqrtf(x * x + y * y + z * z);
    const float thresh = 1e-12f;
    float valid = (length >= thresh) ? 1.f : 0.f;
    float inv_length = 1.f / fmaxf(length, thresh);
    float nx = x * inv_length * valid;
    float ny = y * inv_length * valid;
    float nz = z * inv_length * valid;
    float c = cosf(angle);
    float s = sinf(angle);
    float one_c = 1.f - c;

    float r0  = c + nx * nx * one_c;
    float r1  = nx * ny * one_c - nz * s;
    float r2  = nx * nz * one_c + ny * s;
    float r4  = ny * nx * one_c + nz * s;
    float r5  = c + ny * ny * one_c;
    float r6  = ny * nz * one_c - nx * s;
    float r8  = nz * nx * one_c - ny * s;
    float r9  = nz * ny * one_c + nx * s;
    float r10 = c + nz * nz * one_c;

    R[0]  = r0 * valid + (1.f - valid) * 1.f; R[1]  = r1 * valid;           R[2]  = r2 * valid;           R[3]  = 0.f;
    R[4]  = r4 * valid;           R[5]  = r5 * valid + (1.f - valid) * 1.f; R[6]  = r6 * valid;           R[7]  = 0.f;
    R[8]  = r8 * valid;           R[9]  = r9 * valid;           R[10] = r10 * valid + (1.f - valid) * 1.f; R[11] = 0.f;
    R[12] = 0.f; R[13] = 0.f; R[14] = 0.f; R[15] = 1.f;
}

// Revolute joint: compute rotation transformation then multiply with parent's pose.
__device__ __forceinline__ void revolute_joint_fn_cuda(const float* parent_link_pose,
                                                        const float* joint_pose,
                                                        const float* joint_axis,
                                                        float joint_value,
                                                        float* link_pose)
{
    float joint_transform[16];
    make_rotation_axis_angle(joint_value, joint_axis[0], joint_axis[1], joint_axis[2], joint_transform);
    
    float temp[16];
    multiply4x4(parent_link_pose, joint_pose, temp);
    multiply4x4(temp, joint_transform, link_pose);
}

// Prismatic joint: create a translation matrix and combine with parent's pose.
__device__ __forceinline__ void prism_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    const float* joint_axis,
                                                    float joint_value,
                                                    float* link_pose)
{
    float x = joint_axis[0], y = joint_axis[1], z = joint_axis[2];
    float T[16] = {
        1.0f, 0.0f, 0.0f, x * joint_value,
        0.0f, 1.0f, 0.0f, y * joint_value,
        0.0f, 0.0f, 1.0f, z * joint_value,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float joint_pose_T[16];
    multiply4x4(joint_pose, T, joint_pose_T);
    multiply4x4(parent_link_pose, joint_pose_T, link_pose);
}       

__device__ __forceinline__ void kin_forward(float * configuration, float * self_collision_spheres)
{
)";
        kernel_source_code += "    // based on the default value and fixed joint, calculate the full joint values.\n";
        kernel_source_code += "    float full_joint_values[" + std::to_string(num_of_joints_) + "];\n";
        // set default values
        int local_dim = 0;
        for (int i = 0; i < num_of_joints_; i++)
        {
            if (active_joint_map_[i])
            {
                kernel_source_code += "    full_joint_values[" + std::to_string(i) + "] = configuration[" + std::to_string(local_dim) + "];\n";
                local_dim++; 
            }
            else{
                kernel_source_code += "    full_joint_values[" + std::to_string(i) + "] = " + std::to_string(default_joint_values_[i]) + ";\n";
            }
        }
        kernel_source_code += "    float link_poses[" + std::to_string(num_of_links_ * 16) + "];\n";
        kernel_source_code += "    // set the base link pose to identity\n";
        kernel_source_code += "    link_poses[0] = 1.0f; link_poses[1] = 0.0f; link_poses[2] = 0.0f; link_poses[3] = 0.0f;\n";
        kernel_source_code += "    link_poses[4] = 0.0f; link_poses[5] = 1.0f; link_poses[6] = 0.0f; link_poses[7] = 0.0f;\n";
        kernel_source_code += "    link_poses[8] = 0.0f; link_poses[9] = 0.0f; link_poses[10] = 1.0f; link_poses[11] = 0.0f;\n";
        kernel_source_code += "    link_poses[12] = 0.0f; link_poses[13] = 0.0f; link_poses[14] = 0.0f; link_poses[15] = 1.0f;\n";

        for (size_t i = 1; i < joint_types_.size(); i++)
        {
            kernel_source_code += "    // Unrolled joint " + std::to_string(i) + "\n";
            if (joint_types_[i] == 1) // REVOLUTE
            {
                kernel_source_code += "    // revolute joint\n";
                kernel_source_code += "    revolute_joint_fn_cuda( &link_poses[" + std::to_string(link_parent_link_maps_[i] * 16) + "], \n";
                kernel_source_code += "        &joint_poses[" + std::to_string(i * 16) + "], \n";
                kernel_source_code += "        &joint_axes[" + std::to_string(i * 3) + "], \n";
                kernel_source_code += "        full_joint_values[" + std::to_string(i) + "], \n";
                kernel_source_code += "        &link_poses[" + std::to_string(i * 16) + "] \n";
                kernel_source_code += "    );\n";
            }
            else if(joint_types_[i] == 2) // PRISMATIC
            {
                kernel_source_code += "    // prismatic joint\n";
                kernel_source_code += "    prism_joint_fn_cuda( &link_poses[" + std::to_string(link_parent_link_maps_[i] * 16) + "], \n";
                kernel_source_code += "        &joint_poses[" + std::to_string(i * 16) + "], \n";
                kernel_source_code += "        &joint_axes[" + std::to_string(i * 3) + "], \n";
                kernel_source_code += "        full_joint_values[" + std::to_string(i) + "], \n";
                kernel_source_code += "        &link_poses[" + std::to_string(i * 16) + "] \n";
                kernel_source_code += "    );\n";
            }
            else if(joint_types_[i] == 5) // FIXED
            {
                kernel_source_code += "    // fixed joint\n";
                kernel_source_code += "    fixed_joint_fn_cuda( &link_poses[" + std::to_string(link_parent_link_maps_[i] * 16) + "], \n";
                kernel_source_code += "        &joint_poses[" + std::to_string(i * 16) + "], \n";
                kernel_source_code += "        &link_poses[" + std::to_string(i * 16) + "] \n";
                kernel_source_code += "    );\n";
            }
            else
            {
                kernel_source_code += "    // Unsupported joint type: " + std::to_string(joint_types_[i]) + "\n";
            }

        }

        kernel_source_code += "    // calculate the self collision spheres positions in the base link frame\n";
        kernel_source_code += "    #pragma unroll\n";
        kernel_source_code += "    for (int i = 0; i < " + std::to_string(num_of_self_collision_spheres_) + "; i++)\n";
        kernel_source_code += "    {\n";
        kernel_source_code += "        float * T = &link_poses[self_collision_spheres_to_link_map[i] * 16];\n";
        kernel_source_code += "        float sx = self_collision_spheres_pos_in_link[i * 3 + 0];\n";
        kernel_source_code += "        float sy = self_collision_spheres_pos_in_link[i * 3 + 1];\n";
        kernel_source_code += "        float sz = self_collision_spheres_pos_in_link[i * 3 + 2];\n";

        kernel_source_code += "        float x = T[0] * sx + T[1] * sy + T[2] * sz + T[3];\n";
        kernel_source_code += "        float y = T[4] * sx + T[5] * sy + T[6] * sz + T[7];\n";
        kernel_source_code += "        float z = T[8] * sx + T[9] * sy + T[10] * sz + T[11];\n";

        kernel_source_code += "        self_collision_spheres[i * 3 + 0] = x;\n";
        kernel_source_code += "        self_collision_spheres[i * 3 + 1] = y;\n";
        kernel_source_code += "        self_collision_spheres[i * 3 + 2] = z;\n";

        kernel_source_code += "    }\n";
        kernel_source_code += R"(
}
        )";
        return kernel_source_code;
    }

} // namespace CPRRTC