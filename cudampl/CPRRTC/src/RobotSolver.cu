#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include "RobotSolver.h"
#include <algorithm>  // for std::count


namespace CPRRTC
{
    constexpr float UNWRITTEN_VAL = -9999.0f;

    __global__ void initCurand(curandState * state, unsigned long seed, int state_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_size) return;
        curand_init(seed, idx, 0, &state[idx]);
    }

    __global__ void sample_configuration_kernel(
        curandState_t * d_random_state,
        float * d_sampled_configurations,
        int num_of_config,
        int num_of_dim,
        float * d_lower_bound,
        float * d_upper_bound
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_config * num_of_dim) return;

        int joint_idx = idx % num_of_dim;

        curandState_t local_state = d_random_state[idx];
        d_sampled_configurations[idx] = curand_uniform(&local_state) * (d_upper_bound[joint_idx] - d_lower_bound[joint_idx]) + d_lower_bound[joint_idx];
    }

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
        max_iterations_ = 10000;
        num_of_threads_per_motion_ = max_step_ = 32;
        num_of_thread_blocks_ = 512;

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

        // allocate device memory
        size_t configuration_memory_bytes = max_iterations_ * dim_ * sizeof(float);
        size_t parent_indexs_memory_bytes = max_iterations_ * sizeof(int);

        cudaMalloc(&d_start_tree_configurations_,configuration_memory_bytes);
        cudaMalloc(&d_start_tree_parent_indexs_, parent_indexs_memory_bytes);
        cudaMalloc(&d_goal_tree_configurations_,configuration_memory_bytes);
        cudaMalloc(&d_goal_tree_parent_indexs_, parent_indexs_memory_bytes);
        cudaMalloc(&connected_tree_node_pair_, 2 * sizeof(int) * num_of_thread_blocks_);

        // Sample a set of random configurations in the space for later use
        // allocate memory for the d_sampled_configurations_
        size_t sampled_configurations_memory_bytes = max_iterations_ * dim_ * sizeof(float);
        cudaMalloc(&d_sampled_configurations_, sampled_configurations_memory_bytes);

        sampleConfigurations(d_sampled_configurations_, max_iterations_);
    }

    RobotSolver::~RobotSolver()
    {
        // free memory on the device
        cudaFree(d_start_tree_configurations_);
        cudaFree(d_start_tree_parent_indexs_);

        cudaFree(d_goal_tree_configurations_);
        cudaFree(d_goal_tree_parent_indexs_);

        cudaFree(connected_tree_node_pair_);

        cudaFree(d_sampled_configurations_);

        cudaFree(d_spheres_);
        cudaFree(d_cuboids_);
        cudaFree(d_cylinders_);
        
        cRRTCKernelPtr_.reset();
    }

    void RobotSolver::setEnvObstacleCache(int num_of_spheres, int num_of_cuboids, int num_of_cylinders)
    {
        max_num_of_spheres_ = num_of_spheres;
        max_num_of_cuboids_ = num_of_cuboids;
        max_num_of_cylinders_ = num_of_cylinders;

        // allocate memory for the obstacles
        size_t sphere_memory_bytes = max_num_of_spheres_ * sizeof(Sphere);
        size_t cuboid_memory_bytes = max_num_of_cuboids_ * sizeof(Cuboid);
        size_t cylinder_memory_bytes = max_num_of_cylinders_ * sizeof(Cylinder);

        cudaMalloc(&d_spheres_, sphere_memory_bytes);
        cudaMalloc(&d_cuboids_, cuboid_memory_bytes);
        cudaMalloc(&d_cylinders_, cylinder_memory_bytes);
    }

    void RobotSolver::updateEnvObstacle(
        std::vector<Sphere>& spheres,
        std::vector<Cuboid>& cuboids,
        std::vector<Cylinder>& cylinders
    )
    {
        // check the size of spheres, cuboids and cylinders
        if (spheres.size() > max_num_of_spheres_ || cuboids.size() > max_num_of_cuboids_ || cylinders.size() > max_num_of_cylinders_)
        {
            std::cerr << "\033[31m" << "The number of obstacles exceeds the maximum number of obstacles." << "\033[0m" << std::endl;
            return;
        }

        // convert spheres.size() to int
        num_of_spheres_ = static_cast<int>(spheres.size());
        num_of_cuboids_ = static_cast<int>(cuboids.size());
        num_of_cylinders_ = static_cast<int>(cylinders.size());

        // copy the obstacles to device
        if (spheres.size() > 0)
            cudaMemcpy(d_spheres_, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
        if (cuboids.size() > 0)
            cudaMemcpy(d_cuboids_, cuboids.data(), cuboids.size() * sizeof(Cuboid), cudaMemcpyHostToDevice);
        if (cylinders.size() > 0)
            cudaMemcpy(d_cylinders_, cylinders.data(), cylinders.size() * sizeof(Cylinder), cudaMemcpyHostToDevice);
    }

    void RobotSolver::sampleConfigurations(float * d_configurations, int num_of_config)
    {
        std::vector<float> upper_bound_host;
        std::vector<float> lower_bound_host;

        float * d_lower_bound_in_sample_configurations;
        float * d_upper_bound_in_sample_configurations;

        size_t d_bound_bytes = dim_ * sizeof(float);

        cudaMalloc(&d_lower_bound_in_sample_configurations, d_bound_bytes);
        cudaMalloc(&d_upper_bound_in_sample_configurations, d_bound_bytes);

        for (size_t i = 0; i < active_joint_map_.size(); i++)
        {
            if (active_joint_map_[i])
            {
                // copy the upper and lower bound to host
                upper_bound_host.push_back(upper_bound_[i]);
                lower_bound_host.push_back(lower_bound_[i]);
            }
        }

        // copy the upper and lower bound to device
        cudaMemcpy(d_lower_bound_in_sample_configurations, lower_bound_host.data(), d_bound_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_upper_bound_in_sample_configurations, upper_bound_host.data(), d_bound_bytes, cudaMemcpyHostToDevice);

        curandState * d_random_state;
        size_t d_random_state_bytes = num_of_config * dim_ * sizeof(curandState);
        auto allocate_result = cudaMalloc(&d_random_state, d_random_state_bytes);
        if (allocate_result != cudaSuccess)
        {
            // print in red
            std::cerr << "\033[31m" << "Failed to allocate device memory for random state. Perhaps, the num_of_config is too large." << "\033[0m" << std::endl;
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_config * dim_ + threadsPerBlock - 1) / threadsPerBlock;

        unsigned long seed = dist(gen);
        initCurand<<<blocksPerGrid, threadsPerBlock>>>(d_random_state, seed, num_of_config * dim_);

        // call kernel
        sample_configuration_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_random_state, 
            d_configurations, 
            num_of_config, 
            dim_,
            d_lower_bound_in_sample_configurations, 
            d_upper_bound_in_sample_configurations
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // free device memory
        cudaFree(d_random_state);
        cudaFree(d_lower_bound_in_sample_configurations);
        cudaFree(d_upper_bound_in_sample_configurations);
    }

    std::vector<std::vector<float>> RobotSolver::backtraceTree(const std::vector<float>& tree_configurations,
                                            const std::vector<int>& tree_parent_indexs,
                                            int dim,
                                            int start_index)
    {
        std::vector<std::vector<float>> path;
        int index = start_index;
        // Backtrace until we reach the root (assumed to be index 0).
        while (true)
        {
            // Extract the configuration for the current node.
            std::vector<float> config(dim);
            for (int j = 0; j < dim; j++)
            {
                // Each node's configuration is stored consecutively.
                config[j] = tree_configurations[index * dim + j];
            }
            path.push_back(config);
            
            // If we've reached the root node, we finish.
            if (index == 0)
                break;
            
            // Move to the parent of the current node.
            index = tree_parent_indexs[index];
        }
        
        return path;
    }

    std::vector<std::vector<float>> RobotSolver::constructFinalPath(int dim,
                        const std::vector<float>& start_tree_configurations,
                        const std::vector<int>& start_tree_parent_indexs,
                        const std::vector<float>& goal_tree_configurations,
                        const std::vector<int>& goal_tree_parent_indexs,
                        int connection_index_start, // index in the start tree where connection occurred
                        int connection_index_goal)  // index in the goal tree where connection occurred
    {
        // Backtrace the start tree from the connection node back to the start configuration.
        // This yields a sequence from the connection node up to the start.
        std::vector<std::vector<float>> start_path = backtraceTree(start_tree_configurations,
                                                                    start_tree_parent_indexs,
                                                                    dim,
                                                                    connection_index_start);
        // The backtracing from the start tree gives the path in reverse order (from the connection node to the start).
        // Reverse it to obtain a proper order: from the start configuration to the connection node.
        std::reverse(start_path.begin(), start_path.end());
        
        // Backtrace the goal tree from the connection node back to the goal configuration.
        // Here the tree was grown from the goal, so index 0 should correspond to the goal.
        std::vector<std::vector<float>> goal_path = backtraceTree(goal_tree_configurations,
                                                                goal_tree_parent_indexs,
                                                                dim,
                                                                connection_index_goal);
        // The goal tree path is from the connection node to the goal state.
        // Depending on how you wish to join the two paths, you may not need to reverse the goal_path.
        // In this example, we assume that goal_path is already in the order from the connection node to the goal.
        
        // Combine the two paths.
        // Since the connection node is included in both paths, remove the duplicate by skipping the first element of the goal path.
        std::vector<std::vector<float>> final_path = start_path;
        if (!goal_path.empty())
        {
            final_path.insert(final_path.end(), goal_path.begin(), goal_path.end());
        }

        // (Optional) Print the final path for verification.
        std::cout << "Final path:" << std::endl;
        for (size_t i = 0; i < final_path.size(); i++)
        {
            std::cout << "Node " << i << ": ";
            for (int j = 0; j < dim; j++)
            {
                std::cout << final_path[i][j] << " ";
            }
            std::cout << std::endl;
        }

        return final_path;
    }

    std::vector<std::vector<float>> RobotSolver::solve(
        std::vector<float>& start,
        std::vector<float>& goal
    )
    {
        // check the size of start and goal
        if (start.size() != dim_ || goal.size() != dim_)
        {
            std::cerr << "\033[31m" << "The size of start and goal configurations must be equal to the dimension of the robot." << "\033[0m" << std::endl;
            return {};
        }

        // clear the device memory
        cudaMemset(d_start_tree_configurations_, UNWRITTEN_VAL, max_iterations_ * dim_ * sizeof(float));
        cudaMemset(d_start_tree_parent_indexs_, 0, sizeof(int));
        cudaMemset(d_goal_tree_configurations_, UNWRITTEN_VAL, max_iterations_ * dim_ * sizeof(float));
        cudaMemset(d_goal_tree_parent_indexs_, 0, sizeof(int));
        cudaMemset(connected_tree_node_pair_, -1, 2 * sizeof(int) * num_of_thread_blocks_);

        // pass first start and goal configurations to the device by copying them to the device
        cudaMemcpy(d_start_tree_configurations_, start.data(), (size_t)(dim_ * sizeof(float)), cudaMemcpyHostToDevice);
        cudaMemcpy(d_goal_tree_configurations_, goal.data(), (size_t)(dim_ * sizeof(float)), cudaMemcpyHostToDevice);

        // Retrieve global variable pointers from the compiled module.
        CUdeviceptr d_startTreeCounter, d_goalTreeCounter, d_sampledCounter, d_foundSolution;
        size_t varSize;
        cuModuleGetGlobal(&d_startTreeCounter, &varSize, cRRTCKernelPtr_->module, "startTreeCounter");
        cuModuleGetGlobal(&d_goalTreeCounter, &varSize, cRRTCKernelPtr_->module, "goalTreeCounter");
        cuModuleGetGlobal(&d_sampledCounter, &varSize, cRRTCKernelPtr_->module, "sampledCounter");
        cuModuleGetGlobal(&d_foundSolution, &varSize, cRRTCKernelPtr_->module, "foundSolution");

        int h_startTreeCounter = 1;
        int h_goalTreeCounter = 1;
        int h_sampledCounter = 0;
        int h_foundSolution = 0;

        // Copy the initial values to the device
        cuMemcpyHtoD(d_startTreeCounter, &h_startTreeCounter, sizeof(int));
        cuMemcpyHtoD(d_goalTreeCounter, &h_goalTreeCounter, sizeof(int));
        cuMemcpyHtoD(d_sampledCounter, &h_sampledCounter, sizeof(int));
        cuMemcpyHtoD(d_foundSolution, &h_foundSolution, sizeof(int));

        // Set up kernel launch parameters
        void *args[] = {
            &d_start_tree_configurations_,
            &d_goal_tree_configurations_,
            &d_start_tree_parent_indexs_,
            &d_goal_tree_parent_indexs_,
            &d_sampled_configurations_,
            &connected_tree_node_pair_,
            &d_spheres_,
            &num_of_spheres_
        };

        int threads_per_block = num_of_threads_per_motion_;
        int blocks_per_grid = num_of_thread_blocks_;

        auto kernel_start_time = std::chrono::high_resolution_clock::now();

        cRRTCKernelPtr_->launchKernel(
            dim3(blocks_per_grid, 1, 1), // grid size
            dim3(threads_per_block, 1, 1), // block size
            0, // shared memory size
            nullptr, // stream
            args // kernel arguments
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        auto kernel_end_time = std::chrono::high_resolution_clock::now();
        auto kernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end_time - kernel_start_time).count();
        std::cout << "Kernel execution time: " << kernel_duration << " ms" << std::endl;

        std::vector<int> connected_tree_node_pair(num_of_thread_blocks_ * 2);
        cudaMemcpy(connected_tree_node_pair.data(), connected_tree_node_pair_, num_of_thread_blocks_ * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        int current_start_tree_num = 0;
        int current_goal_tree_num = 0;
        bool found = false;
        for (int i = 0; i < num_of_thread_blocks_; i++)
        {
            if (connected_tree_node_pair[i * 2] != -1  && connected_tree_node_pair[i * 2 + 1] != -1)
            {
                current_start_tree_num = connected_tree_node_pair[i * 2];
                current_goal_tree_num = connected_tree_node_pair[i * 2 + 1];
                found = true;
                break;
            }
        }

        if (found)
        {
            current_start_tree_num += 1;
            current_goal_tree_num += 1;
            // print d_start_tree_configurations_ with first current_start_tree_num configurations
            std::vector<float> start_tree_configurations(current_start_tree_num * dim_);
            std::vector<int> start_tree_parent_indexs(current_start_tree_num);
            cudaMemcpy(start_tree_configurations.data(), d_start_tree_configurations_, current_start_tree_num * dim_ * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(start_tree_parent_indexs.data(), d_start_tree_parent_indexs_, current_start_tree_num * sizeof(int), cudaMemcpyDeviceToHost);

            // print d_goal_tree_configurations_ with first current_goal_tree_num configurations
            std::vector<float> goal_tree_configurations(current_goal_tree_num * dim_);
            std::vector<int> goal_tree_parent_indexs(current_goal_tree_num);
            cudaMemcpy(goal_tree_configurations.data(), d_goal_tree_configurations_, current_goal_tree_num * dim_ * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(goal_tree_parent_indexs.data(), d_goal_tree_parent_indexs_, current_goal_tree_num * sizeof(int), cudaMemcpyDeviceToHost);

            // Get the final path
            // std::cout << "Final path: " << std::endl;
            std::vector<std::vector<float>> final_path = constructFinalPath(dim_, start_tree_configurations, start_tree_parent_indexs, goal_tree_configurations, goal_tree_parent_indexs, current_start_tree_num - 1, current_goal_tree_num - 1);
            
            return final_path;
        }


        return {};
    }

    std::string RobotSolver::generateEnvSphereCollisionCheckSourceCode()
    {
        std::string kernel_code;
        kernel_code += "__device__ __forceinline__ bool checkEnvSphereCollisionConstraint(float * self_collision_sphere_pos, float * d_spheres, int num_of_spheres) {\n";

        kernel_code += "    float dx = 0.0f;\n";
        kernel_code += "    float dy = 0.0f;\n";
        kernel_code += "    float dz = 0.0f;\n";
        kernel_code += "    float sq_dis = 0.0f;\n";
        kernel_code += "    float threshold = 0.0f;\n";
        kernel_code += "    float sq_threshold = 0.0f;\n";
        kernel_code += "    for (int i = 0; i < num_of_spheres; i++) {\n";
        for (int i  = 0 ; i < num_of_self_collision_spheres_; i++)
        {
            kernel_code += "        // self sphere " + std::to_string(i) + "\n";
            kernel_code += "        threshold = d_spheres[i * 4 + 3] + " + std::to_string(self_collision_spheres_radius_[i]) + ";\n";
            kernel_code += "        sq_threshold = threshold * threshold;\n";
            kernel_code += "        dx = self_collision_sphere_pos[" + std::to_string(3 * i) + "] - d_spheres[i * 4];\n";
            kernel_code += "        dy = self_collision_sphere_pos[" + std::to_string(3 * i + 1) + "] - d_spheres[i * 4 + 1];\n";
            kernel_code += "        dz = self_collision_sphere_pos[" + std::to_string(3 * i + 2) + "] - d_spheres[i * 4 + 2];\n";
            kernel_code += "        sq_dis = dx * dx + dy * dy + dz * dz;\n";
            kernel_code += "        if (sq_dis < sq_threshold) {\n";
            kernel_code += "            return true;\n";
            kernel_code += "        }\n";
        }
        kernel_code += "    }\n";

        kernel_code += "    return false;\n";
        kernel_code += "}\n";
        return kernel_code;
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
        kernel_code += generateEnvSphereCollisionCheckSourceCode();

        kernel_code += "__device__ __forceinline__ bool check_partially_written(float *node) {\n";
        kernel_code += "    #pragma unroll\n";
        kernel_code += "    for (int i = 0; i < " + std::to_string(dim_) + "; i++) {\n";
        kernel_code += "        if (node[i] == UNWRITTEN_VAL) return true;\n";
        kernel_code += "    }\n";
        kernel_code += "    return false;\n";
        kernel_code += "}\n";

        // kernel_code += "extern \"C\" __global__ void CPRRTCKernel(float* d_start_tree_configurations, float* d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations, int * connected_tree_node_pair, int num_of_sphere_obstacles, float * d_sphere_obstacles, int num_of_cuboid_obstacles, float * d_cuboid_obstacles, int num_of_cylinder_obstacles, float * d_cylinder_obstacles){\n";
        kernel_code += "extern \"C\" __global__ void CPRRTCKernel(float* d_start_tree_configurations, float* d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations, int * connected_tree_node_pair, float * d_spheres, int num_of_spheres){\n";
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
        kernel_code += "    __shared__ bool local_found_solution;\n";
        kernel_code += "    __shared__ int new_node_index;\n";
        kernel_code += "    __shared__ int connected_node_in_target_tree;\n";
        kernel_code += "    __shared__ int connected_node_in_other_tree;\n";
        kernel_code += "    __shared__ int connected_index_in_other_tree;\n";
        kernel_code += "    const int tid = threadIdx.x;\n";
        kernel_code += "    float self_collision_spheres_pos_in_base[" + std::to_string(num_of_self_collision_spheres_ * 3) + "];\n\n";
        kernel_code += "    for (int t = 0; t < " + std::to_string(max_iterations_) + "; t++) {\n";
        kernel_code += "        __syncthreads();\n";
        kernel_code += "        // Need to decide which tree to grow\n";
        kernel_code += "        if (tid == 0) {\n";
        kernel_code += "            should_skip = false;\n";
        kernel_code += "            local_found_solution = (foundSolution != 0);\n";
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
        kernel_code += "        // If meet the max iterations or found the solution, skip the rest of the iterations\n";
        kernel_code += "        if (localSampledCounter >= " + std::to_string(max_iterations_) + " || local_found_solution){\n";
        kernel_code += "            return;\n";
        kernel_code += "        }\n\n";
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
        kernel_code += "            kin_forward(&(local_motion_configurations[tid * " + std::to_string(dim_) + "]), self_collision_spheres_pos_in_base);\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        // Check for self-collision\n";
        kernel_code += "        if (tid < motion_step) {\n";
        kernel_code += "            if(checkSelfCollisionConstraint(self_collision_spheres_pos_in_base))\n";
        kernel_code += "                should_skip = true;\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
        kernel_code += "        if (should_skip) {\n";
        kernel_code += "            continue;\n";
        kernel_code += "        }\n\n";
        kernel_code += "        // Check for environment collision\n";
        kernel_code += "        if (tid < motion_step) {\n";
        kernel_code += "            if(checkEnvSphereCollisionConstraint(self_collision_spheres_pos_in_base, d_spheres, num_of_spheres))\n";
        kernel_code += "                should_skip = true;\n";
        kernel_code += "        }\n";
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
        kernel_code += "        while (!should_skip) {\n";
        kernel_code += "            __syncthreads();\n\n";
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
        kernel_code += "                kin_forward(&(local_motion_configurations[tid * " + std::to_string(dim_) + "]), self_collision_spheres_pos_in_base);\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            // Check for self-collision\n";
        kernel_code += "            if (tid < motion_step) {\n";
        kernel_code += "                if(checkSelfCollisionConstraint(self_collision_spheres_pos_in_base))\n";
        kernel_code += "                    should_skip = true;\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            if (should_skip) {\n";
        kernel_code += "                break;\n";
        kernel_code += "            }\n\n";
        kernel_code += "            // Check for collision with environment obstacles\n";
        kernel_code += "            if (tid < motion_step) {\n";
        kernel_code += "                if(checkEnvSphereCollisionConstraint(self_collision_spheres_pos_in_base, d_spheres, num_of_spheres))\n";
        kernel_code += "                    should_skip = true;\n";
        kernel_code += "            }\n";
        kernel_code += "            __syncthreads();\n\n";
        kernel_code += "            if (should_skip) {\n";
        kernel_code += "                break;\n";
        kernel_code += "            }\n\n";
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
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n\n";
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