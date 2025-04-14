#include <planners/cRRTC.h>

// include for time
#include <chrono>

// include for file operations
#include <fstream>
#include <iostream>
#include <string>


namespace CUDAMPLib
{
    constexpr float UNWRITTEN_VAL = -9999.0f;

    // Constructor
    cRRTC::cRRTC(BaseSpacePtr space)
        : BasePlanner(space)
    {
        max_interations_ = 100;
        num_of_threads_per_motion_ = 32;
        dim_ = space->getDim();
        forward_kinematics_kernel_source_code_ = space->generateFKKernelSourceCode();
        robot_collision_model_kernel_source_code_ = space->generateRobotCollisionModelSourceCode();
        constraint_functions_kernel_source_code_ = space->generateCheckConstraintCode();
        launch_check_constraint_kernel_source_code_ = space->generateLaunchCheckConstraintCode();

        step_resolution_ = 0.02f;
        max_step_ = 32;

        size_t configuration_memory_bytes = max_interations_ * dim_ * sizeof(float);
        size_t parent_indexs_memory_bytes = max_interations_ * sizeof(int);

        // allocate memory on the device
        cudaMalloc(&d_start_tree_configurations_,configuration_memory_bytes);
        cudaMalloc(&d_start_tree_parent_indexs_, parent_indexs_memory_bytes);

        cudaMalloc(&d_goal_tree_configurations_,configuration_memory_bytes);
        cudaMalloc(&d_goal_tree_parent_indexs_, parent_indexs_memory_bytes);

        // Create the source code for motion planning and compile it with nvrtc.
        std::string source_code = generateSourceCode();

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

        // Create the kernel function using the class's static factory method.
        cRRTCKernelPtr_ = KernelFunction::create(source_code_c_str, "cRRTCKernel");

        if (! cRRTCKernelPtr_ || ! cRRTCKernelPtr_->function) {
            std::cerr << "\033[31m" << "Kernel function 'cRRTCKernel' compilation failed." << "\033[0m" << std::endl;
        }

        // Sample a set of random configurations in the space for later use
        // allocate memory for the d_sampled_configurations_
        size_t sampled_configurations_memory_bytes = max_interations_ * dim_ * sizeof(float);
        cudaMalloc(&d_sampled_configurations_, sampled_configurations_memory_bytes);
        space->sampleConfigurations(d_sampled_configurations_, max_interations_);
    }

    // Destructor
    cRRTC::~cRRTC()
    {
        // free memory on the device
        cudaFree(d_start_tree_configurations_);
        cudaFree(d_start_tree_parent_indexs_);

        cudaFree(d_goal_tree_configurations_);
        cudaFree(d_goal_tree_parent_indexs_);

        cudaFree(d_sampled_configurations_);

        cRRTCKernelPtr_.reset();
    }

    void cRRTC::setMotionTask(BaseTaskPtr task, bool get_full_path)
    {
        // set the get full path flag
        get_full_path_ = get_full_path;
        task_ = task;

        // clear the start and goal states
        start_states_vector_.clear();
        goal_states_vector_.clear();

        // get the start and goal states
        start_states_vector_ = task->getStartStatesVector();
        goal_states_vector_ = task->getGoalStatesVector();
    }

    void cRRTC::solve(BaseTerminationPtr termination_condition)
    {
        std::vector<float> first_start_configuration = start_states_vector_[0];
        std::vector<float> first_goal_configuration = goal_states_vector_[0];

        // clear the device memory
        cudaMemset(d_start_tree_configurations_, UNWRITTEN_VAL, max_interations_ * dim_ * sizeof(float));
        // cudaMemset(d_start_tree_parent_indexs_, 0, max_interations_ * sizeof(int));
        cudaMemset(d_start_tree_parent_indexs_, 0, sizeof(int));
        cudaMemset(d_goal_tree_configurations_, UNWRITTEN_VAL, max_interations_ * dim_ * sizeof(float));
        // cudaMemset(d_goal_tree_parent_indexs_, 0, max_interations_ * sizeof(int));
        cudaMemset(d_goal_tree_parent_indexs_, 0, sizeof(int));

        // pass first start and goal configurations to the device by copying them to the device
        cudaMemcpy(d_start_tree_configurations_, first_start_configuration.data(), (size_t)(dim_ * sizeof(float)), cudaMemcpyHostToDevice);
        cudaMemcpy(d_goal_tree_configurations_, first_goal_configuration.data(), (size_t)(dim_ * sizeof(float)), cudaMemcpyHostToDevice);

        // Retrieve global variable pointers from the compiled module.
        CUdeviceptr d_startTreeCounter, d_goalTreeCounter, d_sampledCounter;
        size_t varSize;
        cuModuleGetGlobal(&d_startTreeCounter, &varSize, cRRTCKernelPtr_->module, "startTreeCounter");
        cuModuleGetGlobal(&d_goalTreeCounter, &varSize, cRRTCKernelPtr_->module, "goalTreeCounter");
        cuModuleGetGlobal(&d_sampledCounter, &varSize, cRRTCKernelPtr_->module, "sampledCounter");

        int h_startTreeCounter = 1;
        int h_goalTreeCounter = 1;
        int h_sampledCounter = 0;

        // Copy the initial values to the device
        cuMemcpyHtoD(d_startTreeCounter, &h_startTreeCounter, sizeof(int));
        cuMemcpyHtoD(d_goalTreeCounter, &h_goalTreeCounter, sizeof(int));
        cuMemcpyHtoD(d_sampledCounter, &h_sampledCounter, sizeof(int));

        // Launch the kernel function

        // Set up kernel launch parameters
        void *args[] = {
            &d_start_tree_configurations_,
            &d_goal_tree_configurations_,
            &d_start_tree_parent_indexs_,
            &d_goal_tree_parent_indexs_,
            &d_sampled_configurations_
        };

        int threads_per_block = num_of_threads_per_motion_;
        int blocks_per_grid = 1;

        cRRTCKernelPtr_->launchKernel(
            dim3(blocks_per_grid, 1, 1), // grid size
            dim3(threads_per_block, 1, 1), // block size
            0, // shared memory size
            nullptr, // stream
            args // kernel arguments
        );

        cudaDeviceSynchronize();
    }

    std::string cRRTC::generateSourceCode()
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
}

)";

        kernel_code += "__device__ __forceinline__ bool check_partially_written(float *node) {\n";
        kernel_code += "    for (int i = 0; i < " + std::to_string(dim_) + "; i++) {\n";
        kernel_code += "        if (node[i] == UNWRITTEN_VAL) return true;\n";
        kernel_code += "    }\n";
        kernel_code += "    return false;\n";
        kernel_code += "}\n";

        kernel_code += forward_kinematics_kernel_source_code_;

        kernel_code += "\n";

        kernel_code += constraint_functions_kernel_source_code_;

        kernel_code += R"(
extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations) {
)";
    kernel_code += "    __shared__ float * tree_to_expand;\n";
    kernel_code += "    __shared__ int * tree_to_expand_parent_indexs;\n";
    kernel_code += "    __shared__ int localTargetTreeCounter;\n";
    kernel_code += "    __shared__ int localSampledCounter;\n";
    kernel_code += "    __shared__ int localStartTreeCounter;\n";
    kernel_code += "    __shared__ int localGoalTreeCounter;\n";
    kernel_code += "    __shared__ float partial_distance_cost_from_nn[" + std::to_string(num_of_threads_per_motion_) + "];\n";
    kernel_code += "    __shared__ int partial_nn_index[" + std::to_string(num_of_threads_per_motion_) + "];\n";
    kernel_code += "    __shared__ float local_sampled_configuration[" + std::to_string(dim_) + "];\n";
    kernel_code += "    __shared__ float local_parent_configuration[" + std::to_string(dim_) + "];\n";
    kernel_code += "    __shared__ float local_delta_motion[" + std::to_string(dim_) + "];\n";
    kernel_code += "    __shared__ int local_parent_index;\n";
    kernel_code += "    __shared__ float local_nearest_neighbor_distance;\n";
    kernel_code += "    __shared__ float local_motion_configurations[" + std::to_string(dim_ * max_step_) + "]; \n";
    kernel_code += "    __shared__ int motion_step;\n";
    kernel_code += "    __shared__ bool should_skip;\n";
    kernel_code += "    __shared__ int * target_tree_counter;\n";
    kernel_code += "    __shared__ int new_node_index;\n";
    kernel_code += "    const int tid = threadIdx.x;\n";
    kernel_code += "    " + robot_collision_model_kernel_source_code_ + "\n";
    kernel_code += "    // run for loop with max_interations_ iterations\n";
    kernel_code += "    for (int i = 0; i < " + std::to_string(max_interations_) + "; i++) {\n";

    kernel_code += R"(
        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            should_skip = false;
            // increase the sampledCounter with atomic operation
            localSampledCounter = atomicAdd(&sampledCounter, 1);
            localStartTreeCounter = startTreeCounter;
            localGoalTreeCounter = goalTreeCounter;

            if (localStartTreeCounter < localGoalTreeCounter) {
                tree_to_expand = d_start_tree_configurations;
                tree_to_expand_parent_indexs = d_start_tree_parent_indexs;
                localTargetTreeCounter = localStartTreeCounter;
                target_tree_counter = &startTreeCounter;
            } else {
                tree_to_expand = d_goal_tree_configurations;
                tree_to_expand_parent_indexs = d_goal_tree_parent_indexs;
                localTargetTreeCounter = localGoalTreeCounter;
                target_tree_counter = &goalTreeCounter;
            }
        }

        __syncthreads();
)";

        kernel_code += "        if (localSampledCounter >= " + std::to_string(max_interations_) + ")\n";
        kernel_code += "            return; // meet the max_iteration, then stop the block.\n";
        // kernel_code += "        if(tid == 0) {\n";
        // kernel_code += "            printf(\"localStartTreeCounter: %d\\n\", localStartTreeCounter);\n";
        // kernel_code += "            printf(\"localGoalTreeCounter: %d\\n\", localGoalTreeCounter);\n";
        // kernel_code += "            printf(\"localSampledCounter: %d\\n\", localSampledCounter);\n";
        // kernel_code += "            printf(\"Sampled configuration: \");\n";
        // for (int j = 0; j < dim_; j++)
        // {
        //     kernel_code += "            printf(\"%f \", d_sampled_configurations[localSampledCounter * " + std::to_string(dim_) + " + " + std::to_string(j) + "]);\n";
        // }
        // kernel_code += "            printf(\"\\n\");\n";
        // kernel_code += "        }\n";

        kernel_code += "        // Load the sampled configuration into shared memory\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            local_sampled_configuration[tid] = d_sampled_configurations[localSampledCounter * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n";

        kernel_code += R"(
        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

        float best_dist = FLT_MAX;
        int best_index = -1;
        for (int j = tid; j < localTargetTreeCounter; j += blockDim.x){
)";
        kernel_code += "            if (check_partially_written(&tree_to_expand[j * " + std::to_string(dim_) + "])) break;\n";
        kernel_code += "            float dist = 0.0f;\n";
        kernel_code += "            float diff = 0.0f;\n";
        for (int j = 0; j < dim_; j++)
        {
            kernel_code += "            diff = tree_to_expand[j * " + std::to_string(dim_) + " + " + std::to_string(j) + "] - local_sampled_configuration[" + std::to_string(j) + "];\n";
            kernel_code += "            dist += diff * diff;\n";
        }

kernel_code += R"(
            if (dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }

        // Write the local best distance and index to the shared memory
        partial_distance_cost_from_nn[tid] = best_dist;
        partial_nn_index[tid] = best_index;
        __syncthreads();

        // Perform reduction to find the best distance and index
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (partial_distance_cost_from_nn[tid + stride] < partial_distance_cost_from_nn[tid]) {
                    partial_distance_cost_from_nn[tid] = partial_distance_cost_from_nn[tid + stride];
                    partial_nn_index[tid] = partial_nn_index[tid + stride];
                }
            }
            __syncthreads();
        }

        // After the reduction, thread 0 has the overall nearest neighbor's index and its squared distance.
        if (tid == 0) {
            local_nearest_neighbor_distance = sqrtf(partial_distance_cost_from_nn[0]);
            local_parent_index = partial_nn_index[0];
)";

        kernel_code += "            motion_step = min((int)(local_nearest_neighbor_distance / " + std::to_string(step_resolution_) + "), " + std::to_string(max_step_) + ");\n";
        // kernel_code += "            printf(\"Nearest neighbor index: %d, Euclidean distance: %f motion step: %d \\n \", local_parent_index, local_nearest_neighbor_distance, motion_step);\n";
        kernel_code += "        }\n";
        kernel_code += "        __syncthreads();\n";
        kernel_code += "        // Calculate the delta motion from the nearest configuration to the sampled configuration\n";
        kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
        kernel_code += "            local_parent_configuration[tid] = tree_to_expand[local_parent_index * " + std::to_string(dim_) + " + tid];\n";
        kernel_code += "            local_delta_motion[tid] = (local_sampled_configuration[tid] - local_parent_configuration[tid]) / local_nearest_neighbor_distance * " + std::to_string(step_resolution_) + ";\n";
        kernel_code += "        }\n";
        kernel_code += R"(
        __syncthreads();
)";

    kernel_code += "        // interpolate the new configuration from the nearest configuration and the sampled configuration\n";
    kernel_code += "        for (int j = tid; j < " + std::to_string(dim_) + " * motion_step; j += blockDim.x) {\n";
    kernel_code += "            int state_ind_in_motion = j / " + std::to_string(dim_) + ";\n";
    kernel_code += "            int joint_ind_in_state = j % " + std::to_string(dim_) + ";\n";
    kernel_code += "            local_motion_configurations[j] = local_parent_configuration[joint_ind_in_state] + local_delta_motion[joint_ind_in_state] * state_ind_in_motion;\n";
    kernel_code += "        }\n";
    kernel_code += "        __syncthreads();\n\n";

    // kernel_code += "        // print the intermediate configurations for debugging\n";
    // kernel_code += "        if (tid == 0) {\n";
    // kernel_code += "            for (int j = 0; j < motion_step; j++) {\n";
    // kernel_code += "                printf(\"Intermediate configuration %d: \", j);\n";
    // for (int j = 0; j < dim_; j++)
    // {
    //     kernel_code += "                printf(\"%f \", local_motion_configurations[j * " + std::to_string(dim_) + " + " + std::to_string(j) + "]);\n";
    // }
    // kernel_code += "                printf(\"\\n\");\n";
    // kernel_code += "             }\n";
    // kernel_code += "        }\n";

    // call the forward kinematics kernel
    kernel_code += "        // call the forward kinematics kernel\n";
    kernel_code += "        if (tid < motion_step) {\n";
    kernel_code += "            kin_forward(&(local_motion_configurations[tid]), self_collision_spheres_pos_in_base);\n";
    kernel_code += "        }\n";
    kernel_code += "        __syncthreads();\n\n";
    kernel_code += launch_check_constraint_kernel_source_code_;
    kernel_code += "        // add the new configuration to the tree_to_expand as a new node\n";
    kernel_code += "        if (tid == 0) {\n";
    kernel_code += "            new_node_index = atomicAdd(target_tree_counter, 1);\n";
    kernel_code += "            tree_to_expand_parent_indexs[new_node_index] = local_parent_index;\n";
    // kernel_code += "            // print the last configuration of the motion\n";
    // kernel_code += "            printf(\"Parent node index: %d New node index: %d \\n\", local_parent_index, new_node_index);\n";
    // kernel_code += "            printf(\"%f %f %f %f %f %f %f \\n\", local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + "], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 1], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 2], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 3], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 4], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 5], local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + 6]);\n";
    kernel_code += "        }\n";
    kernel_code += "        __syncthreads();\n";
    kernel_code += "        if (tid < " + std::to_string(dim_) + ") {\n";
    kernel_code += "            tree_to_expand[new_node_index * " + std::to_string(dim_) + " + tid] = local_motion_configurations[(motion_step - 1) * " + std::to_string(dim_) + " + tid];\n";
    kernel_code += "        }\n";
    kernel_code += "        __syncthreads();\n";
    kernel_code += "    }\n";
    
    kernel_code += R"(
})";
        return kernel_code;
    }
} // namespace CUDAMPLib