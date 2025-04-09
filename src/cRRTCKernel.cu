
#ifndef FLT_MAX
#define FLT_MAX __int_as_float(0x7f7fffff)    // 3.40282347e+38f
#endif

extern "C" {
    __device__ int startTreeCounter = 0;
    __device__ int goalTreeCounter = 0;
    __device__ int sampledCounter = 0;
}

extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations) {
    __shared__ float * tree_to_expand;
    __shared__ int * tree_to_expand_parent_indexs;
    __shared__ int localTargetTreeCounter;
    __shared__ int localSampledCounter;
    __shared__ int localStartTreeCounter;
    __shared__ int localGoalTreeCounter;
    __shared__ float partial_distance_cost_from_nn[32];
    __shared__ int partial_nn_index[32];
    __shared__ float local_sampled_configurations[7];
    const int tid = threadIdx.x;
    // run for loop with max_interations_ iterations
    for (int i = 0; i < 1; i++) {

        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            // increase the sampledCounter with atomic operation
            localSampledCounter = atomicAdd(&sampledCounter, 1);
            localStartTreeCounter = startTreeCounter;
            localGoalTreeCounter = goalTreeCounter;

            if (localStartTreeCounter < localGoalTreeCounter) {
                tree_to_expand = d_start_tree_configurations;
                tree_to_expand_parent_indexs = d_start_tree_parent_indexs;
                localTargetTreeCounter = localStartTreeCounter;
            } else {
                tree_to_expand = d_goal_tree_configurations;
                tree_to_expand_parent_indexs = d_goal_tree_parent_indexs;
                localTargetTreeCounter = localGoalTreeCounter;
            }
        }

        __syncthreads();
        if (localSampledCounter >= 1)
            return; // meet the max_iteration, then stop the block.
        if(tid == 0) {
            printf("localStartTreeCounter: %d\n", localStartTreeCounter);
            printf("localGoalTreeCounter: %d\n", localGoalTreeCounter);
            printf("localSampledCounter: %d\n", localSampledCounter);
            printf("Sampled configuration: ");
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 0]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 1]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 2]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 3]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 4]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 5]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 6]);
            printf("\n");
        }
        // Load the sampled configuration into shared memory
        if (tid < 7) {
            local_sampled_configurations[tid] = d_sampled_configurations[localSampledCounter * 7 + tid];
        }
        __syncthreads();

        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

        float best_dist = FLT_MAX;
        int best_index = -1;
        for (int j = 0; j < localTargetTreeCounter; j += blockDim.x){
            float dist = 0.0f;
            float diff = 0.0f;
            diff = tree_to_expand[j * 7 + 0] - local_sampled_configurations[0];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 1] - local_sampled_configurations[1];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 2] - local_sampled_configurations[2];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 3] - local_sampled_configurations[3];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 4] - local_sampled_configurations[4];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 5] - local_sampled_configurations[5];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 6] - local_sampled_configurations[6];
            dist += diff * diff;

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
            float nearest_dist = sqrtf(partial_distance_cost_from_nn[0]);
            int nearest_idx = partial_nn_index[0];
            printf("Nearest neighbor index: %d, Euclidean distance: %f\n", nearest_idx, nearest_dist);
        }

    }

}