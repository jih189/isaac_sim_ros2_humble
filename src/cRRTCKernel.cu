
extern "C" {
    __device__ int startTreeCounter = 0;
    __device__ int goalTreeCounter = 0;
    __device__ int sampledCounter = 0;
}

extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations) {
    __shared__ float * tree_to_expand;
    __shared__ int localSampledCounter;
    __shared__ float partial_distance_cost_from_nn[256];
    __shared__ int partial_nn_index[256];
    const int tid = threadIdx.x;
    // run for loop with max_interations_ iterations
    for (int i = 0; i < 1; i++) {

        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            // print global variables counters
            printf("startTreeCounter: %d\n", startTreeCounter);
            printf("goalTreeCounter: %d\n", goalTreeCounter);
            printf("sampledCounter: %d\n", sampledCounter);

            // increase the sampledCounter with atomic operation
            localSampledCounter = atomicAdd(&sampledCounter, 1);

            if (startTreeCounter < goalTreeCounter) {
                tree_to_expand = d_start_tree_configurations;
            } else {
                tree_to_expand = d_goal_tree_configurations;
            }
            
            // extract the sampled configuration from the d_sampled_configurations_
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

        __syncthreads();
        if (localSampledCounter >= 1)
            return; // meet the max_iteration, then stop the block.

        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

    }

}