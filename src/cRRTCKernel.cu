
__device__ int start_tree_size_counter;
__device__ int goal_tree_size_counter;
__device__ int sampled_configuration_index_counter;
extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, int * d_start_tree_size, int * d_goal_tree_size, float * d_sampled_configurations) {
    __shared__ float * tree_to_expand;
    __shared__ float partial_distance_cost_from_nn[256];
    __shared__ int partial_nn_index[256];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // initialize all counters
    if (tid == 0 && bid == 0)
    {
        start_tree_size_counter = 1;
        goal_tree_size_counter = 1;
        sampled_configuration_index_counter = 0;
    }
    __syncthreads();
    // run for loop with max_interations_ iterations
    for (int i = 0; i < 1; i++) {

        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            if (*d_start_tree_size < *d_goal_tree_size) {
                tree_to_expand = d_start_tree_configurations;
            } else {
                tree_to_expand = d_goal_tree_configurations;
            }
            
            // extract the sampled configuration from the d_sampled_configurations_
            printf("Sampled configuration: ");
            printf("%f ", d_sampled_configurations[i * 7 + 0]);
            printf("%f ", d_sampled_configurations[i * 7 + 1]);
            printf("%f ", d_sampled_configurations[i * 7 + 2]);
            printf("%f ", d_sampled_configurations[i * 7 + 3]);
            printf("%f ", d_sampled_configurations[i * 7 + 4]);
            printf("%f ", d_sampled_configurations[i * 7 + 5]);
            printf("%f ", d_sampled_configurations[i * 7 + 6]);
            printf("\n");

        }

        __syncthreads();
        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

    }

}