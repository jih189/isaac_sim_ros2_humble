// kernel to calculate the distance between two states
__global__ void calculate_joint_state_distance(
    float * d_states_1, int num_of_states_1,
    float * d_states_2, int num_of_states_2, 
    int num_of_joints, int * d_active_joint_map, float * d_distances) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_of_states_1 * num_of_states_2)
        return;

    int state_1_idx = idx / num_of_states_2;
    int state_2_idx = idx % num_of_states_2;

    float sum = 0.0f;

    for (int i = 0; i < num_of_joints; i++) {
        if (d_active_joint_map[i] != 0) {
            float diff = d_states_1[state_1_idx * num_of_joints + i] - d_states_2[state_2_idx * num_of_joints + i];
            sum += diff * diff;
        }
    }

    d_distances[idx] = sqrtf(sum);
}

int SingleArmStateManager::find_k_nearest_neighbors(
    int k, const BaseStatesPtr & query_states, 
    const std::vector<std::vector<int>> & group_indexs,
    std::vector<std::vector<int>> & neighbors_index
)
{

    if (num_of_states_ == 0)
    {
        // raise error
        throw std::runtime_error("Error in SingleArmStateManager::find_k_nearest_neighbors: manager is empty");
    }
    if (query_states->getNumOfStates() == 0)
    {
        // raise error
        throw std::runtime_error("Error in SingleArmStateManager::find_k_nearest_neighbors: query states is empty");
    }

    // static cast the query states to SingleArmStates
    SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

    // static cast the states to SingleArmStates
    SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(query_states);
    float * d_query_joint_states = single_arm_states->getJointStatesCuda();

    neighbors_index.clear();

    int total_actual_k = 0;
    std::vector<int> actual_k_in_each_group;
    for (size_t i = 0; i < group_indexs.size(); i++)
    {
        actual_k_in_each_group.push_back((int)(group_indexs[i].size()) < k ? (int)(group_indexs[i].size()) : k);
        total_actual_k += actual_k_in_each_group[i];
    }

    float * d_distances_from_query_to_states;
    size_t d_distances_from_query_to_states_bytes = query_states->getNumOfStates() * num_of_states_ * sizeof(float);
    cudaMalloc(&d_distances_from_query_to_states, d_distances_from_query_to_states_bytes);

    // calculate the distance between the query states and the states in the manager
    int block_size = 256;
    int grid_size = (query_states->getNumOfStates() * num_of_states_ + block_size - 1) / block_size;

    calculate_joint_state_distance<<<grid_size, block_size>>>(
        d_query_joint_states, query_states->getNumOfStates(),
        d_joint_states, num_of_states_,
        num_of_joints, single_arm_space_info->d_active_joint_map, d_distances_from_query_to_states
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    std::vector<std::vector<float>> distances_from_query_to_states(query_states->getNumOfStates(), std::vector<float>(num_of_states_));
    std::vector<float> distances_from_query_to_states_flatten(query_states->getNumOfStates() * num_of_states_);

    // copy the distances from device to host
    cudaMemcpy(distances_from_query_to_states_flatten.data(), d_distances_from_query_to_states, query_states->getNumOfStates() * num_of_states_ * sizeof(float), cudaMemcpyDeviceToHost);

    // reshape the distances
    for (int i = 0; i < query_states->getNumOfStates(); i++) {
        for (int j = 0; j < num_of_states_; j++) {
            distances_from_query_to_states[i][j] = distances_from_query_to_states_flatten[i * num_of_states_ + j];
        }
    }

    for (int i = 0; i < query_states->getNumOfStates(); i++) {
        std::vector<int> index_k_nearest_neighbors;
        for (size_t g = 0 ; g < group_indexs.size(); g++)
        {
            // find index of the k least distances of distances_from_query_to_states[i]
            std::vector<int> index_k_nearest_neighbors_of_group = kLeastIndices(distances_from_query_to_states[i], actual_k_in_each_group[g], group_indexs[g]);
            index_k_nearest_neighbors.insert(index_k_nearest_neighbors.end(), index_k_nearest_neighbors_of_group.begin(), index_k_nearest_neighbors_of_group.end());
        }

        neighbors_index.push_back(index_k_nearest_neighbors);
    }

    // free the memory
    cudaFree(d_distances_from_query_to_states);

    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    return total_actual_k;
}