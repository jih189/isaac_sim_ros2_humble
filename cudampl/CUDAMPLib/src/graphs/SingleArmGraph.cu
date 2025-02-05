#include "graphs/SingleArmGraph.h"

namespace CUDAMPLib
{

    // kernel to calculate the distance between two states
    __global__ void calculate_distance(
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
            if (d_active_joint_map[i] == 1) {
                float diff = d_states_1[state_1_idx * num_of_joints + i] - d_states_2[state_2_idx * num_of_joints + i];
                sum += diff * diff;
            }
        }

        d_distances[idx] = sqrtf(sum);
    }

    SingleArmGraph::SingleArmGraph(int num_of_joints) : BaseGraph(), num_of_joints(num_of_joints)
    {
        // Initialize the graph
    }

    SingleArmGraph::~SingleArmGraph() {
        // Destroy the graph
        if (get_size() > 0) {
            // Free the memory
            cudaFree(d_states_in_graph);
        }
    }

    void SingleArmGraph::add_states_helper(const BaseStatesPtr & states, int group) {
        // Add the states to the graph

        // static cast the states to SingleArmStates
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        int data_size = single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints() * sizeof(float);

        int current_graph_size = get_size();

        if (current_graph_size == 0) {
            // current graph is empty, create a new graph.

            // allocate memory for the states
            cudaMalloc(&d_states_in_graph, data_size);

            // copy the states to the graph
            cudaMemcpy(d_states_in_graph, single_arm_states->getJointStatesCuda(), data_size, cudaMemcpyDeviceToDevice);

            // create graph by adding the states
            for (int i = 0; i < single_arm_states->getNumOfStates(); i++) {
                BoostVertex v = add_vertex(graph);
                graph[v].location = i;
                graph[v].group = group;
            }
        }
        else{
            // current graph is not empty, we need to extend the d_states_in_graph.
            int d_new_states_in_graph_bytes = (current_graph_size + single_arm_states->getNumOfStates()) * single_arm_states->getNumOfJoints() * sizeof(float);

            float * d_new_states_in_graph;

            // allocate memory for the new states
            cudaMalloc(&d_new_states_in_graph, d_new_states_in_graph_bytes);

            // copy the old states to the new states
            cudaMemcpy(d_new_states_in_graph, d_states_in_graph, current_graph_size * single_arm_states->getNumOfJoints() * sizeof(float), cudaMemcpyDeviceToDevice);

            // copy the new states to the new states
            cudaMemcpy(d_new_states_in_graph + current_graph_size * single_arm_states->getNumOfJoints(), single_arm_states->getJointStatesCuda(), single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints() * sizeof(float), cudaMemcpyDeviceToDevice);

            // free the old states
            cudaFree(d_states_in_graph);

            // update the d_states_in_graph
            d_states_in_graph = d_new_states_in_graph;

            // create graph by adding the states
            for (int i = 0; i < single_arm_states->getNumOfStates(); i++) {
                BoostVertex v = add_vertex(graph);
                graph[v].location = i + current_graph_size;
                graph[v].group = group;
            }
        }
    }

    void SingleArmGraph::add_states(const BaseStatesPtr & states) {
        // Add the states to the graph
        add_states_helper(states, 0);
    }

    void SingleArmGraph::add_start_states(const BaseStatesPtr & states){
        // Add the start states to the graph
        add_states_helper(states, 1);
    }

    void SingleArmGraph::add_goal_states(const BaseStatesPtr & states){
        // Add the goal states to the graph
        add_states_helper(states, 2);
    }

    BaseMotionsPtr SingleArmGraph::get_motions_to_k_nearest_neighbors(const BaseStatesPtr & states, int k, std::vector<StateIndexPair> & pairs){

        auto space_info = states->getSpaceInfo();
        // static cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(space_info);

        // static cast the states to SingleArmStates
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);
        float * d_joint_states = single_arm_states->getJointStatesCuda();
        
        // clear the pairs
        pairs.clear();

        // print();

        printf("=============debugging================\n");
        printf("Size of states: %d\n", states->getNumOfStates());
        printf("size of Graph: %d\n", get_size());

        // if k is greater than the number of states in the graph, then return motions to all the states in the graph
        if (k >= get_size()) {

            SingleArmMotionsPtr motions = std::make_shared<SingleArmMotions>(states->getNumOfStates() * get_size(), single_arm_space_info);
            float * d_joint_states_1 = motions->getJointStates1Cuda();
            float * d_joint_states_2 = motions->getJointStates2Cuda();
            
            // prepare the motions by numerating the states in the graph and states
            for (int i = 0; i < states->getNumOfStates(); i++){
                for (int j = 0; j < get_size(); j++){
                    cudaMemcpyAsync(
                        d_joint_states_1 + (i * get_size() + j) * num_of_joints,
                        d_joint_states + i * num_of_joints,
                        num_of_joints * sizeof(float),
                        cudaMemcpyDeviceToDevice
                    );
                    StateIndexPair pair;
                    pair.index_in_states = i;
                    pair.index_in_graph = j;
                    pairs.push_back(pair);
                }
                // copy the states in the graph to d_joint_states_2
                cudaMemcpyAsync(
                    d_joint_states_2 + i * get_size() * num_of_joints,
                    d_states_in_graph,
                    get_size() * num_of_joints * sizeof(float),
                    cudaMemcpyDeviceToDevice
                );
            }

            // wait for the copy to finish
            cudaDeviceSynchronize();

            return motions;
        }

        // if k is less than the number of states in the graph, then return motions to k nearest neighbors
        SingleArmMotionsPtr motions = std::make_shared<SingleArmMotions>(k * states->getNumOfStates(), single_arm_space_info);
        float * d_joint_states_1 = motions->getJointStates1Cuda();
        float * d_joint_states_2 = motions->getJointStates2Cuda();

        float * d_distances_from_states_to_graph;
        cudaMalloc(&d_distances_from_states_to_graph, states->getNumOfStates() * get_size() * sizeof(float));

        // calculate the distance between the states and the states in the graph
        int block_size = 256;
        int grid_size = (states->getNumOfStates() * get_size() + block_size - 1) / block_size;

        calculate_distance<<<grid_size, block_size>>>(
            d_joint_states, states->getNumOfStates(),
            d_states_in_graph, get_size(),
            num_of_joints, single_arm_space_info->d_active_joint_map, d_distances_from_states_to_graph
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        std::vector<std::vector<float>> distances_from_states_to_graph(states->getNumOfStates(), std::vector<float>(get_size()));
        std::vector<float> distances_from_states_to_graph_flattened(states->getNumOfStates() * get_size());

        // copy the distances from the states to the graph to the host
        cudaMemcpy(distances_from_states_to_graph_flattened.data(), d_distances_from_states_to_graph, states->getNumOfStates() * get_size() * sizeof(float), cudaMemcpyDeviceToHost);

        // convert the flattened distances to 2D distances
        for (int i = 0; i < states->getNumOfStates(); i++) {
            for (int j = 0; j < get_size(); j++) {
                distances_from_states_to_graph[i][j] = distances_from_states_to_graph_flattened[i * get_size() + j];
                printf("%f ", distances_from_states_to_graph[i][j]);
            }
            printf("\n");
        }

        // free the memory
        cudaFree(d_distances_from_states_to_graph);

        return motions;
    }

    void SingleArmGraph::contain(const BaseStatesPtr & states, int * d_result) const {
        // Check if the states are in the graph
    }

    void SingleArmGraph::connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, float * d_weight) {
        // Connect the states by adding an edge
    }

    void SingleArmGraph::is_connect(const BaseStatesPtr & states1, const BaseStatesPtr & states2, int * d_result) const {
        // Check if the states are connected
    }

    void SingleArmGraph::print() const {
        // Print the graph
        for (auto v : boost::make_iterator_range(vertices(graph))) {
            printf("Location: %d, Group: %d ", graph[v].location, graph[v].group);

            std::vector<float> joint_states(num_of_joints);
            cudaMemcpy(joint_states.data(), d_states_in_graph + graph[v].location * num_of_joints, num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < num_of_joints; i++) {
                printf("%f ", joint_states[i]);
            }
            printf("\n");
        }
    }

} // namespace CUDAMPLib