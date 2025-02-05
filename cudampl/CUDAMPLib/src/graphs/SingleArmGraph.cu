#include "graphs/SingleArmGraph.h"

namespace CUDAMPLib
{
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

    BaseMotionsPtr SingleArmGraph::get_motions_to_k_nearest_neighbors(const BaseStatesPtr & states, int k, std::vector<StateIndexPair> pairs){

        printf("=============debugging================\n");
        printf("Size of states: %d\n", states->getNumOfStates());
        printf("size of Graph: %d\n", get_size());
        return nullptr;
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