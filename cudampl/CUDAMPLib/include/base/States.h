#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

namespace CUDAMPLib
{

    /**
        @brief The information of the space.
        This is a struct to store the information of the space, so we can pass this information to other objects with differenth class such as states, constraints.
        Later, those objects can be aware of the information of the space.
     */
    struct SpaceInfo 
    {
        int dim;
        int num_of_constraints;
        std::vector<std::string> constraint_names;
    };
    typedef std::shared_ptr<SpaceInfo> SpaceInfoPtr;

    /**
        @brief A base state class used to represent the states in the space.
    */
    class BaseStates
    {
        public:
            BaseStates(int num_of_states, SpaceInfoPtr space_info) : num_of_states_(num_of_states), is_valid_(true)
            {
                this->space_info = space_info;

                // Allocate memory for the costs
                size_t d_costs_bytes = (size_t)num_of_states_ * space_info->num_of_constraints * sizeof(float);
                size_t d_total_costs_bytes = (size_t)num_of_states_ * sizeof(float);
                auto allocate_result = cudaMalloc(&d_costs, d_costs_bytes);
                if (allocate_result != cudaSuccess) {
                    cudaGetLastError();
                    std::cerr << "Error allocating memory for d_costs: " << cudaGetErrorString(allocate_result) << std::endl;
                    setValid(false);
                    return;
                }
                allocate_result = cudaMalloc(&d_total_costs, d_total_costs_bytes);
                if (allocate_result != cudaSuccess) {
                    cudaGetLastError();
                    std::cerr << "Error allocating memory for d_total_costs: " << cudaGetErrorString(allocate_result) << std::endl;
                    setValid(false);
                    return;
                }
            }

            virtual ~BaseStates() {
                if (num_of_states_ > 0)
                {
                    // Free the memory
                    cudaFree(d_costs);
                    cudaFree(d_total_costs);
                    // set the pointer to nullptr for safety
                    d_costs = nullptr;
                    d_total_costs = nullptr;
                }
            }

            void setValid(bool is_valid) { is_valid_ = is_valid; }
            bool isValid() const { return is_valid_; }

            virtual void filterStates(const std::vector<bool> & filter_map){

                // calculate the number of feasible states
                int num_left_states = std::count(filter_map.begin(), filter_map.end(), true);
                
                if (num_left_states == 0) {
                    // if there is no feasible states, clear the memory
                    cudaFree(d_costs);
                    cudaFree(d_total_costs);

                    // set the pointer to nullptr for safety
                    d_costs = nullptr;
                    d_total_costs = nullptr;
                }
                else{
                    // allocate memory for the feasible states
                    float * d_costs_new;
                    float * d_total_costs_new;

                    size_t d_costs_new_bytes = (size_t)num_left_states * space_info->num_of_constraints * sizeof(float);
                    size_t d_total_costs_new_bytes = (size_t)num_left_states * sizeof(float);

                    cudaMalloc(&d_costs_new, d_costs_new_bytes);
                    cudaMalloc(&d_total_costs_new, d_total_costs_new_bytes);

                    // copy the feasible states to the new memory
                    int j = 0;
                    for (int i = 0; i < num_of_states_; i++) {
                        if (filter_map[i]) {
                            for (int k = 0; k < space_info->num_of_constraints; k++) {
                                // cudaMemcpy(d_costs_new + j + k * num_left_states,
                                // d_costs + i + k * num_of_states_,
                                // sizeof(float), 
                                // cudaMemcpyDeviceToDevice);
                                // copy asynchonously
                                cudaMemcpyAsync(d_costs_new + j + k * num_left_states,
                                d_costs + i + k * num_of_states_,
                                sizeof(float),
                                cudaMemcpyDeviceToDevice);
                            }
                            // cudaMemcpy(d_total_costs_new + j, d_total_costs + i, sizeof(float), cudaMemcpyDeviceToDevice);
                            // copy asynchonously
                            cudaMemcpyAsync(d_total_costs_new + j, d_total_costs + i, sizeof(float), cudaMemcpyDeviceToDevice);
                            j++;
                        }
                    }

                    // wait for the copy to finish
                    cudaDeviceSynchronize();

                    // free the old memory
                    cudaFree(d_costs);
                    cudaFree(d_total_costs);

                    // update the memory
                    d_costs = d_costs_new;
                    d_total_costs = d_total_costs_new;
                }

                num_of_states_ = num_left_states;
            }

            int getNumOfStates() const { return num_of_states_; }

            float * getCostsCuda() {
                return d_costs;
            }

            float * getTotalCostsCuda() {
                return d_total_costs;
            }

            void calculateTotalCosts();

            virtual void calculateTotalGradientAndError(const std::vector<int> & constraint_indexs)
            {
                // raise an error
                throw std::runtime_error("The function calculateTotalGradientAndError is not implemented.");
            }

            std::vector<std::vector<float>> getCostsHost() {

                std::vector<std::vector<float>> costs_host(num_of_states_, std::vector<float>(space_info->num_of_constraints, 0.0));
                std::vector<float> costs_host_flatten(num_of_states_ * space_info->num_of_constraints, 0.0);
                cudaMemcpy(costs_host_flatten.data(), d_costs, num_of_states_ * space_info->num_of_constraints * sizeof(float), cudaMemcpyDeviceToHost);

                for (int i = 0; i < num_of_states_; i++) {
                    for (int j = 0; j < space_info->num_of_constraints; j++) {
                        costs_host[i][j] = costs_host_flatten[j * num_of_states_ + i];
                    }
                }

                return costs_host;
            }

            std::vector<float> getTotalCostsHost() {
                std::vector<float> total_costs_host(num_of_states_, 0.0);
                cudaMemcpy(total_costs_host.data(), d_total_costs, num_of_states_ * sizeof(float), cudaMemcpyDeviceToHost);
                return total_costs_host;
            }

            /**
                @brief Based on the current states, update robot information and states.
             */
            virtual void update() = 0;

            /**
                @brief Print the states.
             */
            virtual void print() const = 0;

            SpaceInfoPtr getSpaceInfo() const { return space_info; }

        protected:
            int num_of_states_;
            float * d_costs; // cost of each state and different constraints. The format should be [state1_constraint1, state2_constraint1, ..., state1_constraint2, state2_constraint2, ...]
            float * d_total_costs; // total cost of each state
            bool is_valid_;
            SpaceInfoPtr space_info;
    };
    typedef std::shared_ptr<BaseStates> BaseStatesPtr;

    /**
        @brief A base manager class to keep track of states in gpu memory.
        User can use integer as index to locate the states.
        This manager should only keep track of data to identify the states such as joint values.
        We do not want to keep track of other information such as link poses, joint poses, joint axes, etc.
     */
    class BaseStateManager {
        public:
            // Default constructor.
            BaseStateManager(SpaceInfoPtr space_info) : space_info_(space_info) {}
            virtual ~BaseStateManager() {
                num_of_states_ = 0;
            }

            virtual void clear() {
                num_of_states_ = 0;
            }

            // Adds states and returns the index of the states in the manager.
            virtual std::vector<int> add_states(const BaseStatesPtr & states) = 0;

            // Returns the number of states in the manager.
            int get_num_of_states() const { return num_of_states_; }

            /**
                @brief For each query state, find its k nearest neighbors for each group.
             */
            virtual int find_k_nearest_neighbors(
                int k, const BaseStatesPtr & query_states, 
                const std::vector<std::vector<int>> & group_indexs,
                std::vector<std::vector<int>> & neighbors_index // output
            ) = 0;

            /**
                @brief For each query state, find its nearest neighbor for each group.
             */
            virtual void find_the_nearest_neighbors(
                const BaseStatesPtr & query_states, 
                const std::vector<std::vector<int>> & group_indexs, 
                std::vector<std::vector<int>> & neighbors_index // output
            ) = 0;

            /**
                @brief Given indexs of states, find the states in the manager.
             */
            virtual BaseStatesPtr get_states(const std::vector<int> & states_index) = 0;

            /**
                @brief Concatinate the multiple states into one states.
             */
            virtual BaseStatesPtr concatinate_states(const std::vector<BaseStatesPtr> & states) = 0;

        protected:
            int num_of_states_; // number of states in manager
            SpaceInfoPtr space_info_;
    };
    typedef std::shared_ptr<BaseStateManager> BaseStateManagerPtr;

} // namespace CUDAMPLibs