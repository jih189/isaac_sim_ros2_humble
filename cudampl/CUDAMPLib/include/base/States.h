#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

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
        @brief A base state class which has no any member variables.
    */
    class BaseStates
    {
        public:
            BaseStates(int num_of_states, SpaceInfoPtr space_info) {

                this->num_of_states = num_of_states;
                this->space_info = space_info;

                // Allocate memory for the costs
                cudaMalloc(&d_costs, num_of_states * space_info->num_of_constraints * sizeof(float));
                cudaMalloc(&d_total_costs, num_of_states * sizeof(float));
            }

            ~BaseStates() {
                // Free the memory
                cudaFree(d_costs);
                cudaFree(d_total_costs);
            }

            int getNumOfStates() const { return num_of_states; }

            float * getCostsCuda() {
                return d_costs;
            }

            float * getTotalCostsCuda() {
                return d_total_costs;
            }

            void calculateTotalCosts();

            std::vector<std::vector<float>> getCostsHost() {
                std::vector<std::vector<float>> costs_host(num_of_states, std::vector<float>(space_info->num_of_constraints, 0.0));
                std::vector<float> costs_host_flatten(num_of_states * space_info->num_of_constraints, 0.0);
                cudaMemcpy(costs_host_flatten.data(), d_costs, num_of_states * space_info->num_of_constraints * sizeof(float), cudaMemcpyDeviceToHost);

                for (int i = 0; i < num_of_states; i++) {
                    for (int j = 0; j < space_info->num_of_constraints; j++) {
                        costs_host[i][j] = costs_host_flatten[j * num_of_states + i];
                    }
                }

                return costs_host;
            }

            std::vector<float> getTotalCostsHost() {
                std::vector<float> total_costs_host(num_of_states, 0.0);
                cudaMemcpy(total_costs_host.data(), d_total_costs, num_of_states * sizeof(float), cudaMemcpyDeviceToHost);
                return total_costs_host;
            }

            /**
                Based on the current states, update robot information and states.
             */
            virtual void update() = 0;

            SpaceInfoPtr getSpaceInfo() const { return space_info; }

        protected:
            int num_of_states;
            float * d_costs; // cost of each state and different constraints
            float * d_total_costs; // total cost of each state
            SpaceInfoPtr space_info;
    };
    typedef std::shared_ptr<BaseStates> BaseStatesPtr;
} // namespace CUDAMPLibs