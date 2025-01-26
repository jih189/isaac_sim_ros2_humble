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
            }

            ~BaseStates() {
                // Free the memory
                cudaFree(d_costs);
            }

            int getNumOfStates() const { return num_of_states; }

            float * getCostsCuda() {
                return d_costs;
            }

            /**
                Based on the current states, update robot information and states.
             */
            virtual void update() = 0;

            SpaceInfoPtr getSpaceInfo() const { return space_info; }

        protected:
            int num_of_states;
            float * d_costs; // cost of each state
            SpaceInfoPtr space_info;
    };
    typedef std::shared_ptr<BaseStates> BaseStatesPtr;
} // namespace CUDAMPLibs