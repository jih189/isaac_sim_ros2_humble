#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace CUDAMPLib
{
    /**
        A base state class which has no any member variables.
    */
    class BaseStates
    {
        public:
            BaseStates(int num_of_states) {

                this->num_of_states = num_of_states;

                // Allocate memory for the costs
                cudaMalloc(&d_costs, num_of_states * sizeof(float));
            }

            ~BaseStates() {
                // Free the memory
                cudaFree(d_costs);
            }

            int getNumOfStates() const { return num_of_states; }

        protected:
            int num_of_states;
            float * d_costs; // cost of each state
    };

    typedef std::shared_ptr<BaseStates> BaseStatesPtr;
} // namespace CUDAMPLibs