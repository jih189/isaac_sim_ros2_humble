#include <base/States.h>

namespace CUDAMPLib{

    __global__ void calculateTotalCostsKernel(
        const float* __restrict__ d_costs,
        int num_of_states,
        int num_of_constraints,
        float* __restrict__ d_total_costs
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_of_states)
            return;
        
        float total_cost = 0.0f;
        for (int i = 0; i < num_of_constraints; i++){
            total_cost += d_costs[num_of_states * i + idx];
        }
        d_total_costs[idx] = total_cost;
    }

    __global__ void filterStatesKernel(const int* d_filter,
                                   const int* d_prefix,
                                   const float* d_costs,
                                   float* d_costs_new,
                                   const float* d_total_costs,
                                   float* d_total_costs_new,
                                   int numStates,
                                   int numConstraints,
                                   int newNumStates)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numStates) {
            if (d_filter[i]) {  // state i is feasible (nonzero)
                int newIndex = d_prefix[i]; // new index for this state
                // For each constraint, copy the corresponding cost.
                for (int k = 0; k < numConstraints; k++) {
                    // In the source, each constraint is stored in a block of numStates.
                    // In the destination, each constraint is stored in a block of newNumStates.
                    d_costs_new[newIndex + k * newNumStates] = d_costs[i + k * numStates];
                }
                // Copy the total cost.
                d_total_costs_new[newIndex] = d_total_costs[i];
            }
        }
    }

    void BaseStates::calculateTotalCosts()
    {
        // Calculate the total cost of the states by summing the costs of all constraints in kernel

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;

        // Call the kernel to calculate the total cost
        calculateTotalCostsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_costs,
            num_of_states_,
            space_info->num_of_constraints,
            d_total_costs
        );
    }

} // namespace CUDAMPLib