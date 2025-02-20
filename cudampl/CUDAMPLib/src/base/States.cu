#include <base/States.h>

namespace CUDAMPLib{

    __global__ void calculateTotalCostsKernel(
        float* d_costs,
        int num_of_states,
        int num_of_constraints,
        float* d_total_costs
    )
    {
        // Get the index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_states){
            float total_cost = 0.0f;
            for (int i = 0; i < num_of_constraints; i++){
                total_cost += d_costs[num_of_states * i + idx];
            }
            d_total_costs[idx] = total_cost;
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