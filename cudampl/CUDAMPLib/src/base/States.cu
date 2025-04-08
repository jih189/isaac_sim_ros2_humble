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

    std::vector<std::vector<float>> BaseStates::getCostsHost()
    {
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

    std::vector<float> BaseStates::getTotalCostsHost() {
        std::vector<float> total_costs_host(num_of_states_, 0.0);
        cudaMemcpy(total_costs_host.data(), d_total_costs, num_of_states_ * sizeof(float), cudaMemcpyDeviceToHost);
        return total_costs_host;
    }

} // namespace CUDAMPLib