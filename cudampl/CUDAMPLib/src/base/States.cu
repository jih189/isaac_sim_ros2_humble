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

    void BaseStates::cudaFilterStates(const std::vector<bool> & filter_map){

        // calculate the number of feasible states
        int num_left_states = std::count(filter_map.begin(), filter_map.end(), true);

        if (num_left_states == 0) {
            cudaFree(d_costs);
            cudaFree(d_total_costs);
            num_of_states_ = 0;
            return;
        }
        
        // Allocate new memory for the filtered data.
        float* d_costs_new = nullptr;
        float* d_total_costs_new = nullptr;
        int numConstraints = space_info->num_of_constraints;
        size_t d_costs_new_bytes = (size_t)num_left_states * numConstraints * sizeof(float);
        size_t d_total_costs_new_bytes = (size_t)num_left_states * sizeof(float);
        cudaMalloc(&d_costs_new, d_costs_new_bytes);
        cudaMalloc(&d_total_costs_new, d_total_costs_new_bytes);

        // Prepare the filter map on device.
        // We'll convert the bool filter_map to an int array (0 or 1).
        thrust::device_vector<int> d_filter(num_of_states_);
        for (int i = 0; i < num_of_states_; i++) {
            d_filter[i] = filter_map[i] ? 1 : 0;
        }

        // Compute the exclusive prefix sum on the filter.
        thrust::device_vector<int> d_prefix(num_of_states_);
        thrust::exclusive_scan(d_filter.begin(), d_filter.end(), d_prefix.begin());

        // Launch a kernel to compact the data in parallel.
        int threadsPerBlock = 256;
        int blocks = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;
        filterStatesKernel<<<blocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_filter.data()),
            thrust::raw_pointer_cast(d_prefix.data()),
            d_costs, 
            d_costs_new, 
            d_total_costs, 
            d_total_costs_new, 
            num_of_states_, 
            numConstraints,
            num_left_states);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            // Handle error appropriately by throwing an exception.
            throw std::runtime_error("Error in filterStatesKernel");
        }

        // Free old device memory.
        cudaFree(d_costs);
        cudaFree(d_total_costs);

        // Update pointers.
        d_costs = d_costs_new;
        d_total_costs = d_total_costs_new;
        num_of_states_ = num_left_states;
    }

} // namespace CUDAMPLib