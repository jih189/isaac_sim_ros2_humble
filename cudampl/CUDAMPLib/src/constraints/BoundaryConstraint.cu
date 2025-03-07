#include <constraints/BoundaryConstraint.h>

#include <chrono>

namespace CUDAMPLib{
    BoundaryConstraint::BoundaryConstraint(
        const std::string& constraint_name,
        const std::vector<float>& lower_bound,
        const std::vector<float>& upper_bound,
        const std::vector<bool>& active_joint_map
    ) : BaseConstraint(constraint_name, true) // This constraint is projectable.
    {
        this->constraint_name = constraint_name;

        // Convert active_joint_map to int
        std::vector<int> active_joint_map_as_int(active_joint_map.size());
        // Use std::transform to convert each bool to int
        std::transform(active_joint_map.begin(), active_joint_map.end(), active_joint_map_as_int.begin(),
            [](bool b) { return b ? 1 : 0; });

        // Construct the boundary for full joints include non-active joints
        std::vector<float> lower_bound_full(active_joint_map.size());
        std::vector<float> upper_bound_full(active_joint_map.size());

        int active_joint_index = 0;
        for (size_t i = 0; i < active_joint_map.size(); i++)
        {
            if (active_joint_map[i])
            {
                lower_bound_full[i] = lower_bound[active_joint_index];
                upper_bound_full[i] = upper_bound[active_joint_index];
                active_joint_index++;
            }
            else
            {
                // Set to 0.0 for non-active joints
                lower_bound_full[i] = 0.0; 
                upper_bound_full[i] = 0.0;
            }
        }

        size_t active_joint_map_as_int_bytes = active_joint_map_as_int.size() * sizeof(int);
        size_t lower_bound_full_bytes = lower_bound_full.size() * sizeof(float);
        size_t upper_bound_full_bytes = upper_bound_full.size() * sizeof(float);

        // Allocate memory on device
        cudaMalloc(&d_active_joint_map_as_int_, active_joint_map_as_int_bytes);
        cudaMalloc(&d_lower_bound_full_, lower_bound_full_bytes);
        cudaMalloc(&d_upper_bound_full_, upper_bound_full_bytes);

        // Copy data to device
        cudaMemcpy(d_active_joint_map_as_int_, active_joint_map_as_int.data(), active_joint_map_as_int_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower_bound_full_, lower_bound_full.data(), lower_bound_full_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_upper_bound_full_, upper_bound_full.data(), upper_bound_full_bytes, cudaMemcpyHostToDevice);
    }

    BoundaryConstraint::~BoundaryConstraint()
    {
        cudaFree(d_active_joint_map_as_int_);
        cudaFree(d_lower_bound_full_);
        cudaFree(d_upper_bound_full_);
    }

    __global__ void computeBoundaryCostKernel(
        float * d_states,
        int * d_active_joint_map_as_int,
        float * d_lower_bound_full,
        float * d_upper_bound_full,
        int num_of_states,
        int num_of_joints,
        float * d_cost_of_current_constraint
    )
    {
        int state_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (state_id >= num_of_states)
        {
            return;
        }

        float cost = 0.0;
        // Check if the joint is active
        for (int joint_id = 0; joint_id < num_of_joints; joint_id++)
        {
            if (d_active_joint_map_as_int[joint_id] == 1) // Active joint
            {
                float joint_value = d_states[state_id * num_of_joints + joint_id];
                float lower_bound = d_lower_bound_full[joint_id];
                float upper_bound = d_upper_bound_full[joint_id];

                if (joint_value < lower_bound)
                {
                    cost += (lower_bound - joint_value) * (lower_bound - joint_value);
                }
                else if (joint_value > upper_bound)
                {
                    cost += (joint_value - upper_bound) * (joint_value - upper_bound);
                }
            }
        }

        d_cost_of_current_constraint[state_id] = sqrtf(cost);
    }

    void BoundaryConstraint::computeCost(BaseStatesPtr states)
    {
        // Cast the states and space information for SingleArmSpace
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // check the cost location of this constraint
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1){
            // raise an error
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Get the number of states
        int num_of_states = single_arm_states->getNumOfStates();

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states + threadsPerBlock - 1) / threadsPerBlock;

        // Call the kernel
        computeBoundaryCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getJointStatesCuda(),
            d_active_joint_map_as_int_,
            d_lower_bound_full_,
            d_upper_bound_full_,
            num_of_states,
            space_info->num_of_joints,
            d_cost_of_current_constraint
        );

        cudaDeviceSynchronize();
    }

    void BoundaryConstraint::computeCostLarge(BaseStatesPtr states)
    {
        // Let it be here for now
        computeCost(states);
    }

    void BoundaryConstraint::computeCostFast(BaseStatesPtr states)
    {
        // Let it be here for now
        computeCost(states);
    }

    __global__ void computeGradientErrorKernel(
        float * d_states,
        int * d_active_joint_map_as_int,
        float * d_lower_bound_full,
        float * d_upper_bound_full,
        int num_of_states,
        int num_of_joints,
        float * d_cost_of_current_constraint,
        float * d_grad_of_current_constraint
    )
    {
        int state_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (state_id >= num_of_states)
        {
            return;
        }

        float cost = 0.0;

        // Check if the joint is active
        for (int joint_id = 0; joint_id < num_of_joints; joint_id++)
        {
            if (d_active_joint_map_as_int[joint_id] == 1) // Active joint
            {
                float joint_value = d_states[state_id * num_of_joints + joint_id];
                float lower_bound = d_lower_bound_full[joint_id];
                float upper_bound = d_upper_bound_full[joint_id];

                if (joint_value < lower_bound)
                {
                    d_grad_of_current_constraint[state_id * num_of_joints + joint_id] = lower_bound - joint_value;
                    cost += (lower_bound - joint_value) * (lower_bound - joint_value);
                }
                else if (joint_value > upper_bound)
                {
                    d_grad_of_current_constraint[state_id * num_of_joints + joint_id] = upper_bound - joint_value;
                    cost += (joint_value - upper_bound) * (joint_value - upper_bound);
                }
                else
                {
                    d_grad_of_current_constraint[state_id * num_of_joints + joint_id] = 0.0;
                }
            }
            else
            {
                d_grad_of_current_constraint[state_id * num_of_joints + joint_id] = 0.0;
            }
        }

        d_cost_of_current_constraint[state_id] = sqrtf(cost);
    }

    void BoundaryConstraint::computeGradientAndError(BaseStatesPtr states)
    {
        // Cast the states and space information for SingleArmSpace
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // check the cost location of this constraint
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1){
            // raise an error
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        float * d_grad_of_current_constraint = &(single_arm_states->getGradientCuda()[single_arm_states->getNumOfStates() * space_info->num_of_joints * constraint_index]);
        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Get the number of states
        int num_of_states = single_arm_states->getNumOfStates();

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states + threadsPerBlock - 1) / threadsPerBlock;

        // Call the kernel
        computeGradientErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getJointStatesCuda(),
            d_active_joint_map_as_int_,
            d_lower_bound_full_,
            d_upper_bound_full_,
            num_of_states,
            space_info->num_of_joints,
            d_cost_of_current_constraint,
            d_grad_of_current_constraint
        );

        cudaDeviceSynchronize();
    }
} // namespace CUDAMPLib