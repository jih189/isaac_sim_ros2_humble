#include <constraints/EnvConstraint.h>

namespace CUDAMPLib{

    EnvConstraint::EnvConstraint(
        const std::string& constraint_name,
        const std::vector<std::vector<float>>& env_collision_spheres_pos,
        const std::vector<float>& env_collision_spheres_radius
    )
    : BaseConstraint(constraint_name)
    {
        // Prepare the cuda memory for the collision cost
        num_of_env_collision_spheres = env_collision_spheres_pos.size();

        // Allocate memory for the environment collision spheres
        int env_collision_spheres_pos_bytes = num_of_env_collision_spheres * sizeof(float) * 3;
        int env_collision_spheres_radius_bytes = num_of_env_collision_spheres * sizeof(float);

        cudaMalloc(&d_env_collision_spheres_pos_in_base_link, env_collision_spheres_pos_bytes);
        cudaMalloc(&d_env_collision_spheres_radius, env_collision_spheres_radius_bytes);

        // Copy the environment collision spheres to the device
        cudaMemcpy(d_env_collision_spheres_pos_in_base_link, floatVectorFlatten(env_collision_spheres_pos).data(), env_collision_spheres_pos_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_env_collision_spheres_radius, env_collision_spheres_radius.data(), env_collision_spheres_radius_bytes, cudaMemcpyHostToDevice);
    }

    EnvConstraint::~EnvConstraint()
    {
        cudaFree(d_env_collision_spheres_pos_in_base_link);
        cudaFree(d_env_collision_spheres_radius);
    }

    __global__ void computeCollisionCostKernel(
        float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        float* d_obstacle_sphere_pos_in_base_link, // num_of_obstacle_spheres x 3
        float* d_obstacle_sphere_radius, // num_of_obstacle_spheres
        int num_of_obstacle_collision_spheres,
        float* d_cost // num_of_configurations
    )
    {
        // Get the index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations){
            float cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){ // For each self collision sphere
                for (int j = 0; j < num_of_obstacle_collision_spheres; j++){ // For each obstacle sphere

                    float diff_in_x = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 0] - d_obstacle_sphere_pos_in_base_link[j * 3 + 0];
                    float diff_in_y = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 1] - d_obstacle_sphere_pos_in_base_link[j * 3 + 1];
                    float diff_in_z = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 2] - d_obstacle_sphere_pos_in_base_link[j * 3 + 2];

                    float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
                    float sum_of_radius = d_self_collision_spheres_radius[i] + d_obstacle_sphere_radius[j];

                    // the cost the overlap of the two spheres
                    cost += fmaxf(0.0f, sum_of_radius - distance);
                }
            }
            d_cost[idx] = cost;
        }
    }

    

    __global__ void newComputeCollisionCostKernel(
        float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        float* d_obstacle_sphere_pos_in_base_link, // num_of_obstacle_spheres x 3
        float* d_obstacle_sphere_radius, // num_of_obstacle_spheres
        int num_of_obstacle_collision_spheres,
        float* d_cost // num_of_configurations x num_of_self_collision_spheres x num_of_obstacle_collision_spheres
    )
    {
        // Get the index of the thread
        int state_index = threadIdx.x + blockIdx.x * blockDim.x;
        int self_collision_sphere_index = threadIdx.y + blockIdx.y * blockDim.y;
        int obstacle_collision_sphere_index = threadIdx.z + blockIdx.z * blockDim.z;

        if (state_index < num_of_configurations && 
            self_collision_sphere_index < num_of_self_collision_spheres && 
            obstacle_collision_sphere_index < num_of_obstacle_collision_spheres){
            float diff_in_x = d_self_collision_spheres_pos_in_base_link[state_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 0] - d_obstacle_sphere_pos_in_base_link[obstacle_collision_sphere_index * 3 + 0];
            float diff_in_y = d_self_collision_spheres_pos_in_base_link[state_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 1] - d_obstacle_sphere_pos_in_base_link[obstacle_collision_sphere_index * 3 + 1];
            float diff_in_z = d_self_collision_spheres_pos_in_base_link[state_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 2] - d_obstacle_sphere_pos_in_base_link[obstacle_collision_sphere_index * 3 + 2];

            float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
            float sum_of_radius = d_self_collision_spheres_radius[self_collision_sphere_index] + d_obstacle_sphere_radius[obstacle_collision_sphere_index];

            // the cost the overlap of the two spheres
            d_cost[state_index * num_of_self_collision_spheres * num_of_obstacle_collision_spheres + self_collision_sphere_index * num_of_obstacle_collision_spheres + obstacle_collision_sphere_index] = fmaxf(0.0f, sum_of_radius - distance);
        }
    }

    __global__ void sumCollisionCostKernel(
        float* d_collision_cost,
        int num_of_states,
        int num_of_self_collision_spheres,
        int num_of_obstacle_collision_spheres,
        float* d_cost
    )
    {
        int state_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (state_index < num_of_states){
            float cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){
                for (int j = 0; j < num_of_obstacle_collision_spheres; j++){
                    cost += d_collision_cost[state_index * num_of_self_collision_spheres * num_of_obstacle_collision_spheres + i * num_of_obstacle_collision_spheres + j];
                }
            }
            d_cost[state_index] = cost;
        }
    }

    void EnvConstraint::computeCost(BaseStatesPtr states)
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

        int threadsPerBlock = 256;
        int blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        computeCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
            space_info->d_self_collision_spheres_radius, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            d_env_collision_spheres_pos_in_base_link, 
            d_env_collision_spheres_radius, 
            num_of_env_collision_spheres, 
            d_cost_of_current_constraint 
        );

        cudaDeviceSynchronize();
    }

    void EnvConstraint::newComputeCost(BaseStatesPtr states)
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

        // allocate memory for 3d collision cost, num_of_configurations x num_of_self_collision_spheres x num_of_obstacle_collision_spheres
        float * d_collision_cost;
        cudaMalloc(&d_collision_cost, single_arm_states->getNumOfStates() * space_info->num_of_self_collision_spheres * num_of_env_collision_spheres * sizeof(float));

        dim3 blockDim(8, 8, 8);
        dim3 gridDim((single_arm_states->getNumOfStates() + blockDim.x - 1) / blockDim.x, 
                    (space_info->num_of_self_collision_spheres + blockDim.y - 1) / blockDim.y, 
                    (num_of_env_collision_spheres + blockDim.z - 1) / blockDim.z);

        newComputeCollisionCostKernel<<<gridDim, blockDim>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
            space_info->d_self_collision_spheres_radius, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            d_env_collision_spheres_pos_in_base_link, 
            d_env_collision_spheres_radius, 
            num_of_env_collision_spheres, 
            d_collision_cost 
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // sum the collision cost
        int threadsPerBlock = 256;
        int blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        sumCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_collision_cost,
            single_arm_states->getNumOfStates(),
            space_info->num_of_self_collision_spheres,
            num_of_env_collision_spheres,
            d_cost_of_current_constraint
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        cudaFree(d_collision_cost);
    }
} // namespace CUDAMPLib