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


    void EnvConstraint::computeCost(BaseStatesPtr states, float* d_cost)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        // // print information
        // std::cout << "computeCost" << std::endl;
        // std::cout << "Number of states: " << states->getNumOfStates() << std::endl;
        // SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());

        // // Get the self collision spheres
        // std::cout << "num_of_self_collision_spheres: " << space_info->num_of_self_collision_spheres << std::endl;

        // // Print bounds
        // std::cout << "Lower bound: ";
        // for (int i = 0; i < space_info->num_of_joints; i++){
        //     std::cout << space_info->lower_bound[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "Upper bound: ";
        // for (int i = 0; i < space_info->num_of_joints; i++){
        //     std::cout << space_info->upper_bound[i] << " ";
        // }
        // std::cout << std::endl;


        // computeCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
        //     d_self_collision_spheres_pos_in_base_link,
        //     d_self_collision_spheres_radius,
        //     num_of_self_collision_spheres,
        //     num_of_configurations,
        //     d_env_collision_spheres_pos_in_base_link,
        //     d_env_collision_spheres_radius,
        //     num_of_env_collision_spheres,
        //     d_cost
        // );
        
    }

    void EnvConstraint::computeCost(BaseMotionsPtr motions, float* d_cost)
    {

    }
        

} // namespace CUDAMPLib