#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014
#include <constraints/EnvConstraint.h>
#include <cuda_runtime.h>

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

    void EnvConstraint::computeCost(BaseStatesPtr states)
    {
        
    }

    void EnvConstraint::computeCost(BaseMotionsPtr motions)
    {

    }
        

} // namespace CUDAMPLib