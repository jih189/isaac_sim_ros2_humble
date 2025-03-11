#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014
#include "cost.h"
#include <cuda_runtime.h>

CUDAMPLib::CollisionCost::CollisionCost(
    const std::vector<std::vector<float>>& env_collision_spheres_pos,
    const std::vector<float>& env_collision_spheres_radius
)
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

CUDAMPLib::CollisionCost::~CollisionCost()
{
    // Free the cuda memory for the collision cost
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

void CUDAMPLib::CollisionCost::computeCost(
    float *d_joint_values, 
    int num_of_configurations,
    float *d_self_collision_spheres_pos_in_base_link, 
    float *d_self_collision_spheres_radius,
    int num_of_self_collision_spheres,
    float *d_cost
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_configurations + threadsPerBlock - 1) / threadsPerBlock;

    computeCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_self_collision_spheres_pos_in_base_link,
        d_self_collision_spheres_radius,
        num_of_self_collision_spheres,
        num_of_configurations,
        d_env_collision_spheres_pos_in_base_link,
        d_env_collision_spheres_radius,
        num_of_env_collision_spheres,
        d_cost
    );

    cudaDeviceSynchronize();
}

CUDAMPLib::SelfCollisionCost::SelfCollisionCost(
    const std::vector<int>& robot_collision_spheres_map,
    const std::vector<std::vector<bool>>& robot_collision_enables_map
)
{
    // Prepare the cuda memory for the self collision cost
    num_of_self_collision_spheres_in_cost = robot_collision_spheres_map.size();
    num_of_robot_links = robot_collision_enables_map.size();

    // Allocate memory for the self collision spheres
    int self_collision_spheres_map_bytes = num_of_self_collision_spheres_in_cost * sizeof(int);
    int self_collision_enables_map_bytes = num_of_robot_links * num_of_robot_links * sizeof(int);

    cudaMalloc(&d_self_collision_spheres_map, self_collision_spheres_map_bytes);
    cudaMalloc(&d_self_collision_enables_map, self_collision_enables_map_bytes);

    // Copy the self collision spheres to the device
    cudaMemcpy(d_self_collision_spheres_map, robot_collision_spheres_map.data(), self_collision_spheres_map_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_self_collision_enables_map, boolMatrixFlatten(robot_collision_enables_map).data(), self_collision_enables_map_bytes, cudaMemcpyHostToDevice);
}

CUDAMPLib::SelfCollisionCost::~SelfCollisionCost()
{
    // Free the cuda memory for the self collision cost
    cudaFree(d_self_collision_spheres_map);
    cudaFree(d_self_collision_enables_map);
}

__global__ void computeSelfCollisionCostKernel(
    float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
    float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
    int num_of_self_collision_spheres,
    int num_of_configurations,
    int* d_self_collision_spheres_map, // num_of_self_collision_spheres
    int num_of_robot_links,
    int* d_self_collision_enables_map, // num_of_robot_links x num_of_robot_links
    float* d_cost // num_of_configurations
)
{
    // Get the index of the thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_of_configurations){
        float cost = 0.0f;
        for (int i = 0; i < num_of_self_collision_spheres; i++){ // For each self collision sphere
            for (int j = i + 1; j < num_of_self_collision_spheres; j++){ // For each self collision sphere
                // check if the two spheres are not in the same link
                int link_i = d_self_collision_spheres_map[i];
                int link_j = d_self_collision_spheres_map[j];
                if ( link_i != link_j){
                    // check if two links are enabled for collision
                    if (d_self_collision_enables_map[link_i * num_of_robot_links + link_j] != 0)
                    {
                        float diff_in_x = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 0] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 0];
                        float diff_in_y = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 1] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 1];
                        float diff_in_z = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 2] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 2];

                        float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
                        float sum_of_radius = d_self_collision_spheres_radius[i] + d_self_collision_spheres_radius[j];

                        // the cost the overlap of the two spheres
                        cost += fmaxf(0.0f, sum_of_radius - distance);
                    }
                }
            }
        }
        d_cost[idx] = cost;
    }
}

void CUDAMPLib::SelfCollisionCost::computeCost(
    float *d_joint_values, 
    int num_of_configurations,
    float *d_self_collision_spheres_pos_in_base_link, 
    float *d_self_collision_spheres_radius,
    int num_of_self_collision_spheres,
    float *d_cost
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_configurations + threadsPerBlock - 1) / threadsPerBlock;

    if (num_of_self_collision_spheres_in_cost != num_of_self_collision_spheres){
        throw std::runtime_error("The number of self collision spheres in the cost object is not equal to the number of self collision spheres in the input");
    }

    computeSelfCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_self_collision_spheres_pos_in_base_link,
        d_self_collision_spheres_radius,
        num_of_self_collision_spheres,
        num_of_configurations,
        d_self_collision_spheres_map,
        num_of_robot_links,
        d_self_collision_enables_map,
        d_cost
    );
    cudaDeviceSynchronize();
}