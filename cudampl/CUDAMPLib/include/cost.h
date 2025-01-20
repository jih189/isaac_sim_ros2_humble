#pragma once

#include <memory>
#include "util.h"

namespace CUDAMPLib
{
    class CostBase
    {
        public:
        // virtual destructor. We need to define how to clean the cuda memory in the derived class.
        virtual ~CostBase() {}
        virtual void computeCost(
            float *d_joint_values, 
            int num_of_configurations,
            float *d_self_collision_spheres_pos_in_base_link,
            float *d_self_collision_spheres_radius,
            int num_of_self_collision_spheres,
            float *d_cost) = 0; 
    };

    class CollisionCost : public CostBase
    {
        public:
        CollisionCost(
            const std::vector<std::vector<float>>& env_collision_spheres_pos,
            const std::vector<float>& env_collision_spheres_radius
        );

        ~CollisionCost() override;

        void computeCost(
            float *d_joint_values, 
            int num_of_configurations,
            float *d_self_collision_spheres_pos_in_base_link, 
            float *d_self_collision_spheres_radius,
            int num_of_self_collision_spheres,
            float *d_cost) override;
    
        private:
        int num_of_env_collision_spheres;
        float *d_env_collision_spheres_pos_in_base_link;
        float *d_env_collision_spheres_radius;
    };

    // create shared pointer type for cost   
    typedef std::shared_ptr<CostBase> CostBasePtr;
    typedef std::shared_ptr<CollisionCost> CollisionCostPtr;

} // namespace CUDAMPLib