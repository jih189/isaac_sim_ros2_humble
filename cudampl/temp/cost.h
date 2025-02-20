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
        /**
            * @brief Construct a new Collision Cost object
            * 
            * @param env_collision_spheres_pos 
            * @param env_collision_spheres_radius
         */
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

    class SelfCollisionCost : public CostBase
    {
        /**
            * @brief Construct a new Self Collision Cost object
            *
            * @param robot_collision_spheres_map: a map which defines which link the collision sphere belongs to
            * @param robot_collision_enables_map: a map which defines whether the collision between two links is enabled
         */
        public:
        SelfCollisionCost(
            const std::vector<int>& robot_collision_spheres_map,
            const std::vector<std::vector<bool>>& robot_collision_enables_map
        );

        ~SelfCollisionCost() override;

        void computeCost(
            float *d_joint_values, 
            int num_of_configurations,
            float *d_self_collision_spheres_pos_in_base_link, 
            float *d_self_collision_spheres_radius,
            int num_of_self_collision_spheres,
            float *d_cost) override;
    
        private:
        int num_of_self_collision_spheres_in_cost;
        int num_of_robot_links;
        int *d_self_collision_spheres_map;
        int *d_self_collision_enables_map;
    };

    // create shared pointer type for cost   
    typedef std::shared_ptr<CostBase> CostBasePtr;
    typedef std::shared_ptr<CollisionCost> CollisionCostPtr;
    typedef std::shared_ptr<SelfCollisionCost> SelfCollisionCostPtr;

} // namespace CUDAMPLib