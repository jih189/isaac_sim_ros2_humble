#pragma once

#include <base/Constraint.h>
#include <base/States.h>
#include <base/Graph.h>

namespace CUDAMPLib
{
    // A base space class
    class BaseSpace {
        public:
            /**
                Warning: We should set the constraints for the space with constructor only. It is
                bad idea to have function for adding constraints later. Because when we create the
                states in spaces, the states should be aware of the constraints. If we add constraints
                later, the states will not be aware of the new constraints. Or we have to update all
                generated states with the new constraints which is not efficient.
             */
            BaseSpace(size_t dim, std::vector<BaseConstraintPtr> constraints) : dim(dim), constraints(constraints) {}

            virtual ~BaseSpace() {}

            size_t getDim() const { return dim; }

            /**
                * @brief Sample a set of states.
                * @param num_of_config The number of states to sample.
                * @return A set of states.
            */
            virtual BaseStatesPtr sample(int num_of_config) = 0;

            /**
                * @brief Get the waypoints between a set of configuration pairs.
                * @param start The start configurations. A list
                * of configurations, each represented as a list of floats.
                * @param end The end configurations. A list
                * of configurations, each represented as a list of floats.
                * @param waypoints The waypoints between the start and end configurations.
                * @param motion_feasibility The feasibility of the waypoints.
            */
            virtual void getMotions(
                const std::vector<std::vector<float>>& start, 
                const std::vector<std::vector<float>>& end, 
                std::vector<std::vector<std::vector<float>>>& motions,
                std::vector<bool> motion_feasibility
            ) = 0;

            /**
                * @brief Check the feasibility of a set of motions.
                * @param start The start configurations. A list
                * of configurations, each represented as a list of floats.
                * @param end The end configurations. A list
                * of configurations, each represented as a list of floats.
                * @param motion_feasibility The feasibility of the motions.
            */
            virtual void checkMotions(
                const std::vector<std::vector<float>>& start, 
                const std::vector<std::vector<float>>& end, 
                std::vector<bool>& motion_feasibility
            ) = 0;

            /**
                * @brief Check the feasibility of a set of states.
                * @param states The states to check.
                * @param state_feasibility The feasibility of the states.
            */
            virtual void checkStates(
                const BaseStatesPtr & states,
                std::vector<bool> & state_feasibility
            ) = 0;

            /**
                * @brief Create a graph for the space.
                * @return The graph.
             */
            virtual BaseGraphPtr createGraph() = 0;

            /** 
                @brief Get space information.
                @return The space information.
             */
            void getSpaceInfo(SpaceInfoPtr space_info) {
                space_info->dim = dim;
                space_info->num_of_constraints = constraints.size();
                for (const auto & constraint : constraints) {
                    space_info->constraint_names.push_back(constraint->getName());
                }
            }

        protected:
            size_t dim;
            std::vector<BaseConstraintPtr> constraints;
    };

    typedef std::shared_ptr<BaseSpace> BaseSpacePtr;
} // namespace CUDAMPLib