#pragma once

#include <base/Constraint.h>
#include <base/States.h>

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
            BaseSpace(size_t dim, std::vector<BaseConstraintPtr> constraints) : dim(dim), constraints_(constraints) {
                for (size_t i = 0; i < constraints.size(); i++) {
                    if (constraints[i]->isProjectable()) {
                        projectable_constraint_indices_.push_back(i);
                    }
                }
            }

            virtual ~BaseSpace() {}

            size_t getDim() const { return dim; }

            /**
                @brief Sample a set of states.
                @param num_of_config The number of states to sample.
                @return A set of states.
            */
            virtual BaseStatesPtr sample(int num_of_config) = 0;

            /**
                @brief Given two sets of states, check the feasibility of motions between them.
                @param states1 The first set of states.
                @param states2 The second set of states.
                @param motion_feasibility The feasibility of the motions.
                @param motion_costs The costs of the motions.
                @return True if the check motions feasible.
             */
            virtual bool checkMotions(
                const BaseStatesPtr & states1, 
                const BaseStatesPtr & states2, 
                std::vector<bool>& motion_feasibility,
                std::vector<float>& motion_costs
            ) = 0;

            /**
                @brief Get path from waypoints.
                @param waypoints The waypoints.
                @return The path.
             */
            virtual BaseStatesPtr getPathFromWaypoints(
                const BaseStatesPtr & waypoints
            ) = 0;

            /**
                @brief Check the feasibility of a set of states.
                @param states The states to check.
                @param state_feasibility The feasibility of the states as output.
            */
            virtual void checkStates(
                const BaseStatesPtr & states,
                std::vector<bool> & state_feasibility
            ) = 0;

            /**
                @brief Check the feasibility of a set of states.
                @param states The states to check.
             */
            virtual void checkStates(const BaseStatesPtr & states) = 0;

            /**
                @brief interpolate between two states. If the distance between two states is larger than max_distance, 
                        it will interpolate between two states with max_distance. Then the to_states will be updated.
             */
            virtual void interpolate(
                const BaseStatesPtr & from_states,
                const BaseStatesPtr & to_states,
                float max_distance
            ) = 0;

            /**
                @brief Project the states to satisfy the constraints. This project operation is done in-place.
                @param states The states to project.
             */
            virtual void projectStates(BaseStatesPtr states)
            {
                // raise an exception if the function is not implemented
                throw std::runtime_error("The function projectStates is not implemented.");
            }

            /**
                @brief Create a state manager.
                @return The state manager.
             */
            virtual BaseStateManagerPtr createStateManager() = 0;

            /** 
                @brief Get space information.
                @param space_info The space information as output.
             */
            void getSpaceInfo(SpaceInfoPtr space_info) {
                space_info->dim = dim;
                space_info->num_of_constraints = constraints_.size();
                for (const auto & constraint : constraints_) {
                    space_info->constraint_names.push_back(constraint->getName());
                }
            }

        protected:
            size_t dim;
            std::vector<BaseConstraintPtr> constraints_;
            std::vector<int> projectable_constraint_indices_;
    };

    typedef std::shared_ptr<BaseSpace> BaseSpacePtr;
} // namespace CUDAMPLib