#pragma once

#include <base/Constraint.h>
#include <base/States.h>
#include <vector>

namespace CUDAMPLib
{
    // A base space class
    class BaseSpace {
        public:
            BaseSpace(size_t dim) : dim(dim) {}

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
                std::vector<bool>& state_feasibility
            ) = 0;

        protected:
            size_t dim;
    };

    typedef std::shared_ptr<BaseSpace> BaseSpacePtr;
} // namespace CUDAMPLib