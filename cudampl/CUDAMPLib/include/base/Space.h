#pragma once

#include <base/Constraint.h>
#include <vector>

namespace CUDAMPLib
{
    // A base space class
    class BaseSpace {
        public:
            BaseSpace(int dim) : dim(dim) {}

            virtual ~BaseSpace() {}

            int getDim() const { return dim; }

            void setBounds(const std::vector<float>& lower, const std::vector<float>& upper) {
                this->lower = lower;
                this->upper = upper;
            }

            const std::vector<float>& getLower() const { return lower; }

            const std::vector<float>& getUpper() const { return upper; }

            /**
                * @brief Sample a set of configurations.
                * @param num_of_config The number of configurations to sample.
                * @param samples The sampled configurations.
            */
            virtual void sample(int num_of_config, std::vector<std::vector<float>>& samples) = 0;

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
                const std::vector<std::vector<float>>& states,
                std::vector<bool>& state_feasibility
            ) = 0;

        private:
            int dim;
            std::vector<float> lower;
            std::vector<float> upper;
    };

    typedef std::shared_ptr<BaseSpace> BaseSpacePtr;
} // namespace CUDAMPLib