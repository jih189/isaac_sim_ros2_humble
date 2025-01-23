#pragma once

#include <memory>

namespace CUDAMPLib
{
    /**
        A base state class which has no any member variables.
    */
    class BaseStates
    {
        public:
            virtual ~BaseStates() {}

            float * getCosts() {
                return d_costs;
            }

            void setCosts(float * d_costs) {
                this->d_costs = d_costs;
            }

        protected:
            int num_of_states;
            float * d_costs; // costs of each state
    };

    typedef std::shared_ptr<BaseStates> BaseStatesPtr;
} // namespace CUDAMPLibs