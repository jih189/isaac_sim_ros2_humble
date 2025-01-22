#pragma once

#include <States.h>

namespace CUDAMPLib
{
    class BaseMotions
    {
        public:
            virtual ~BaseMotions() {}

            float * getCosts() {
                return d_costs;
            }

            void setCosts(float * d_costs) {
                this->d_costs = d_costs;
            }

        private:
            float * d_costs; // costs of each motion
    };

    typedef std::shared_ptr<BaseMotions> BaseMotionsPtr;
} // namespace CUDAMPLibs