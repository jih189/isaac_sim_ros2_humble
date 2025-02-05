#pragma once

#include <base/States.h>

namespace CUDAMPLib
{
    class BaseMotions
    {
        public:
            BaseMotions(int num_of_motions, SpaceInfoPtr space_info) {
                this->num_of_motions = num_of_motions;
                this->space_info = space_info;

                // Allocate memory for the costs
                cudaMalloc(&d_costs, num_of_motions * sizeof(float));
            }

            virtual ~BaseMotions() {
                // Free the memory
                cudaFree(d_costs);
            }

            float * getCostsCuda() {
                return d_costs;
            }

            int getNumOfMotions() const {
                return num_of_motions;
            }

            /**
                @brief Print the motions.
             */
            virtual void print() const = 0;

        protected:
            float * d_costs; // costs of each motion
            int num_of_motions;
            SpaceInfoPtr space_info;
    };

    typedef std::shared_ptr<BaseMotions> BaseMotionsPtr;
} // namespace CUDAMPLibs