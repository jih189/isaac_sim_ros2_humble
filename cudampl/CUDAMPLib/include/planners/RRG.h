#pragma once

#include <Planner.h>

namespace CUDAMPLib
{
    class RRG : public BasePlanner
    {
        public:
            RRG();
            ~RRG() override;

            void setMotionTask(BaseTaskPtr task) override;
            void solve() override;
    };
} // namespace CUDAMPLibs