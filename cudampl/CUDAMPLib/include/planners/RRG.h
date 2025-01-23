#pragma once

#include <base/Planner.h>

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

    typedef std::shared_ptr<RRG> RRGPtr;
} // namespace CUDAMPLibs