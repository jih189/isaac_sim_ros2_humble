#pragma once

#include <base/Planner.h>
#include <base/Graph.h>

namespace CUDAMPLib
{
    class RRG : public BasePlanner
    {
        public:
            RRG(BaseSpacePtr space);
            ~RRG() override;

            void setMotionTask(BaseTaskPtr task) override;
            void solve() override;
        private:
            BaseGraphPtr graph;

            // parameters
            int sample_attempts;
            int k;
    };

    typedef std::shared_ptr<RRG> RRGPtr;
} // namespace CUDAMPLibs