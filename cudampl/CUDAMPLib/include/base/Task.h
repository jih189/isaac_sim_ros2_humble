#pragma once

#include <memory>
#include <base/States.h>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            virtual ~BaseTask() {}
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs