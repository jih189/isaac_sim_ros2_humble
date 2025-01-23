#pragma once

#include <memory>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            virtual ~BaseTask() {}
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs