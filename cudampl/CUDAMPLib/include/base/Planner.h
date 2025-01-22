#pragma once
#include <Task.h>

class BasePlanner
{
    public:
        virtual ~BasePlanner() {}
        virtual void setMotionTask(BaseTaskPtr task) = 0;
        virtual void solve() = 0;
};