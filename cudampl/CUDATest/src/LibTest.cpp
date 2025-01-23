#include <iostream>

#include <planners/RRG.h>

int main()
{
    CUDAMPLib::RRGPtr rrg_planner = std::make_shared<CUDAMPLib::RRG>();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}