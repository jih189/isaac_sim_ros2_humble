#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include <iostream>
#include <vector>

TEST(Test, lib)
{

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}