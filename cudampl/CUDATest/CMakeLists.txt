#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Ensure include files are properly used
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#GLOB source files 
file(GLOB_RECURSE CUDATest_src ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDATest.cu)
file(GLOB_RECURSE NVRTCTest_src ${CMAKE_CURRENT_SOURCE_DIR}/src/NVRTCTest.cu)

# Executable for CUDATest
add_executable(CUDATest ${CUDATest_src} ${include_files})

# Executable for NVRTCTest
add_executable(NVRTCTest ${NVRTCTest_src})

#Libs linked to the executable
target_link_libraries( CUDATest   
	CUDAMPLib
	$<BUILD_INTERFACE:developer_flags>	
	gtest_main
)

# Locate CUDA toolkit (and NVRTC library).
# This will find CUDA and set variables like CUDA_TOOLKIT_ROOT_DIR.
find_package(CUDA REQUIRED)

target_link_libraries(NVRTCTest PRIVATE nvrtc cuda)