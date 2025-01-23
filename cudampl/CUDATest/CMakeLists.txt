#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Ensure include files are properly used
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#GLOB source files 
file(GLOB_RECURSE CUDATest_src ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDATest.cu)

#GLOB source files 
file(GLOB_RECURSE LibTest_src ${CMAKE_CURRENT_SOURCE_DIR}/src/LibTest.cpp)

# Executable for CUDATest
add_executable(CUDATest ${CUDATest_src} ${include_files})

# Executable for LibTest
add_executable(LibTest ${LibTest_src} ${include_files})

#Libs linked to the executable
target_link_libraries( CUDATest   
	CUDAMPLib
	$<BUILD_INTERFACE:developer_flags>	
	gtest_main
)

#Libs linked to the executable
target_link_libraries( LibTest   
	CUDAMPLib
	$<BUILD_INTERFACE:developer_flags>	
	gtest_main
)