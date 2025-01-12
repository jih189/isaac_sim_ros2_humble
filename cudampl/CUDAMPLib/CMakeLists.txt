#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "CUDAMPLib")

file(GLOB_RECURSE src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Collect all header files from the include directory
file(GLOB_RECURSE include_files
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

#Executable
if(${MY_BUILD_SHARED_LIBS})
	add_library(${lib_name} SHARED ${src_files} ${include_files})
else()
	add_library(${lib_name} STATIC ${src_files} ${include_files})
endif()

target_include_directories(${lib_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

#Pass the flags to the library
target_link_libraries( ${lib_name}
	PUBLIC $<BUILD_INTERFACE:developer_flags>	
)