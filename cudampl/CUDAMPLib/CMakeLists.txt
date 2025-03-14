#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "CUDAMPLib")

file(GLOB_RECURSE src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_library(NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    NO_DEFAULT_PATH
)

if(NOT NVRTC_LIB)
    message(FATAL_ERROR "NVRTC library not found in ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
endif()

#Executable
if(${MY_BUILD_SHARED_LIBS})
	add_library(${lib_name} SHARED ${src_files} ${include_files})
else()
	add_library(${lib_name} STATIC ${src_files} ${include_files})
endif()

#target_include_directories(${lib_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(${lib_name}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${lib_name}>
        ${CUDA_INCLUDE_DIRS}  # Propagate CUDA include directories
)

#Pass the flags to the library
target_link_libraries( ${lib_name}
    PUBLIC ${NVRTC_LIB} ${CUDA_CUDA_LIBRARY}
	PUBLIC $<BUILD_INTERFACE:developer_flags>	
)

#--------------------------- install ---------------------------------------------

# Install the library
install(TARGETS ${lib_name}
    EXPORT ${lib_name}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install include directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/${lib_name})

# Export targets
install(EXPORT ${lib_name}Targets
    FILE ${lib_name}Config.cmake
    DESTINATION lib/cmake/${lib_name}
)