cmake_minimum_required(VERSION 3.22.1)
project(foliation_planner LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Release)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_cmake
  Boost
  control_msgs
  Eigen3
  interactive_markers
  moveit_core
  moveit_msgs
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_common
  moveit_servo
  pluginlib
  rclcpp
  rclcpp_action
  rviz_visual_tools
  tf2_geometry_msgs
  tf2_ros
  moveit_visual_tools
  CUDA
  ompl
)

foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

find_package(CUDAMPLib REQUIRED)

moveit_package()

include_directories(
  include
  /usr/include/ompl-1.6
)

set(MOVEIT_LIB_NAME moveit_foliation_planner_plugin)

# planning plugin
add_library(${MOVEIT_LIB_NAME}
  SHARED
  src/foliation_interface.cpp
  src/foliation_planner_manager.cpp
  src/foliation_planning_context.cpp
  )

add_executable(foliation_example src/foliation_example.cpp)
ament_target_dependencies(foliation_example ${THIS_PACKAGE_INCLUDE_DEPENDS})

add_executable(cuda_test src/cuda_test.cpp)
target_link_libraries(cuda_test CUDAMPLib ompl)
ament_target_dependencies(cuda_test ${THIS_PACKAGE_INCLUDE_DEPENDS})

add_executable(create_motion_tasks src/create_motion_tasks.cpp)
ament_target_dependencies(create_motion_tasks ${THIS_PACKAGE_INCLUDE_DEPENDS})

add_executable(evaluate_cudampl src/evaluate_cudampl.cpp)
target_link_libraries(evaluate_cudampl CUDAMPLib)
ament_target_dependencies(evaluate_cudampl ${THIS_PACKAGE_INCLUDE_DEPENDS})

set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES VERSION "${${PROJECT_NAME}_VERSION}")
ament_target_dependencies(${MOVEIT_LIB_NAME} ${THIS_PACKAGE_INCLUDE_DEPENDS})


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${MOVEIT_LIB_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS foliation_example
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS cuda_test
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS create_motion_tasks
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS evaluate_cudampl
  DESTINATION lib/${PROJECT_NAME})

# Install config files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)
pluginlib_export_plugin_description_file(moveit_core foliation_planner_plugin_description.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()