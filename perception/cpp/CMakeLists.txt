cmake_minimum_required(VERSION 3.16)
project(r1_perception_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)

add_executable(lowstate_shm lowstate_shm.cpp)
target_link_libraries(lowstate_shm unitree_sdk2 rt pthread)
