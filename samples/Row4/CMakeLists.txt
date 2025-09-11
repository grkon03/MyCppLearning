cmake_minimum_required(VERSION 3.16)
project(MCLsamples)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ROW4_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(row4sample ${ROW4_SOURCES})
target_link_libraries(row4sample PRIVATE MCLlib)