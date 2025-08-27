set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 23)
project(MCLsamples)

file(GLOB_RECURSE 3D4R_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(3d4rsample ${3D4R_SOURCES})
target_link_libraries(3d4rsample PRIVATE MCLlib)