set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 23)
project(MCL)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)