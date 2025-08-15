cmake_minimum_required(VERSION 3.16)
project(MCLProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

include(FetchContent)

# lapackpp を自動で取得
FetchContent_Declare(
    blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG        v2025.05.28
)

FetchContent_MakeAvailable(blaspp)

add_library(MCLlib ${SOURCES})

target_include_directories(MCLlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MCLlib PUBLIC blaspp)
