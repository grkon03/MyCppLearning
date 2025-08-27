cmake_minimum_required(VERSION 3.16)
project(MCLsamples)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE MNIST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
# 保存先ディレクトリ
set(MNIST_DIR ${CMAKE_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${MNIST_DIR})

# ファイル名とURLを交互に並べる
set(MNIST_FILES
    train-images-idx3-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
)

# リストの長さを取得
list(LENGTH MNIST_FILES total_len)
math(EXPR num_files "${total_len} / 2 - 1")

# 2個ずつ取り出す
foreach(i RANGE 0 ${num_files})
    math(EXPR idx_file "2 * ${i}")
    math(EXPR idx_url  "2 * ${i} + 1")

    list(GET MNIST_FILES ${idx_file} filename_gz)
    list(GET MNIST_FILES ${idx_url} url)

    set(download_path ${MNIST_DIR}/${filename_gz})

    if(NOT EXISTS ${download_path})
        message(STATUS "Downloading ${url}")
        file(DOWNLOAD ${url} ${download_path} SHOW_PROGRESS)
    endif()

    string(REPLACE ".gz" "" decompressed_file ${filename_gz})
    set(decompressed_path ${MNIST_DIR}/${decompressed_file})

    if(NOT EXISTS ${decompressed_path})
        message(STATUS "Extracting ${filename_gz}")
        execute_process(COMMAND gzip -d -c ${download_path}
                        OUTPUT_FILE ${decompressed_path})
    endif()
endforeach()

add_executable(mnistsample ${MNIST_SOURCES})
target_link_libraries(mnistsample PRIVATE MCLlib)

set_target_properties(mnistsample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})