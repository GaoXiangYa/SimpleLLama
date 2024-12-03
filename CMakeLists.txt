cmake_minimum_required(VERSION 3.20)

option(USE_CUDA "Enable CUDA" ON)
message(STATUS "Option USE_CUDA: ${USE_CUDA}")
option(USE_OPENCL OFF)
message(STATUS "Option USE_OPENCL: ${USE_OPENCL}")
option(USE_GTEST OFF)
message(STATUS "Option USE_GTEST: ${USE_GTEST}")
option(USE_DEMO OFF)
message(STATUS "Option USE_DEMO: ${USE_DEMO}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build type is Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
  message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
  message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
endif()

project(simple_llama CXX C CUDA)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/simple/include)

# 自动检索各目录下的文件
# file(GLOB DIR_TENSOR simple/src/tensor/*.cpp)
file(GLOB DIR_BASE simple/src/base/*.cpp)
# file(GLOB DIR_OP simple/src/op/*.cpp)
# file(GLOB DIR_MODEL simple/src/model/*.cpp)
# file(GLOB DIR_KERNEL_CPU simple/src/op/kernels/cpu/*.cpp)
# file(GLOB DIR_KERNEL_CUDA simple/src/op/kernels/cuda/*.cpp)
# file(GLOB DIR_KERNEL simple/src/op/kernels/*.cpp)
# file(GLOB DIR_SAMPLE simple/src/sampler/*.cpp)

if(USE_CUDA)
  # 查找 CUDA
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  # enable_language(CUDA)
endif()

add_library(llama SHARED ${DIR_BASE})
# add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama glog::glog gtest_main pthread)
# target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/simple/include)
# target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})

if (USE_CUDA)
  target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# if (USE_GTEST)
#   add_subdirectory(test)
# endif()
