cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Home_MoE LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(COMPILE_CC 80)  # set the cuda arch

# 设置CUDA路径
set(CUDA_HOME "/usr/local/cuda-12.8")
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")
set(CMAKE_CUDA_RUNTIME_LIBRARY "STATIC")

# 设置PyTorch路径
set(TORCH_PREFIX "/root/miniconda3/lib/python3.12/site-packages/torch")
set(TORCH_INSTALL_PREFIX "${TORCH_PREFIX}")
set(TORCH_CXX_FLAGS "-DTORCH_EXTENSION_NAME=Home_MoE")

# lib
find_package(Torch REQUIRED)  # 更新到实际版本
find_package(Python3 3.12 REQUIRED Interpreter Development)  # 更新到实际版本
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_library(${PROJECT_NAME} SHARED "${CMAKE_SOURCE_DIR}/csrc/home_kernel.cu")

# For torch 2.3, C++17 or later compatible compiler is required to use ATen.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_CC=${COMPILE_CC})

target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "${CUDA_HOME}/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Add test subdirectory
add_subdirectory(test)
