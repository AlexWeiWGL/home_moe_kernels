# Test CMakeLists.txt for MoE GEMM ReLU testing

# 设置CUDA路径
set(CUDA_HOME "/usr/local/cuda-12.8")
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")

# 设置PyTorch路径
set(TORCH_PREFIX "/root/miniconda3/lib/python3.12/site-packages/torch")
set(TORCH_INSTALL_PREFIX "${TORCH_PREFIX}")

# Add test executable
add_executable(test_moe_gemm_relu
    test_moe_gemm_relu.cpp
    moe_gemm_relu_kernel.cu
)

# Set CUDA architecture
set_target_properties(test_moe_gemm_relu PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(test_moe_gemm_relu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(test_moe_gemm_relu ${TORCH_LIBRARIES})

# Include directories
target_include_directories(test_moe_gemm_relu PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test_moe_gemm_relu PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(test_moe_gemm_relu PUBLIC "${CUDA_HOME}/include")
target_include_directories(test_moe_gemm_relu PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(test_moe_gemm_relu PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(test_moe_gemm_relu PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features
target_compile_features(test_moe_gemm_relu PUBLIC cxx_std_17)

# CUDA flags
set_target_properties(test_moe_gemm_relu PROPERTIES 
    CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr"
)

# Compile definitions
target_compile_definitions(test_moe_gemm_relu PUBLIC COMPILE_CC=${COMPILE_CC})

# Add simple test executable
add_executable(simple_test
    simple_test.cpp
    moe_gemm_relu_kernel.cu
)

# Set CUDA architecture for simple test
set_target_properties(simple_test PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(simple_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries for simple test
target_link_libraries(simple_test ${TORCH_LIBRARIES})

# Include directories for simple test
target_include_directories(simple_test PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(simple_test PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(simple_test PUBLIC "${CUDA_HOME}/include")
target_include_directories(simple_test PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(simple_test PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(simple_test PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for simple test
target_compile_features(simple_test PUBLIC cxx_std_17)

# Compile definitions for simple test
target_compile_definitions(simple_test PUBLIC COMPILE_CC=${COMPILE_CC})

# Add debug simple test executable
add_executable(debug_simple
    debug_simple.cpp
    moe_gemm_relu_kernel.cu
)

# Set CUDA architecture for debug simple test
set_target_properties(debug_simple PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(debug_simple PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries for debug simple test
target_link_libraries(debug_simple ${TORCH_LIBRARIES})

# Include directories for debug simple test
target_include_directories(debug_simple PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(debug_simple PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(debug_simple PUBLIC "${CUDA_HOME}/include")
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for debug simple test
target_compile_features(debug_simple PUBLIC cxx_std_17)

# Compile definitions for debug simple test
target_compile_definitions(debug_simple PUBLIC COMPILE_CC=${COMPILE_CC})

# Add FP16 vs FP32 test executable
add_executable(test_fp16_vs_fp32
    test_fp16_vs_fp32.cpp
    moe_gemm_relu_kernel.cu
)

# Set CUDA architecture for FP16 vs FP32 test
set_target_properties(test_fp16_vs_fp32 PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(test_fp16_vs_fp32 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries for FP16 vs FP32 test
target_link_libraries(test_fp16_vs_fp32 ${TORCH_LIBRARIES})

# Include directories for FP16 vs FP32 test
target_include_directories(test_fp16_vs_fp32 PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test_fp16_vs_fp32 PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(test_fp16_vs_fp32 PUBLIC "${CUDA_HOME}/include")
target_include_directories(test_fp16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(test_fp16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(test_fp16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for FP16 vs FP32 test
target_compile_features(test_fp16_vs_fp32 PUBLIC cxx_std_17)

# Compile definitions for FP16 vs FP32 test
target_compile_definitions(test_fp16_vs_fp32 PUBLIC COMPILE_CC=${COMPILE_CC})

# Add BF16 vs FP32 test executable
add_executable(test_bf16_vs_fp32
    test_bf16_vs_fp32.cpp
    moe_gemm_relu_kernel.cu
)

# Set CUDA architecture for BF16 vs FP32 test
set_target_properties(test_bf16_vs_fp32 PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(test_bf16_vs_fp32 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries for BF16 vs FP32 test
target_link_libraries(test_bf16_vs_fp32 ${TORCH_LIBRARIES})

# Include directories for BF16 vs FP32 test
target_include_directories(test_bf16_vs_fp32 PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test_bf16_vs_fp32 PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(test_bf16_vs_fp32 PUBLIC "${CUDA_HOME}/include")
target_include_directories(test_bf16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(test_bf16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(test_bf16_vs_fp32 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for BF16 vs FP32 test
target_compile_features(test_bf16_vs_fp32 PUBLIC cxx_std_17)

# Compile definitions for BF16 vs FP32 test
target_compile_definitions(test_bf16_vs_fp32 PUBLIC COMPILE_CC=${COMPILE_CC})

# Add optimized MoE GEMM test executable
add_executable(test_optimized_moe_gemm
    test_optimized_moe_gemm.cpp
    moe_gemm_relu_kernel.cu
    moe_gemm_relu_kernel_optimized.cu
)

# Set CUDA architecture for optimized MoE GEMM test
set_target_properties(test_optimized_moe_gemm PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(test_optimized_moe_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries for optimized MoE GEMM test
target_link_libraries(test_optimized_moe_gemm ${TORCH_LIBRARIES})

# Include directories for optimized MoE GEMM test
target_include_directories(test_optimized_moe_gemm PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test_optimized_moe_gemm PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(test_optimized_moe_gemm PUBLIC "${CUDA_HOME}/include")
target_include_directories(test_optimized_moe_gemm PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(test_optimized_moe_gemm PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(test_optimized_moe_gemm PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for optimized MoE GEMM test
target_compile_features(test_optimized_moe_gemm PUBLIC cxx_std_17)

# Compile definitions for optimized MoE GEMM test
target_compile_definitions(test_optimized_moe_gemm PUBLIC COMPILE_CC=${COMPILE_CC})

# New: BN+SiLU BF16 test
add_executable(test_moe_bn_silu_bf16
    test_moe_bn_silu_bf16.cpp
    moe_gemm_relu_kernel.cu
)
set_target_properties(test_moe_bn_silu_bf16 PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(test_moe_bn_silu_bf16 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_moe_bn_silu_bf16 ${TORCH_LIBRARIES})

target_include_directories(test_moe_bn_silu_bf16 PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test_moe_bn_silu_bf16 PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(test_moe_bn_silu_bf16 PUBLIC "${CUDA_HOME}/include")
target_include_directories(test_moe_bn_silu_bf16 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(test_moe_bn_silu_bf16 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(test_moe_bn_silu_bf16 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

target_compile_features(test_moe_bn_silu_bf16 PUBLIC cxx_std_17)

target_compile_definitions(test_moe_bn_silu_bf16 PUBLIC COMPILE_CC=${COMPILE_CC})
