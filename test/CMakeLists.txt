# Test CMakeLists.txt for MoE GEMM ReLU testing

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
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(debug_simple PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")

# Compile features for debug simple test
target_compile_features(debug_simple PUBLIC cxx_std_17)

# Compile definitions for debug simple test
target_compile_definitions(debug_simple PUBLIC COMPILE_CC=${COMPILE_CC})
