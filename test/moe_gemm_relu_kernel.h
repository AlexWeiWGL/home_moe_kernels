/*
 * MoE GEMM ReLU 测试内核头文件
 */

#ifndef MOE_GEMM_RELU_KERNEL_H
#define MOE_GEMM_RELU_KERNEL_H

#include <cstdint>
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#ifdef __cplusplus
extern "C" {
#endif

// Float 版本
void launch_moe_gemm_relu_kernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

void launch_moe_gemm_kernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

void launch_simple_gemm_relu(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K,
    bool use_bias
);

// 优化版（共享输入）Float
void launch_moe_gemm_relu_kernel_optimized(
    const float* shared_input,
    const float* weights,
    const float* biases,
    float* output,
    int num_experts,
    int batch_size,
    int input_dim,
    int output_dim,
    bool use_bias
);

void launch_moe_gemm_kernel_optimized(
    const float* shared_input,
    const float* weights,
    const float* biases,
    float* output,
    int num_experts,
    int batch_size,
    int input_dim,
    int output_dim,
    bool use_bias
);

// FP16 版本
void launch_moe_gemm_relu_kernel_fp16(
    const cutlass::half_t* input,
    const cutlass::half_t* weights,
    const cutlass::half_t* biases,
    cutlass::half_t* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

void launch_moe_gemm_kernel_fp16(
    const cutlass::half_t* input,
    const cutlass::half_t* weights,
    const cutlass::half_t* biases,
    cutlass::half_t* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

// BF16 版本
void launch_moe_gemm_relu_kernel_bf16(
    const cutlass::bfloat16_t* input,
    const cutlass::bfloat16_t* weights,
    const cutlass::bfloat16_t* biases,
    cutlass::bfloat16_t* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

void launch_moe_gemm_kernel_bf16(
    const cutlass::bfloat16_t* input,
    const cutlass::bfloat16_t* weights,
    const cutlass::bfloat16_t* biases,
    cutlass::bfloat16_t* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
);

// BN + SiLU（BF16）
void launch_moe_gemm_bn_silu_kernel_bf16(
    const cutlass::bfloat16_t* input,
    const cutlass::bfloat16_t* weights,
    const cutlass::bfloat16_t* biases,
    cutlass::bfloat16_t* output,
    const int64_t* expert_offsets,
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    const float* bn_gamma,
    const float* bn_beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    bool use_bias
);

#ifdef __cplusplus
}
#endif

#endif // MOE_GEMM_RELU_KERNEL_H
