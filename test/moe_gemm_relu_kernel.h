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

/**
 * 启动MoE GEMM ReLU kernel
 * @param input 输入矩阵 [total_tokens, input_dim]
 * @param weights 权重矩阵 [num_experts, input_dim, output_dim] (row-major)
 * @param biases 偏置向量 [num_experts, output_dim] 或 nullptr
 * @param output 输出矩阵 [total_tokens, output_dim]
 * @param expert_offsets 专家偏移量 [num_experts + 1] - 累积token数量
 * @param num_experts 专家数量
 * @param total_tokens 总token数量
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param use_bias 是否使用偏置
 */
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

/**
 * 启动MoE GEMM kernel (不带ReLU)
 */
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

/**
 * 简单的GEMM + ReLU实现（用于对比）
 * @param A 输入矩阵A [M, K]
 * @param B 权重矩阵B [K, N]
 * @param bias 偏置向量 [N] 或 nullptr
 * @param C 输出矩阵C [M, N]
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数/矩阵B的行数
 * @param use_bias 是否使用偏置
 */
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

/**
 * 启动MoE GEMM ReLU kernel (BF16版本)
 */
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

/**
 * 启动MoE GEMM kernel (不带ReLU, BF16版本)
 */
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

/**
 * 启动MoE GEMM ReLU kernel (FP16版本)
 */
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

/**
 * 启动MoE GEMM kernel (不带ReLU, FP16版本)
 */
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

/**
 * 启动MoE GEMM ReLU kernel (BF16版本)
 */
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

/**
 * 启动MoE GEMM kernel (不带ReLU, BF16版本)
 */
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

#ifdef __cplusplus
}
#endif

#endif // MOE_GEMM_RELU_KERNEL_H
