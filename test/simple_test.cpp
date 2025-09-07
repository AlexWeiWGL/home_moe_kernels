/*
 * 简单的单专家MoE GEMM ReLU测试
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "=== 简单单专家MoE GEMM ReLU测试 ===" << std::endl;
    
    // 简单的测试参数
    const int num_experts = 1;
    const int input_dim = 4;
    const int output_dim = 4;
    const int total_tokens = 2;
    const bool use_bias = true;
    
    std::cout << "测试配置: " << num_experts << " experts, " 
              << total_tokens << " tokens, " 
              << input_dim << " -> " << output_dim << std::endl;
    
    // 设置CUDA设备
    torch::Device device(torch::kCUDA, 0);
    
    // 创建简单的测试数据
    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,    // token 0
        5.0f, 6.0f, 7.0f, 8.0f     // token 1
    };
    
    std::vector<float> h_weights_raw = {
        0.1f, 0.2f, 0.3f, 0.4f,    // row 0
        0.5f, 0.6f, 0.7f, 0.8f,    // row 1
        0.9f, 1.0f, 1.1f, 1.2f,    // row 2
        1.3f, 1.4f, 1.5f, 1.6f     // row 3
    };
    
    // 转置为column-major布局（MoeFCGemm期望的布局）
    std::vector<float> h_weights(input_dim * output_dim);
    for (int k = 0; k < input_dim; ++k) {
        for (int n = 0; n < output_dim; ++n) {
            // 从row-major转为column-major
            int src_idx = k * output_dim + n;
            int dst_idx = n * input_dim + k;
            h_weights[dst_idx] = h_weights_raw[src_idx];
        }
    }
    
    std::vector<float> h_biases = {0.1f, 0.2f, 0.3f, 0.4f};
    
    // 专家token分配
    std::vector<int> expert_tokens = {total_tokens};
    std::vector<int64_t> expert_offsets = {0, total_tokens};
    
    std::cout << "输入数据:" << std::endl;
    for (int i = 0; i < total_tokens; ++i) {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < input_dim; ++j) {
            std::cout << h_input[i * input_dim + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "权重矩阵:" << std::endl;
    for (int i = 0; i < input_dim; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < output_dim; ++j) {
            std::cout << h_weights[i * output_dim + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "偏置: ";
    for (float b : h_biases) {
        std::cout << b << " ";
    }
    std::cout << std::endl << std::endl;
    
    // 分配GPU内存
    float *d_input, *d_weights, *d_biases, *d_output;
    int64_t *d_expert_offsets;
    
    CHECK_CUDA(cudaMalloc(&d_input, total_tokens * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, num_experts * input_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_biases, num_experts * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total_tokens * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int64_t)));
    
    // 复制数据到GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), total_tokens * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), num_experts * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases, h_biases.data(), num_experts * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, expert_offsets.data(), (num_experts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // 执行CUDA kernel
    launch_moe_gemm_relu_kernel(
        d_input, d_weights, d_biases, d_output, d_expert_offsets,
        num_experts, total_tokens, input_dim, output_dim, use_bias
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 获取结果
    std::vector<float> h_output_cuda(total_tokens * output_dim);
    CHECK_CUDA(cudaMemcpy(h_output_cuda.data(), d_output, total_tokens * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "CUDA结果:" << std::endl;
    for (int i = 0; i < total_tokens; ++i) {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < output_dim; ++j) {
            std::cout << h_output_cuda[i * output_dim + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // PyTorch参考实现
    auto torch_input = torch::from_blob(h_input.data(), {total_tokens, input_dim}, torch::kFloat32).to(device);
    auto torch_weight = torch::from_blob(h_weights.data(), {input_dim, output_dim}, torch::kFloat32).to(device);
    auto torch_bias = torch::from_blob(h_biases.data(), {output_dim}, torch::kFloat32).to(device);
    
    auto torch_output = torch::mm(torch_input, torch_weight) + torch_bias.unsqueeze(0);
    torch_output = torch::relu(torch_output);
    
    auto h_output_torch = torch_output.cpu();
    
    std::cout << std::endl << "PyTorch参考结果:" << std::endl;
    for (int i = 0; i < total_tokens; ++i) {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < output_dim; ++j) {
            std::cout << h_output_torch[i][j].item<float>() << " ";
        }
        std::cout << std::endl;
    }
    
    // 比较结果
    std::cout << std::endl << "误差分析:" << std::endl;
    double max_error = 0.0;
    for (int i = 0; i < total_tokens; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            float cuda_val = h_output_cuda[i * output_dim + j];
            float torch_val = h_output_torch[i][j].item<float>();
            double error = std::abs(cuda_val - torch_val);
            max_error = std::max(max_error, error);
            std::cout << "(" << i << "," << j << "): CUDA=" << cuda_val 
                      << " PyTorch=" << torch_val << " Error=" << error << std::endl;
        }
    }
    
    std::cout << std::endl << "最大绝对误差: " << max_error << std::endl;
    std::cout << "测试" << (max_error < 1e-5 ? "通过" : "失败") << std::endl;
    
    // 清理内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_biases));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    
    return 0;
}
