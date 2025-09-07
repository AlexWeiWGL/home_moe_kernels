/*
 * 使用简单GEMM kernel进行调试
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "=== 调试简单GEMM + ReLU ===" << std::endl;
    
    // 简单的测试参数
    const int M = 2;  // 输入tokens数量
    const int N = 4;  // 输出维度
    const int K = 4;  // 输入维度
    const bool use_bias = true;
    
    // 创建测试数据
    std::vector<float> h_A = {
        1.0f, 2.0f, 3.0f, 4.0f,    // token 0
        5.0f, 6.0f, 7.0f, 8.0f     // token 1
    };
    
    std::vector<float> h_B = {
        0.1f, 0.2f, 0.3f, 0.4f,    // row 0
        0.5f, 0.6f, 0.7f, 0.8f,    // row 1
        0.9f, 1.0f, 1.1f, 1.2f,    // row 2
        1.3f, 1.4f, 1.5f, 1.6f     // row 3
    };
    
    std::vector<float> h_bias = {0.1f, 0.2f, 0.3f, 0.4f};
    
    std::cout << "使用简单GEMM kernel测试..." << std::endl;
    
    // 分配GPU内存
    float *d_A, *d_B, *d_bias, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // 复制数据
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 执行简单GEMM
    launch_simple_gemm_relu(d_A, d_B, d_bias, d_C, M, N, K, use_bias);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 获取结果
    std::vector<float> h_C(M * N);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "简单GEMM结果:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // PyTorch参考
    torch::Device device(torch::kCUDA, 0);
    auto torch_A = torch::from_blob(h_A.data(), {M, K}, torch::kFloat32).to(device);
    auto torch_B = torch::from_blob(h_B.data(), {K, N}, torch::kFloat32).to(device);
    auto torch_bias = torch::from_blob(h_bias.data(), {N}, torch::kFloat32).to(device);
    
    auto torch_C = torch::mm(torch_A, torch_B) + torch_bias.unsqueeze(0);
    torch_C = torch::relu(torch_C);
    
    auto h_C_torch = torch_C.cpu();
    
    std::cout << std::endl << "PyTorch参考结果:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < N; ++j) {
            std::cout << h_C_torch[i][j].item<float>() << " ";
        }
        std::cout << std::endl;
    }
    
    // 清理内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_C));
    
    return 0;
}
