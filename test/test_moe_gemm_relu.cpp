/*
 * MoE GEMM ReLU 测试主程序
 * 对比CUDA MoE kernel与PyTorch实现的速度和精度
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// 生成随机测试数据
void generate_random_data(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以便复现
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
}

// 计算两个向量的相对误差
double compute_relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Vector sizes don't match!" << std::endl;
        return -1.0;
    }
    
    double max_error = 0.0;
    double sum_error = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        double error = std::abs(a[i] - b[i]);
        double relative_error = error / (std::abs(a[i]) + 1e-8);
        
        max_error = std::max(max_error, relative_error);
        sum_error += relative_error;
        count++;
    }
    
    std::cout << "Max relative error: " << max_error << std::endl;
    std::cout << "Avg relative error: " << sum_error / count << std::endl;
    
    return max_error;
}

// PyTorch实现的MoE GEMM + ReLU
torch::Tensor torch_moe_gemm_relu(
    const torch::Tensor& input,     // [total_tokens, input_dim]
    const torch::Tensor& weights,   // [num_experts, input_dim, output_dim]
    const torch::Tensor& biases,    // [num_experts, output_dim]
    const std::vector<int>& expert_tokens, // 每个专家的token数量
    bool use_bias
) {
    auto device = input.device();
    int num_experts = weights.size(0);
    int input_dim = weights.size(1);
    int output_dim = weights.size(2);
    int total_tokens = input.size(0);
    
    auto output = torch::zeros({total_tokens, output_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    int token_offset = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        int num_tokens = expert_tokens[expert_idx];
        if (num_tokens == 0) continue;
        
        // 获取当前专家的输入
        auto expert_input = input.slice(0, token_offset, token_offset + num_tokens); // [num_tokens, input_dim]
        
        // 获取当前专家的权重和偏置
        auto expert_weight = weights[expert_idx]; // [input_dim, output_dim]
        auto expert_bias = use_bias ? biases[expert_idx] : torch::Tensor(); // [output_dim]
        
        // 执行矩阵乘法
        auto expert_output = torch::mm(expert_input, expert_weight); // [num_tokens, output_dim]
        
        // 添加偏置（如果使用）
        if (use_bias) {
            expert_output += expert_bias.unsqueeze(0).expand_as(expert_output);
        }
        
        // 应用ReLU激活函数
        expert_output = torch::relu(expert_output);
        
        // 将结果写入输出tensor
        output.slice(0, token_offset, token_offset + num_tokens) = expert_output;
        
        token_offset += num_tokens;
    }
    
    return output;
}

int main() {
    std::cout << "=== MoE GEMM ReLU 性能和精度测试 ===" << std::endl;
    
    // 测试参数
    const int num_experts = 8;
    const int input_dim = 512;
    const int output_dim = 1024;
    const int total_tokens = 2048;
    const bool use_bias = true;
    const int num_iterations = 100; // 性能测试迭代次数
    
    std::cout << "测试配置:" << std::endl;
    std::cout << "  专家数量: " << num_experts << std::endl;
    std::cout << "  输入维度: " << input_dim << std::endl;
    std::cout << "  输出维度: " << output_dim << std::endl;
    std::cout << "  总token数: " << total_tokens << std::endl;
    std::cout << "  使用偏置: " << (use_bias ? "是" : "否") << std::endl;
    std::cout << "  性能测试迭代次数: " << num_iterations << std::endl << std::endl;
    
    // 设置CUDA设备
    torch::Device device(torch::kCUDA, 0);
    
    // 生成测试数据
    std::vector<float> h_input(total_tokens * input_dim);
    std::vector<float> h_weights(num_experts * input_dim * output_dim);
    std::vector<float> h_biases(num_experts * output_dim);
    
    generate_random_data(h_input, -0.5f, 0.5f);
    generate_random_data(h_weights, -0.1f, 0.1f);
    generate_random_data(h_biases, -0.1f, 0.1f);
    
    // MoeFCGemm期望权重布局：每个专家的权重矩阵按专家索引连续存储
    // 关键发现：对于float类型，MixedGemmArchTraits使用ColumnMajor布局！
    // 需要将每个专家的权重矩阵从row-major转置为column-major
    std::vector<float> h_weights_reordered(num_experts * input_dim * output_dim);
    for (int expert = 0; expert < num_experts; ++expert) {
        for (int k = 0; k < input_dim; ++k) {
            for (int n = 0; n < output_dim; ++n) {
                // 原始布局: [expert][k][n] (row-major)
                // 目标布局: [expert][n][k] (column-major)
                int src_idx = expert * input_dim * output_dim + k * output_dim + n;
                int dst_idx = expert * input_dim * output_dim + n * input_dim + k;
                h_weights_reordered[dst_idx] = h_weights[src_idx];
            }
        }
    }
    
    // 生成每个专家的token分配（随机但平衡）
    std::vector<int> expert_tokens(num_experts);
    int remaining_tokens = total_tokens;
    for (int i = 0; i < num_experts - 1; ++i) {
        expert_tokens[i] = remaining_tokens / (num_experts - i);
        remaining_tokens -= expert_tokens[i];
    }
    expert_tokens[num_experts - 1] = remaining_tokens;
    
    std::cout << "专家token分配: ";
    for (int i = 0; i < num_experts; ++i) {
        std::cout << expert_tokens[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    // 创建专家偏移量数组
    std::vector<int64_t> expert_offsets(num_experts + 1);
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        expert_offsets[i + 1] = expert_offsets[i] + expert_tokens[i];
    }
    
    // 重新排列输入数据以匹配MoeFCGemm期望的布局
    // MoeFCGemm期望输入按专家分组：[expert0_tokens, expert1_tokens, ...]
    std::vector<float> h_input_reordered(total_tokens * input_dim);
    int input_offset = 0;
    int output_offset = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
        int num_tokens = expert_tokens[expert];
        // 复制当前专家的输入数据
        for (int token = 0; token < num_tokens; ++token) {
            for (int dim = 0; dim < input_dim; ++dim) {
                h_input_reordered[output_offset * input_dim + dim] = h_input[input_offset * input_dim + dim];
            }
            input_offset++;
            output_offset++;
        }
    }
    
    // 分配GPU内存
    float *d_input, *d_weights, *d_biases, *d_output_cuda;
    int64_t *d_expert_offsets;
    
    CHECK_CUDA(cudaMalloc(&d_input, total_tokens * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, num_experts * input_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_biases, num_experts * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cuda, total_tokens * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int64_t)));
    
    // 复制数据到GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input_reordered.data(), total_tokens * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights_reordered.data(), num_experts * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases, h_biases.data(), num_experts * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, expert_offsets.data(), (num_experts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // 创建PyTorch tensors（使用重新排列的数据以确保公平比较）
    auto torch_input = torch::from_blob(h_input_reordered.data(), {total_tokens, input_dim}, torch::kFloat32).to(device);
    auto torch_weights = torch::from_blob(h_weights.data(), {num_experts, input_dim, output_dim}, torch::kFloat32).to(device);
    auto torch_biases = torch::from_blob(h_biases.data(), {num_experts, output_dim}, torch::kFloat32).to(device);
    
    std::cout << "=== 精度测试 ===" << std::endl;
    
    // 执行CUDA kernel
    launch_moe_gemm_relu_kernel(
        d_input, d_weights, d_biases, d_output_cuda, d_expert_offsets,
        num_experts, total_tokens, input_dim, output_dim, use_bias
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 获取CUDA结果
    std::vector<float> h_output_cuda(total_tokens * output_dim);
    CHECK_CUDA(cudaMemcpy(h_output_cuda.data(), d_output_cuda, total_tokens * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 执行PyTorch实现
    auto torch_output = torch_moe_gemm_relu(torch_input, torch_weights, torch_biases, expert_tokens, use_bias);
    auto h_output_torch = torch_output.cpu().contiguous();
    std::vector<float> torch_result(h_output_torch.data_ptr<float>(), 
                                   h_output_torch.data_ptr<float>() + h_output_torch.numel());
    
    // 比较精度
    double max_error = compute_relative_error(h_output_cuda, torch_result);
    
    // 添加调试信息：打印前几个输出值
    std::cout << "CUDA结果前10个值: ";
    for (int i = 0; i < std::min(10, (int)h_output_cuda.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << h_output_cuda[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "PyTorch结果前10个值: ";
    for (int i = 0; i < std::min(10, (int)torch_result.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << torch_result[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "精度测试 " << (max_error < 1e-3 ? "通过" : "失败") << " (阈值: 1e-3)" << std::endl << std::endl;
    
    std::cout << "=== 性能测试 ===" << std::endl;
    
    Timer timer;
    
    // 测试CUDA kernel性能
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        launch_moe_gemm_relu_kernel(
            d_input, d_weights, d_biases, d_output_cuda, d_expert_offsets,
            num_experts, total_tokens, input_dim, output_dim, use_bias
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double cuda_time = timer.stop();
    
    // 测试PyTorch性能
    torch::cuda::synchronize();
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        auto result = torch_moe_gemm_relu(torch_input, torch_weights, torch_biases, expert_tokens, use_bias);
    }
    torch::cuda::synchronize();
    double torch_time = timer.stop();
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CUDA MoE Kernel: " << cuda_time / num_iterations << " ms/次" << std::endl;
    std::cout << "PyTorch实现:     " << torch_time / num_iterations << " ms/次" << std::endl;
    std::cout << "加速比:          " << torch_time / cuda_time << "x" << std::endl;
    
    // 清理内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_biases));
    CHECK_CUDA(cudaFree(d_output_cuda));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    
    std::cout << std::endl << "测试完成!" << std::endl;
    
    return 0;
}
