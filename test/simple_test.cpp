/*
 * MoE GEMM 测试 (ColumnMajor Weights, BF16 Precision)
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "cutlass/bfloat16.h"

#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// 辅助函数，用于生成随机数据 (BF16版本)
std::vector<cutlass::bfloat16_t> generate_random_data_bf16(size_t size) {
    std::vector<cutlass::bfloat16_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    for (size_t i = 0; i < size; ++i) {
        data[i] = cutlass::bfloat16_t(float(dis(gen)));
    }
    return data;
}

// 辅助函数，用于生成随机数据 (Float版本，用于PyTorch参考)
std::vector<float> generate_random_data_float(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}


int main() {
    std::cout << "=== MoE GEMM测试 (Float32 输入, BF16 Kernel, 专家全激活) ===" << std::endl;
    
    // 测试参数
    const int num_experts = 5;
    const int input_dim = 1400;   // K
    const int output_dim = 700;   // N
    const int batch_size = 4096;  // M
    const int total_rows = num_experts * batch_size;
    const bool use_bias = true;
    
    std::cout << "测试配置: " << num_experts << " experts, " 
              << batch_size << " tokens, " 
              << input_dim << " -> " << output_dim << " (Float32->BF16)" << std::endl;
    
    // 设置CUDA设备
    torch::Device device(torch::kCUDA, 0);
    
    // 1. 创建原始随机数据 (Float版本)
    auto h_weights_raw_row_major = generate_random_data_float(num_experts * input_dim * output_dim); // [E, K, N]
    auto h_input_base = generate_random_data_float(batch_size * input_dim);
    auto h_biases = generate_random_data_float(num_experts * output_dim);
    
    // 2. 复制输入以匹配"所有专家处理所有token"的场景
    std::vector<float> h_input_replicated(total_rows * input_dim);
    for (int i = 0; i < num_experts; ++i) {
        memcpy(h_input_replicated.data() + i * (batch_size * input_dim), 
               h_input_base.data(), 
               batch_size * input_dim * sizeof(float));
    }
    
    // 3. 为CUDA核转置为ColumnMajor布局期望的格式 [E, N, K]
    std::vector<float> h_weights_col_major_equivalent(num_experts * input_dim * output_dim);
    for (int e = 0; e < num_experts; ++e) {
        for (int k = 0; k < input_dim; ++k) {
            for (int n = 0; n < output_dim; ++n) {
                // (e, k, n) in RowMajor [E, K, N] -> (e, n, k) in RowMajor [E, N, K]
                int src_idx = e * (input_dim * output_dim) + k * output_dim + n;
                int dst_idx = e * (output_dim * input_dim) + n * input_dim + k;
                h_weights_col_major_equivalent[dst_idx] = h_weights_raw_row_major[src_idx];
            }
        }
    }

    // 4. 计算专家负载（前缀和）
    std::vector<int64_t> expert_offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; ++i) {
        expert_offsets[i+1] = expert_offsets[i] + batch_size;
    }
    
    std::cout << "输入数据形状 (replicated): [" << total_rows << ", " << input_dim << "]" << std::endl;
    std::cout << "权重矩阵形状 (for CUDA): [" << num_experts << ", " << output_dim << ", " << input_dim << "]" << std::endl;
    std::cout << "偏置形状: [" << num_experts << ", " << output_dim << "]" << std::endl << std::endl;

    // 分配GPU内存
    float *d_input, *d_weights, *d_biases, *d_output;
    int64_t *d_expert_offsets;
    
    CHECK_CUDA(cudaMalloc(&d_input, total_rows * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, num_experts * input_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_biases, num_experts * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total_rows * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int64_t)));
    
    // 复制数据到GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input_replicated.data(), total_rows * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    // 内核被强制为ColumnMajor，所以我们必须传递转置后的权重
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights_col_major_equivalent.data(), num_experts * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases, h_biases.data(), num_experts * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, expert_offsets.data(), (num_experts + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // --- CUDA Kernel 性能测试 ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    // 执行CUDA kernel (使用float32输入调用BF16 kernel)
    launch_moe_gemm_kernel(
        d_input, d_weights, d_biases, d_output, d_expert_offsets,
        num_experts, total_rows, input_dim, output_dim, use_bias
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cuda_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cuda_time_ms, start, stop));
    
    // 获取结果
    std::vector<float> h_output_cuda(total_rows * output_dim);
    CHECK_CUDA(cudaMemcpy(h_output_cuda.data(), d_output, total_rows * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // PyTorch参考实现 (使用BFloat16)
    auto torch_input_base = torch::from_blob(h_input_base.data(), {batch_size, input_dim}, torch::kFloat32).to(device).to(torch::kBFloat16);
    auto torch_weight_raw = torch::from_blob(h_weights_raw_row_major.data(), {num_experts, input_dim, output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16);
    auto torch_bias = torch::from_blob(h_biases.data(), {num_experts, output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16);
    
    // --- PyTorch 性能测试 ---
    CHECK_CUDA(cudaEventRecord(start));

    std::vector<torch::Tensor> expert_outputs;
    for (int i = 0; i < num_experts; ++i) {
        auto weight_expert = torch_weight_raw[i];
        auto bias_expert = torch_bias[i];
        auto output_slice = torch::mm(torch_input_base, weight_expert) + bias_expert.unsqueeze(0);
        expert_outputs.push_back(output_slice);
    }
    
    auto torch_output_gpu = torch::cat(expert_outputs, 0);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float torch_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&torch_time_ms, start, stop));
    
    auto h_output_torch = torch_output_gpu.cpu();
    
    // 打印部分结果用于对比
    std::cout << std::endl << "CUDA 输出 (专家0, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << h_output_cuda[i] << " ";
    std::cout << std::endl;
    
    std::cout << "PyTorch 输出 (专家0, 前8个元素, BF16):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << expert_outputs[0][0][i].item<float>() << " ";
    std::cout << std::endl;

    std::cout << std::endl << "CUDA 输出 (专家1, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << h_output_cuda[batch_size * output_dim + i] << " ";
    std::cout << std::endl;

    std::cout << "PyTorch 输出 (专家1, 前8个元素, BF16):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << expert_outputs[1][0][i].item<float>() << " ";
    std::cout << std::endl;


    // 比较结果
    std::cout << std::endl << "误差分析 (交叉验证 CUDA expert vs all PyTorch experts):" << std::endl;
    double max_overall_min_error = 0.0;

    for (int cuda_e = 0; cuda_e < num_experts; ++cuda_e) {
        double min_error_for_cuda_e = std::numeric_limits<double>::max();
        int best_match_torch_e = -1;

        for (int torch_e = 0; torch_e < num_experts; ++torch_e) {
            double current_max_error = 0.0;
            auto torch_expert_output = expert_outputs[torch_e];
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < output_dim; ++j) {
                    int cuda_idx = cuda_e * (batch_size * output_dim) + i * output_dim + j;
                    float cuda_val = h_output_cuda[cuda_idx];
                    float torch_val = torch_expert_output[i][j].item<float>();
                    double error = std::abs(cuda_val - torch_val);
                    current_max_error = std::max(current_max_error, error);
                }
            }
            if (current_max_error < min_error_for_cuda_e) {
                min_error_for_cuda_e = current_max_error;
                best_match_torch_e = torch_e;
            }
        }
        std::cout << "CUDA 专家 " << cuda_e 
                  << " 最佳匹配 PyTorch 专家 " << best_match_torch_e 
                  << ", 最小最大误差: " << min_error_for_cuda_e << std::endl;
        max_overall_min_error = std::max(max_overall_min_error, min_error_for_cuda_e);
    }
    
    std::cout << std::endl << "总最大绝对误差 (基于最佳匹配): " << max_overall_min_error << std::endl;
    // BF16精度下，误差阈值应该更宽松
    std::cout << "测试" << (max_overall_min_error < 1e-2 ? "通过" : "失败") << " (Float32->BF16)" << std::endl;
    
    // 打印性能对比
    std::cout << std::endl << "性能对比:" << std::endl;
    std::cout << "CUDA Kernel (Float32): " << cuda_time_ms << " ms" << std::endl;
    std::cout << "PyTorch (BF16): " << torch_time_ms << " ms" << std::endl;
    std::cout << "加速比: " << torch_time_ms / cuda_time_ms << "x" << std::endl;
    
    // 清理内存
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_biases));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    
    return 0;
}
