/*
 * BF16 vs FP32 精度对比测试
 * CUDA BF16 kernel vs PyTorch FP32 计算
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// 将float转换为cutlass::bfloat16_t
cutlass::bfloat16_t float_to_bf16(float f) {
    return cutlass::bfloat16_t(f);
}

// 将cutlass::bfloat16_t转换为float
float bf16_to_float(cutlass::bfloat16_t bf) {
    return float(bf);
}

int main() {
    std::cout << "=== BF16 vs FP32 精度对比测试 ===" << std::endl;
    
    // 测试参数 - 使用对齐的维度，采用"所有专家处理所有token"模式
    const int num_experts = 5;
    const int batch_size = 4096;  // 每个专家处理的token数量
    const int input_dim = 1408;   // 1408 = 176 * 8，对齐到8
    const int output_dim = 704;   // 704 = 88 * 8，对齐到8
    const int total_rows = num_experts * batch_size;  // 总行数
    const bool use_bias = true;
    
    std::cout << "测试配置: " << num_experts << " experts, " 
              << batch_size << " tokens per expert, " 
              << input_dim << " -> " << output_dim 
              << " (CUDA BF16 vs PyTorch FP32)" << std::endl;
    
    // 设置随机种子
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // 生成基础输入数据 (FP32) - 每个专家都处理相同的输入
    std::vector<float> input_base_fp32(batch_size * input_dim);
    for (auto& val : input_base_fp32) {
        val = dis(gen);
    }
    
    // 复制输入数据以匹配"所有专家处理所有token"的场景
    std::vector<float> input_fp32(total_rows * input_dim);
    for (int i = 0; i < num_experts; ++i) {
        memcpy(input_fp32.data() + i * (batch_size * input_dim), 
               input_base_fp32.data(), 
               batch_size * input_dim * sizeof(float));
    }
    
    // 生成权重数据 (FP32) - 需要转置为ColumnMajor格式
    std::vector<float> weights_raw_fp32(num_experts * input_dim * output_dim);
    for (auto& val : weights_raw_fp32) {
        val = dis(gen) * 0.1f; // 较小的权重
    }
    
    // 转置权重为ColumnMajor格式 [E, N, K]
    std::vector<float> weights_fp32(num_experts * input_dim * output_dim);
    for (int e = 0; e < num_experts; ++e) {
        for (int k = 0; k < input_dim; ++k) {
            for (int n = 0; n < output_dim; ++n) {
                // (e, k, n) in RowMajor [E, K, N] -> (e, n, k) in RowMajor [E, N, K]
                int src_idx = e * (input_dim * output_dim) + k * output_dim + n;
                int dst_idx = e * (output_dim * input_dim) + n * input_dim + k;
                weights_fp32[dst_idx] = weights_raw_fp32[src_idx];
            }
        }
    }
    
    // 生成偏置数据 (FP32)
    std::vector<float> biases_fp32(num_experts * output_dim);
    for (auto& val : biases_fp32) {
        val = dis(gen) * 0.01f; // 较小的偏置
    }
    
    // 专家偏移量 (每个专家处理相同数量的tokens)
    std::vector<int64_t> expert_offsets(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert_offsets[i] = (i + 1) * batch_size;
    }
    
    // 分配GPU内存 (使用对齐分配)
    cutlass::bfloat16_t *d_input_bf16, *d_weights_bf16, *d_biases_bf16, *d_output_bf16;
    int64_t *d_expert_offsets;
    
    size_t input_size = total_rows * input_dim * sizeof(cutlass::bfloat16_t);
    size_t weights_size = num_experts * input_dim * output_dim * sizeof(cutlass::bfloat16_t);
    size_t biases_size = num_experts * output_dim * sizeof(cutlass::bfloat16_t);
    size_t output_size = total_rows * output_dim * sizeof(cutlass::bfloat16_t);
    size_t offsets_size = num_experts * sizeof(int64_t);
    
    CHECK_CUDA(cudaMalloc(&d_input_bf16, input_size));
    CHECK_CUDA(cudaMalloc(&d_weights_bf16, weights_size));
    CHECK_CUDA(cudaMalloc(&d_biases_bf16, biases_size));
    CHECK_CUDA(cudaMalloc(&d_output_bf16, output_size));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, offsets_size));
    
    // 转换数据到BF16并复制到GPU
    std::vector<cutlass::bfloat16_t> input_bf16(total_rows * input_dim);
    std::vector<cutlass::bfloat16_t> weights_bf16(num_experts * input_dim * output_dim);
    std::vector<cutlass::bfloat16_t> biases_bf16(num_experts * output_dim);
    
    for (int i = 0; i < total_rows * input_dim; ++i) {
        input_bf16[i] = float_to_bf16(input_fp32[i]);
    }
    
    for (int i = 0; i < num_experts * input_dim * output_dim; ++i) {
        weights_bf16[i] = float_to_bf16(weights_fp32[i]);
    }
    
    for (int i = 0; i < num_experts * output_dim; ++i) {
        biases_bf16[i] = float_to_bf16(biases_fp32[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(d_input_bf16, input_bf16.data(), input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights_bf16, weights_bf16.data(), weights_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases_bf16, biases_bf16.data(), biases_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, expert_offsets.data(), offsets_size, cudaMemcpyHostToDevice));
    
    // 创建PyTorch张量 (FP32) - 使用原始RowMajor格式的权重
    auto input_base_tensor = torch::from_blob(input_base_fp32.data(), {batch_size, input_dim}, torch::kFloat32).clone();
    auto weights_raw_tensor = torch::from_blob(weights_raw_fp32.data(), {num_experts, input_dim, output_dim}, torch::kFloat32).clone();
    auto biases_tensor = torch::from_blob(biases_fp32.data(), {num_experts, output_dim}, torch::kFloat32).clone();
    
    // 复制到GPU
    input_base_tensor = input_base_tensor.cuda();
    weights_raw_tensor = weights_raw_tensor.cuda();
    biases_tensor = biases_tensor.cuda();
    
    std::cout << "输入数据形状 (replicated): [" << total_rows << ", " << input_dim << "]" << std::endl;
    std::cout << "权重矩阵形状 (for CUDA): [" << num_experts << ", " << output_dim << ", " << input_dim << "]" << std::endl;
    std::cout << "偏置形状: [" << num_experts << ", " << output_dim << "]" << std::endl;
    
    // 运行CUDA BF16 kernel - 使用CUDA事件进行精确计时
    std::cout << "\n运行CUDA BF16 kernel..." << std::endl;
    
    // 创建CUDA事件
    cudaEvent_t start_event, end_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&end_event));
    
    // 预热运行
    launch_moe_gemm_relu_kernel_bf16(
        d_input_bf16, d_weights_bf16, d_biases_bf16, d_output_bf16,
        d_expert_offsets, num_experts, total_rows, input_dim, output_dim, use_bias
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    const int num_runs = 10;
    float total_cuda_time = 0.0f;
    
    for (int run = 0; run < num_runs; ++run) {
        // 记录开始事件
        CHECK_CUDA(cudaEventRecord(start_event));
        
        launch_moe_gemm_relu_kernel_bf16(
            d_input_bf16, d_weights_bf16, d_biases_bf16, d_output_bf16,
            d_expert_offsets, num_experts, total_rows, input_dim, output_dim, use_bias
        );
        
        // 记录结束事件
        CHECK_CUDA(cudaEventRecord(end_event));
        
        // 等待事件完成并计算时间
        CHECK_CUDA(cudaEventSynchronize(end_event));
        float run_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&run_time_ms, start_event, end_event));
        total_cuda_time += run_time_ms;
    }
    
    float cuda_time_ms = total_cuda_time / num_runs;
    
    // 检查kernel执行后的错误
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::cerr << "Kernel执行错误: " << cudaGetErrorString(kernel_error) << std::endl;
        return 1;
    }
    
    // 清理事件
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(end_event));
    
    // 运行PyTorch FP32计算 - 使用CUDA事件进行精确计时
    std::cout << "运行PyTorch FP32计算..." << std::endl;
    
    // 创建PyTorch CUDA事件
    cudaEvent_t pytorch_start_event, pytorch_end_event;
    CHECK_CUDA(cudaEventCreate(&pytorch_start_event));
    CHECK_CUDA(cudaEventCreate(&pytorch_end_event));
    
    // 预热运行
    std::vector<torch::Tensor> expert_outputs_warmup;
    for (int i = 0; i < num_experts; ++i) {
        auto weight_expert = weights_raw_tensor[i];
        auto bias_expert = biases_tensor[i];
        auto output_slice = torch::mm(input_base_tensor, weight_expert) + bias_expert.unsqueeze(0);
        output_slice = torch::relu(output_slice);
        expert_outputs_warmup.push_back(output_slice);
    }
    auto pytorch_output_warmup = torch::cat(expert_outputs_warmup, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    float total_pytorch_time = 0.0f;
    
    for (int run = 0; run < num_runs; ++run) {
        // 记录开始事件
        CHECK_CUDA(cudaEventRecord(pytorch_start_event));
        
        // 为每个专家分别计算，使用与simple_test.cpp相同的方式，并添加ReLU激活
        std::vector<torch::Tensor> expert_outputs;
        for (int i = 0; i < num_experts; ++i) {
            auto weight_expert = weights_raw_tensor[i];
            auto bias_expert = biases_tensor[i];
            auto output_slice = torch::mm(input_base_tensor, weight_expert) + bias_expert.unsqueeze(0);
            // 添加ReLU激活，与CUDA kernel保持一致
            output_slice = torch::relu(output_slice);
            expert_outputs.push_back(output_slice);
        }
        
        // 合并所有专家的输出
        auto pytorch_output = torch::cat(expert_outputs, 0);
        
        // 记录结束事件
        CHECK_CUDA(cudaEventRecord(pytorch_end_event));
        
        // 等待事件完成并计算时间
        CHECK_CUDA(cudaEventSynchronize(pytorch_end_event));
        float run_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&run_time_ms, pytorch_start_event, pytorch_end_event));
        total_pytorch_time += run_time_ms;
    }
    
    float pytorch_time_ms = total_pytorch_time / num_runs;
    
    // 清理PyTorch事件
    CHECK_CUDA(cudaEventDestroy(pytorch_start_event));
    CHECK_CUDA(cudaEventDestroy(pytorch_end_event));
    
    // 重新计算一次结果用于精度对比
    std::vector<torch::Tensor> expert_outputs;
    for (int i = 0; i < num_experts; ++i) {
        auto weight_expert = weights_raw_tensor[i];
        auto bias_expert = biases_tensor[i];
        auto output_slice = torch::mm(input_base_tensor, weight_expert) + bias_expert.unsqueeze(0);
        output_slice = torch::relu(output_slice);
        expert_outputs.push_back(output_slice);
    }
    auto pytorch_output = torch::cat(expert_outputs, 0);
    
    // 将CUDA结果复制回CPU
    std::vector<cutlass::bfloat16_t> cuda_output_bf16(total_rows * output_dim);
    CHECK_CUDA(cudaMemcpy(cuda_output_bf16.data(), d_output_bf16, output_size, cudaMemcpyDeviceToHost));
    
    // 转换CUDA结果到FP32
    std::vector<float> cuda_output_fp32(total_rows * output_dim);
    for (int i = 0; i < total_rows * output_dim; ++i) {
        cuda_output_fp32[i] = bf16_to_float(cuda_output_bf16[i]);
    }
    
    // 将PyTorch结果复制到CPU
    auto pytorch_output_cpu = pytorch_output.cpu();
    auto pytorch_output_data = pytorch_output_cpu.data_ptr<float>();
    
    // 立即输出部分结果用于对比 - 与simple_test.cpp对齐
    std::cout << std::endl << "CUDA BF16 输出 (专家0, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << cuda_output_fp32[i] << " ";
    std::cout << std::endl;
    
    std::cout << "PyTorch FP32 输出 (专家0, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << expert_outputs[0][0][i].item<float>() << " ";
    std::cout << std::endl;

    std::cout << std::endl << "CUDA BF16 输出 (专家1, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << cuda_output_fp32[batch_size * output_dim + i] << " ";
    std::cout << std::endl;

    std::cout << "PyTorch FP32 输出 (专家1, 前8个元素):" << std::endl;
    for(int i=0; i<8; ++i) std::cout << expert_outputs[1][0][i].item<float>() << " ";
    std::cout << std::endl;
    
    // 计算误差 - 直接对比对应的专家
    double max_error = 0.0;
    double total_error = 0.0;
    int total_elements = total_rows * output_dim;
    
    for (int expert = 0; expert < num_experts; ++expert) {
        double expert_max_error = 0.0;
        double expert_total_error = 0.0;
        int expert_elements = batch_size * output_dim;
        
        auto torch_expert_output = expert_outputs[expert];
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < output_dim; ++j) {
                int cuda_idx = expert * (batch_size * output_dim) + i * output_dim + j;
                float cuda_val = cuda_output_fp32[cuda_idx];
                float torch_val = torch_expert_output[i][j].item<float>();
                double error = std::abs(cuda_val - torch_val);
                expert_max_error = std::max(expert_max_error, error);
                expert_total_error += error;
            }
        }
        
        double expert_mean_error = expert_total_error / expert_elements;
        std::cout << "专家 " << expert 
                  << " 最大误差: " << expert_max_error 
                  << ", 平均误差: " << expert_mean_error << std::endl;
        
        max_error = std::max(max_error, expert_max_error);
        total_error += expert_total_error;
    }
    
    double mean_error = total_error / total_elements;
    std::cout << std::endl << "总最大绝对误差: " << max_error << std::endl;
    std::cout << "总平均绝对误差: " << mean_error << std::endl;
    // BF16精度下，误差阈值应该更宽松
    // BF16的尾数精度约为FP32的1/64，所以误差阈值应该相应调整
    bool test_passed = max_error < 0.05;  // 5%的绝对误差阈值对于BF16是合理的
    std::cout << "测试" << (test_passed ? "通过" : "失败") << " (CUDA BF16 vs PyTorch FP32)" << std::endl;
    
    // 计算相对误差 - 使用第一个非零值作为参考
    double reference_value = 0.0;
    for (int i = 0; i < total_elements; ++i) {
        if (std::abs(pytorch_output_data[i]) > 1e-6) {
            reference_value = std::abs(pytorch_output_data[i]);
            break;
        }
    }
    if (reference_value > 1e-8) {
        double relative_error = max_error / reference_value;
        std::cout << "最大相对误差: " << (relative_error * 100) << "%" << std::endl;
    } else {
        std::cout << "最大相对误差: 无法计算 (参考值过小)" << std::endl;
    }
    
    std::cout << "\n=== 性能对比 (基于" << num_runs << "次运行平均值) ===" << std::endl;
    std::cout << "CUDA Kernel (BF16): " << cuda_time_ms << " ms" << std::endl;
    std::cout << "PyTorch (FP32): " << pytorch_time_ms << " ms" << std::endl;
    std::cout << "加速比: " << (pytorch_time_ms / cuda_time_ms) << "x" << std::endl;
    
    // 计算吞吐量 (FLOPS)
    long long total_flops = (long long)total_rows * input_dim * output_dim * 2; // 2 for multiply-add
    double cuda_throughput = (double)total_flops / (cuda_time_ms / 1000.0) / 1e12; // TFLOPS
    double pytorch_throughput = (double)total_flops / (pytorch_time_ms / 1000.0) / 1e12; // TFLOPS
    
    std::cout << "\n=== 吞吐量分析 ===" << std::endl;
    std::cout << "总FLOPS: " << total_flops << std::endl;
    std::cout << "CUDA Kernel吞吐量: " << cuda_throughput << " TFLOPS" << std::endl;
    std::cout << "PyTorch吞吐量: " << pytorch_throughput << " TFLOPS" << std::endl;
    std::cout << "吞吐量提升: " << (cuda_throughput / pytorch_throughput) << "x" << std::endl;
    
    // 计算内存带宽
    size_t input_bytes = total_rows * input_dim * sizeof(cutlass::bfloat16_t);
    size_t weight_bytes = num_experts * input_dim * output_dim * sizeof(cutlass::bfloat16_t);
    size_t output_bytes = total_rows * output_dim * sizeof(cutlass::bfloat16_t);
    size_t total_memory_bytes = input_bytes + weight_bytes + output_bytes;
    
    double cuda_memory_bandwidth = (double)total_memory_bytes / (cuda_time_ms / 1000.0) / 1e9; // GB/s
    double pytorch_memory_bandwidth = (double)total_memory_bytes / (pytorch_time_ms / 1000.0) / 1e9; // GB/s
    
    std::cout << "\n=== 内存带宽分析 ===" << std::endl;
    std::cout << "总内存访问: " << total_memory_bytes / 1024 / 1024 << " MB" << std::endl;
    std::cout << "CUDA Kernel内存带宽: " << cuda_memory_bandwidth << " GB/s" << std::endl;
    std::cout << "PyTorch内存带宽: " << pytorch_memory_bandwidth << " GB/s" << std::endl;
    
    // 计算效率指标
    double cuda_efficiency = cuda_throughput / pytorch_throughput * 100; // 相对于PyTorch的效率百分比
    std::cout << "\n=== 效率分析 ===" << std::endl;
    std::cout << "CUDA相对于PyTorch的效率: " << cuda_efficiency << "%" << std::endl;
    std::cout << "BF16 vs FP32 理论加速比: 2x (理论上限)" << std::endl;
    std::cout << "实际加速比: " << (pytorch_time_ms / cuda_time_ms) << "x" << std::endl;
    
    // 清理GPU内存
    CHECK_CUDA(cudaFree(d_input_bf16));
    CHECK_CUDA(cudaFree(d_weights_bf16));
    CHECK_CUDA(cudaFree(d_biases_bf16));
    CHECK_CUDA(cudaFree(d_output_bf16));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    
    return test_passed ? 0 : 1;
}
