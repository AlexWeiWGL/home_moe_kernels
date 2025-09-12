/*
 * 测试优化的MoE GEMM kernel - 共享输入版本
 * 对比原始版本和优化版本的性能和正确性
 */

 #include <iostream>
 #include <vector>
 #include <chrono>
 #include <random>
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
 
 // 生成随机数据
 std::vector<float> generate_random_data(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
     std::random_device rd;
     std::mt19937 gen(rd());
     std::uniform_real_distribution<float> dis(min_val, max_val);
     
     std::vector<float> data(size);
     for (size_t i = 0; i < size; ++i) {
         data[i] = dis(gen);
     }
     return data;
 }
 
 // 验证结果正确性
 bool verify_results(const float* result1, const float* result2, size_t size, float tolerance = 1e-5f) {
     for (size_t i = 0; i < size; ++i) {
         if (std::abs(result1[i] - result2[i]) > tolerance) {
             std::cout << "Mismatch at index " << i << ": " << result1[i] << " vs " << result2[i] << std::endl;
             return false;
         }
     }
     return true;
 }
 
 // 计算内存使用量
 size_t calculate_memory_usage(int num_experts, int batch_size, int input_dim, int output_dim) {
     // 输入数据大小
     size_t input_size = batch_size * input_dim * sizeof(float);
     
     // 权重数据大小
     size_t weight_size = num_experts * input_dim * output_dim * sizeof(float);
     
     // 输出数据大小
     size_t output_size = num_experts * batch_size * output_dim * sizeof(float);
     
     return input_size + weight_size + output_size;
 }
 
 int main() {
     std::cout << "=== 优化MoE GEMM Kernel测试 (共享输入版本) ===" << std::endl;
     
     // 测试参数
     const int num_experts = 10;
     const int batch_size = 4096;
     const int input_dim = 1408;
     const int output_dim = 704;
     const bool use_bias = true;
     
     std::cout << "测试配置: " << num_experts << " experts, " 
               << batch_size << " batch_size, " 
               << input_dim << " -> " << output_dim << std::endl;
     
     // 设置CUDA设备
     torch::Device device(torch::kCUDA, 0);
     
     // 1. 生成测试数据
     std::cout << "\n1. 生成测试数据..." << std::endl;
     
     // 原始输入数据 (只生成一次)
     auto h_input = generate_random_data(batch_size * input_dim);
     auto h_weights = generate_random_data(num_experts * input_dim * output_dim);
     auto h_biases = generate_random_data(num_experts * output_dim);
 
    // 为优化版本的kernel准备权重: 保持 [E, K, N] 布局，但确保内存对齐
    // 内核期望 [num_experts, input_dim, output_dim] 布局
    auto h_weights_transposed = std::vector<float>(num_experts * input_dim * output_dim);
    for (int e = 0; e < num_experts; ++e) {
        for (int k = 0; k < input_dim; ++k) {
            for (int n = 0; n < output_dim; ++n) {
                // 保持原始布局 [E, K, N]，不需要转置
                h_weights_transposed[e * input_dim * output_dim + k * output_dim + n] = 
                    h_weights[e * input_dim * output_dim + k * output_dim + n];
            }
        }
    }
     
     // 2. 分配GPU内存
     std::cout << "\n2. 分配GPU内存..." << std::endl;
     
     float *d_input, *d_weights, *d_biases, *d_output_original, *d_output_optimized;
     
     CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
     CHECK_CUDA(cudaMalloc(&d_weights, num_experts * input_dim * output_dim * sizeof(float)));
     CHECK_CUDA(cudaMalloc(&d_biases, num_experts * output_dim * sizeof(float)));
     CHECK_CUDA(cudaMalloc(&d_output_original, num_experts * batch_size * output_dim * sizeof(float)));
     CHECK_CUDA(cudaMalloc(&d_output_optimized, num_experts * batch_size * output_dim * sizeof(float)));
     
     // 复制数据到GPU
     CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
     // 为原始版本复制未转置的权重
     CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), num_experts * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_biases, h_biases.data(), num_experts * output_dim * sizeof(float), cudaMemcpyHostToDevice));
     
     // 3. 测试原始版本 (需要复制输入)
     std::cout << "\n3. 测试原始版本 (输入复制版本)..." << std::endl;
     
     // 为原始版本复制输入数据
     std::vector<float> h_input_replicated(num_experts * batch_size * input_dim);
     for (int i = 0; i < num_experts; ++i) {
         memcpy(h_input_replicated.data() + i * batch_size * input_dim, 
                h_input.data(), 
                batch_size * input_dim * sizeof(float));
     }
     
     float *d_input_replicated;
     CHECK_CUDA(cudaMalloc(&d_input_replicated, num_experts * batch_size * input_dim * sizeof(float)));
     CHECK_CUDA(cudaMemcpy(d_input_replicated, h_input_replicated.data(), 
                          num_experts * batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
     
     // 创建expert_offsets
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
     
     int64_t *d_expert_offsets;
     CHECK_CUDA(cudaMalloc(&d_expert_offsets, num_experts * sizeof(int64_t)));
     CHECK_CUDA(cudaMemcpy(d_expert_offsets, expert_offsets.data(), num_experts * sizeof(int64_t), cudaMemcpyHostToDevice));
     
     // 执行原始版本 - 使用CUDA events测量时间，多次运行取平均
     cudaEvent_t start_event, stop_event;
     CHECK_CUDA(cudaEventCreate(&start_event));
     CHECK_CUDA(cudaEventCreate(&stop_event));
     
     const int num_warmup = 3;
     const int num_runs = 10;
     
     // 预热运行
     for (int i = 0; i < num_warmup; ++i) {
         launch_moe_gemm_relu_kernel(
             d_input_replicated,
             d_weights,
             d_biases,
             d_output_original,
             d_expert_offsets,
             num_experts,
             num_experts * batch_size,  // total_tokens
             input_dim,
             output_dim,
             use_bias
         );
     }
     
     // 正式测量
     float total_time = 0.0f;
     for (int i = 0; i < num_runs; ++i) {
         CHECK_CUDA(cudaEventRecord(start_event));
         
         launch_moe_gemm_relu_kernel(
             d_input_replicated,
             d_weights,
             d_biases,
             d_output_original,
             d_expert_offsets,
             num_experts,
             num_experts * batch_size,  // total_tokens
             input_dim,
             output_dim,
             use_bias
         );
         
         CHECK_CUDA(cudaEventRecord(stop_event));
         CHECK_CUDA(cudaEventSynchronize(stop_event));
         
         float duration_ms;
         CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start_event, stop_event));
         total_time += duration_ms;
     }
     
     float original_duration_ms = total_time / num_runs;
     
     // 4. 测试优化版本 (共享输入) - 使用CUDA events测量时间，多次运行取平均
     std::cout << "\n4. 测试优化版本 (共享输入版本)..." << std::endl;
 
     // 为优化版本分配并复制转置后的权重
     float *d_weights_transposed;
     CHECK_CUDA(cudaMalloc(&d_weights_transposed, num_experts * input_dim * output_dim * sizeof(float)));
     CHECK_CUDA(cudaMemcpy(d_weights_transposed, h_weights_transposed.data(), num_experts * input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
     
     // 预热运行
     for (int i = 0; i < num_warmup; ++i) {
         launch_moe_gemm_relu_kernel_optimized(
             d_input,           // 直接使用原始输入，不复制
             d_weights_transposed,
             d_biases,
             d_output_optimized,
             num_experts,
             batch_size,
             input_dim,
             output_dim,
             use_bias
         );
     }
     
    // 正式测量
    total_time = 0.0f;
    for (int i = 0; i < num_runs; ++i) {
        CHECK_CUDA(cudaEventRecord(start_event));
        
        // 添加CUDA错误检查
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error before kernel launch: " << cudaGetErrorString(err) << std::endl;
        }
        
        launch_moe_gemm_relu_kernel_optimized(
            d_input,           // 直接使用原始输入，不复制
            d_weights_transposed,
            d_biases,
            d_output_optimized,
            num_experts,
            batch_size,
            input_dim,
            output_dim,
            use_bias
        );
        
        // 检查内核执行后的错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
        }
        
        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
         
         float duration_ms;
         CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start_event, stop_event));
         total_time += duration_ms;
     }
     
     float optimized_duration_ms = total_time / num_runs;
     
     // 5. 验证结果正确性
     std::cout << "\n5. 验证结果正确性..." << std::endl;
     
     std::vector<float> h_output_original(num_experts * batch_size * output_dim);
     std::vector<float> h_output_optimized(num_experts * batch_size * output_dim);
     
     CHECK_CUDA(cudaMemcpy(h_output_original.data(), d_output_original, 
                          num_experts * batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
     CHECK_CUDA(cudaMemcpy(h_output_optimized.data(), d_output_optimized, 
                          num_experts * batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
     
     bool results_match = verify_results(h_output_original.data(), h_output_optimized.data(), 
                                        num_experts * batch_size * output_dim);
     
     if (results_match) {
         std::cout << "✓ 结果验证通过！两个版本输出完全一致" << std::endl;
     } else {
         std::cout << "✗ 结果验证失败！两个版本输出不一致" << std::endl;
         return 1;
     }
     
     // 6. 性能对比
     std::cout << "\n6. 性能对比..." << std::endl;
     
     double speedup = original_duration_ms / optimized_duration_ms;
     
     std::cout << "原始版本执行时间: " << original_duration_ms << " ms (平均" << num_runs << "次运行)" << std::endl;
     std::cout << "优化版本执行时间: " << optimized_duration_ms << " ms (平均" << num_runs << "次运行)" << std::endl;
     std::cout << "性能提升: " << speedup << "x" << std::endl;
     
     // 7. 内存使用对比
     std::cout << "\n7. 内存使用对比..." << std::endl;
     
     size_t original_memory = calculate_memory_usage(num_experts, batch_size, input_dim, output_dim) + 
                             (num_experts * batch_size * input_dim * sizeof(float)); // 额外的输入复制
     size_t optimized_memory = calculate_memory_usage(num_experts, batch_size, input_dim, output_dim);
     
     double memory_savings = (double)(original_memory - optimized_memory) / original_memory * 100.0;
     
     std::cout << "原始版本内存使用: " << original_memory / (1024 * 1024) << " MB" << std::endl;
     std::cout << "优化版本内存使用: " << optimized_memory / (1024 * 1024) << " MB" << std::endl;
     std::cout << "内存节省: " << memory_savings << "%" << std::endl;
     
     // 8. 清理资源
     std::cout << "\n8. 清理资源..." << std::endl;
     
     CHECK_CUDA(cudaEventDestroy(start_event));
     CHECK_CUDA(cudaEventDestroy(stop_event));
     
     CHECK_CUDA(cudaFree(d_input));
     CHECK_CUDA(cudaFree(d_weights));
     CHECK_CUDA(cudaFree(d_biases));
     CHECK_CUDA(cudaFree(d_output_original));
     CHECK_CUDA(cudaFree(d_output_optimized));
     CHECK_CUDA(cudaFree(d_input_replicated));
     CHECK_CUDA(cudaFree(d_expert_offsets));
     CHECK_CUDA(cudaFree(d_weights_transposed));
     
     std::cout << "\n=== 测试完成 ===" << std::endl;
     
     return 0;
 }
 