#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "moe_gemm_relu_kernel.h"
#include "cutlass/bfloat16.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

static inline cutlass::bfloat16_t float_to_bf16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint16_t h = (uint16_t)(x >> 16);
    return *reinterpret_cast<cutlass::bfloat16_t*>(&h);
}

static inline float bf16_to_float(cutlass::bfloat16_t bf) {
    uint16_t h = *reinterpret_cast<uint16_t*>(&bf);
    uint32_t x = ((uint32_t)h) << 16;
    return *reinterpret_cast<float*>(&x);
}

void generate_random(std::vector<float>& v, float lo, float hi, uint32_t seed=42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : v) x = dist(rng);
}

int main() {
    c10::InferenceMode guard; // 关闭 autograd
    std::cout << "=== MoE GEMM + BN + SiLU (BF16 kernel vs Torch BF16/FP32) ===" << std::endl;

    const int num_experts = 10;
    const int input_dim = 1408;
    const int output_dim = 704;
    const int total_tokens = 4096;
    const bool use_bias = true;
    const int num_iterations = 50;
    const int num_warmup = 10;
    const float eps = 1e-5f;

    torch::Device device(torch::kCUDA, 0);

    std::vector<float> h_input(total_tokens * input_dim);
    std::vector<float> h_weights(num_experts * input_dim * output_dim);
    std::vector<float> h_biases(num_experts * output_dim);

    std::vector<float> h_gamma(num_experts * output_dim);
    std::vector<float> h_beta(num_experts * output_dim);
    std::vector<float> h_mean(num_experts * output_dim);
    std::vector<float> h_var(num_experts * output_dim);

    // 限制数据范围，使其更适合BF16精度
    generate_random(h_input, -0.1f, 0.1f, 1);        // 输入范围缩小
    generate_random(h_weights, -0.05f, 0.05f, 2);     // 权重范围缩小
    generate_random(h_biases, -0.02f, 0.02f, 3);      // 偏置范围缩小
    generate_random(h_gamma, 0.9f, 1.1f, 4);          // gamma范围缩小
    generate_random(h_beta, -0.02f, 0.02f, 5);        // beta范围缩小
    generate_random(h_mean, -0.01f, 0.01f, 6);        // mean范围缩小
    generate_random(h_var, 0.05f, 0.15f, 7);          // var范围调整

    // 强制等长切分，便于 GemmBatched 验证（total_tokens 需能被 num_experts 整除）
    std::vector<int> expert_tokens(num_experts);
    int rows_per_expert = total_tokens / num_experts;
    for (int i = 0; i < num_experts; ++i) expert_tokens[i] = rows_per_expert;

    std::vector<int64_t> expert_offsets_plus1(num_experts + 1);
    expert_offsets_plus1[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        expert_offsets_plus1[i + 1] = expert_offsets_plus1[i] + expert_tokens[i];
    }
    std::vector<int64_t> cum_rows(num_experts);
    for (int i = 0; i < num_experts; ++i) cum_rows[i] = expert_offsets_plus1[i + 1];

    std::vector<float> h_input_reordered(total_tokens * input_dim);
    int in_off = 0, out_off = 0;
    for (int e = 0; e < num_experts; ++e) {
        int nt = expert_tokens[e];
        for (int t = 0; t < nt; ++t) {
            std::copy_n(&h_input[in_off * input_dim], input_dim,
                        &h_input_reordered[out_off * input_dim]);
            ++in_off; ++out_off;
        }
    }

    std::vector<float> h_weights_colmajor(num_experts * input_dim * output_dim);
    for (int e = 0; e < num_experts; ++e) {
        const float* src = &h_weights[e * input_dim * output_dim];
        float* dst = &h_weights_colmajor[e * input_dim * output_dim];
        for (int k = 0; k < input_dim; ++k) {
            for (int n = 0; n < output_dim; ++n) {
                dst[n * input_dim + k] = src[k * output_dim + n];
            }
        }
    }

    std::vector<cutlass::bfloat16_t> h_input_bf16(total_tokens * input_dim);
    std::vector<cutlass::bfloat16_t> h_weights_bf16(num_experts * input_dim * output_dim);
    std::vector<cutlass::bfloat16_t> h_biases_bf16(num_experts * output_dim);
    for (size_t i = 0; i < h_input_reordered.size(); ++i) h_input_bf16[i] = float_to_bf16(h_input_reordered[i]);
    for (size_t i = 0; i < h_weights_colmajor.size(); ++i) h_weights_bf16[i] = float_to_bf16(h_weights_colmajor[i]);
    for (size_t i = 0; i < h_biases.size(); ++i) h_biases_bf16[i] = float_to_bf16(h_biases[i]);

    cutlass::bfloat16_t *d_input_bf16, *d_weights_bf16, *d_biases_bf16, *d_output_bf16;
    int64_t *d_expert_offsets;
    float *d_gamma, *d_beta, *d_mean, *d_var;

    size_t in_bytes = h_input_bf16.size() * sizeof(cutlass::bfloat16_t);
    size_t w_bytes = h_weights_bf16.size() * sizeof(cutlass::bfloat16_t);
    size_t b_bytes = h_biases_bf16.size() * sizeof(cutlass::bfloat16_t);
    size_t out_bytes = (size_t)total_tokens * output_dim * sizeof(cutlass::bfloat16_t);
    size_t off_bytes = (size_t)num_experts * sizeof(int64_t);
    size_t bn_bytes = (size_t)num_experts * output_dim * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input_bf16, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights_bf16, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_biases_bf16, b_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_bf16, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, off_bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, bn_bytes));
    CHECK_CUDA(cudaMalloc(&d_beta, bn_bytes));
    CHECK_CUDA(cudaMalloc(&d_mean, bn_bytes));
    CHECK_CUDA(cudaMalloc(&d_var, bn_bytes));

    CHECK_CUDA(cudaMemcpy(d_input_bf16, h_input_bf16.data(), in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights_bf16, h_weights_bf16.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases_bf16, h_biases_bf16.data(), b_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, cum_rows.data(), off_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), bn_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), bn_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mean, h_mean.data(), bn_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_var, h_var.data(), bn_bytes, cudaMemcpyHostToDevice));

    // 先跑一次，确保 lazy 初始化完成
    launch_moe_gemm_bn_silu_kernel_bf16(
        d_input_bf16, d_weights_bf16, d_biases_bf16, d_output_bf16,
        d_expert_offsets,
        num_experts, total_tokens, input_dim, output_dim,
        d_gamma, d_beta, d_mean, d_var, eps, use_bias);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 精度参考（FP32）
    std::vector<cutlass::bfloat16_t> h_out_bf16(total_tokens * output_dim);
    CHECK_CUDA(cudaMemcpy(h_out_bf16.data(), d_output_bf16, out_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> h_out_cuda(total_tokens * output_dim);
    for (size_t i = 0; i < h_out_bf16.size(); ++i) h_out_cuda[i] = bf16_to_float(h_out_bf16[i]);

    auto torch_input_fp32 = torch::from_blob(h_input_reordered.data(), {total_tokens, input_dim}, torch::kFloat32).to(device);
    auto torch_weights_fp32 = torch::from_blob(h_weights.data(), {num_experts, input_dim, output_dim}, torch::kFloat32).to(device);
    auto torch_biases_fp32 = torch::from_blob(h_biases.data(), {num_experts, output_dim}, torch::kFloat32).to(device);

    // 构建 per-expert 模块（FP32）
    std::vector<torch::nn::Linear> linears_fp32;
    std::vector<torch::nn::BatchNorm1d> bns_fp32;
    linears_fp32.reserve(num_experts);
    bns_fp32.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(input_dim, output_dim).bias(use_bias));
        auto W_e = torch_weights_fp32[e].transpose(0, 1).contiguous();
        linear->weight.set_data(W_e);
        if (use_bias) linear->bias.set_data(torch_biases_fp32[e].contiguous());
        linear->eval();
        linears_fp32.push_back(linear);

        auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(output_dim).affine(true).track_running_stats(true));
        bn->weight.set_data(torch::from_blob(h_gamma.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).clone());
        bn->bias.set_data(torch::from_blob(h_beta.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).clone());
        bn->running_mean.set_data(torch::from_blob(h_mean.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).clone());
        bn->running_var.set_data(torch::from_blob(h_var.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).clone());
        bn->eval();
        bns_fp32.push_back(bn);
    }

    // 预创建 per-expert 输入/输出视图（FP32 & BF16），避免计时循环中进行 slice
    std::vector<torch::Tensor> input_views_fp32, input_views_bf16, out_views_fp32, out_views_bf16;
    input_views_fp32.reserve(num_experts);
    input_views_bf16.reserve(num_experts);
    out_views_fp32.reserve(num_experts);
    out_views_bf16.reserve(num_experts);

    auto torch_input_bf16 = torch_input_fp32.to(torch::kBFloat16);
    auto torch_out_fp32 = torch::zeros({total_tokens, output_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto torch_out_bf16 = torch::empty({total_tokens, output_dim}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    for (int e = 0; e < num_experts; ++e) {
        int start = expert_offsets_plus1[e];
        int nt = expert_tokens[e];
        if (nt == 0) {
            input_views_fp32.push_back(torch::Tensor());
            input_views_bf16.push_back(torch::Tensor());
            out_views_fp32.push_back(torch::Tensor());
            out_views_bf16.push_back(torch::Tensor());
            continue;
        }
        input_views_fp32.push_back(torch_input_fp32.narrow(0, start, nt));
        input_views_bf16.push_back(torch_input_bf16.narrow(0, start, nt));
        out_views_fp32.push_back(torch_out_fp32.narrow(0, start, nt));
        out_views_bf16.push_back(torch_out_bf16.narrow(0, start, nt));
    }

    // 为后续基线准备 max_nt
    int max_nt = 0;
    for (int e = 0; e < num_experts; ++e) max_nt = std::max(max_nt, expert_tokens[e]);

    // 参考 FP32 计算
    for (int e = 0; e < num_experts; ++e) {
        int nt = expert_tokens[e];
        if (nt == 0) continue;
        auto y = linears_fp32[e]->forward(input_views_fp32[e]);
        auto bn = bns_fp32[e]->forward(y);
        auto sw = torch::silu(bn);
        out_views_fp32[e].copy_(sw);
    }

    auto torch_out_cpu = torch_out_fp32.cpu().contiguous();
    std::vector<float> h_out_torch(torch_out_cpu.data_ptr<float>(), torch_out_cpu.data_ptr<float>() + torch_out_cpu.numel());

    double max_abs = 0.0, sum_abs = 0.0;
    double max_rel = 0.0, sum_rel = 0.0;
    int max_abs_idx = -1, max_rel_idx = -1;
    
    for (size_t i = 0; i < h_out_cuda.size(); ++i) {
        double d = std::abs(h_out_cuda[i] - h_out_torch[i]);
        double rel = (h_out_torch[i] != 0.0) ? d / std::abs(h_out_torch[i]) : 0.0;
        
        if (d > max_abs) {
            max_abs = d;
            max_abs_idx = i;
        }
        if (rel > max_rel) {
            max_rel = rel;
            max_rel_idx = i;
        }
        
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_rel += rel;
    }
    double avg_abs = sum_abs / h_out_cuda.size();
    double avg_rel = sum_rel / h_out_cuda.size();
    
    // 输出详细的精度对比信息
    std::cout << "\n=== 精度对比详情 ===" << std::endl;
    std::cout << "BF16 Kernel vs Torch FP32:" << std::endl;
    std::cout << "  最大绝对误差: " << max_abs << " (位置: " << max_abs_idx << ")" << std::endl;
    std::cout << "  平均绝对误差: " << avg_abs << std::endl;
    std::cout << "  最大相对误差: " << max_rel << " (位置: " << max_rel_idx << ")" << std::endl;
    std::cout << "  平均相对误差: " << avg_rel << std::endl;
    
    // 输出前几个值的对比
    std::cout << "\n前10个值的对比:" << std::endl;
    std::cout << "位置\tBF16 Kernel\tTorch FP32\t绝对误差\t相对误差" << std::endl;
    for (int i = 0; i < std::min(10, (int)h_out_cuda.size()); ++i) {
        double abs_err = std::abs(h_out_cuda[i] - h_out_torch[i]);
        double rel_err = (h_out_torch[i] != 0.0) ? abs_err / std::abs(h_out_torch[i]) : 0.0;
        std::cout << i << "\t" << h_out_cuda[i] << "\t\t" << h_out_torch[i] 
                  << "\t\t" << abs_err << "\t\t" << rel_err << std::endl;
    }
    
    // 输出最大误差位置的详细信息
    if (max_abs_idx >= 0) {
        std::cout << "\n最大绝对误差位置 (" << max_abs_idx << ") 的详细信息:" << std::endl;
        std::cout << "BF16 Kernel: " << h_out_cuda[max_abs_idx] << std::endl;
        std::cout << "Torch FP32:  " << h_out_torch[max_abs_idx] << std::endl;
        std::cout << "绝对误差:    " << max_abs << std::endl;
        std::cout << "相对误差:    " << max_rel << std::endl;
    }
    
    // 输出最大相对误差位置的详细信息
    if (max_rel_idx >= 0) {
        std::cout << "\n最大相对误差位置 (" << max_rel_idx << ") 的详细信息:" << std::endl;
        std::cout << "BF16 Kernel: " << h_out_cuda[max_rel_idx] << std::endl;
        std::cout << "Torch FP32:  " << h_out_torch[max_rel_idx] << std::endl;
        double abs_err_rel = std::abs(h_out_cuda[max_rel_idx] - h_out_torch[max_rel_idx]);
        double rel_err_rel = (h_out_torch[max_rel_idx] != 0.0) ? abs_err_rel / std::abs(h_out_torch[max_rel_idx]) : 0.0;
        std::cout << "绝对误差:    " << abs_err_rel << std::endl;
        std::cout << "相对误差:    " << rel_err_rel << std::endl;
        
        // 计算这个位置在哪个expert和哪个token
        int expert_idx = max_rel_idx / (total_tokens * output_dim);
        int token_idx = (max_rel_idx % (total_tokens * output_dim)) / output_dim;
        int channel_idx = max_rel_idx % output_dim;
        std::cout << "位置解析: Expert=" << expert_idx << ", Token=" << token_idx << ", Channel=" << channel_idx << std::endl;
    }
    
    // 查找一些接近0的值，看看相对误差为什么这么大
    std::cout << "\n查找接近0的值（可能导致大相对误差）:" << std::endl;
    int near_zero_count = 0;
    for (size_t i = 0; i < h_out_cuda.size() && near_zero_count < 5; ++i) {
        if (std::abs(h_out_torch[i]) < 0.001 && std::abs(h_out_torch[i]) > 1e-8) {
            double abs_err = std::abs(h_out_cuda[i] - h_out_torch[i]);
            double rel_err = (h_out_torch[i] != 0.0) ? abs_err / std::abs(h_out_torch[i]) : 0.0;
            std::cout << "位置 " << i << ": BF16=" << h_out_cuda[i] << ", FP32=" << h_out_torch[i] 
                      << ", 绝对误差=" << abs_err << ", 相对误差=" << rel_err << std::endl;
            near_zero_count++;
        }
    }

    // CUDA Events 计时 - Kernel
    cudaEvent_t start1, stop1;
    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));

    for (int it = 0; it < num_warmup; ++it) {
        launch_moe_gemm_bn_silu_kernel_bf16(
            d_input_bf16, d_weights_bf16, d_biases_bf16, d_output_bf16,
            d_expert_offsets,
            num_experts, total_tokens, input_dim, output_dim,
            d_gamma, d_beta, d_mean, d_var, eps, use_bias);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start1));
    for (int it = 0; it < num_iterations; ++it) {
        launch_moe_gemm_bn_silu_kernel_bf16(
            d_input_bf16, d_weights_bf16, d_biases_bf16, d_output_bf16,
            d_expert_offsets,
            num_experts, total_tokens, input_dim, output_dim,
            d_gamma, d_beta, d_mean, d_var, eps, use_bias);
    }
    CHECK_CUDA(cudaEventRecord(stop1));
    CHECK_CUDA(cudaEventSynchronize(stop1));
    float kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start1, stop1));
    kernel_ms /= num_iterations;

    // CUDA Events 计时 - Torch BF16（使用 nn::Linear + nn::BatchNorm1d + SiLU），避免循环内 slice
    // 构建 per-expert BF16 模块
    std::vector<torch::nn::Linear> linears_bf16;
    std::vector<torch::nn::BatchNorm1d> bns_bf16;
    linears_bf16.reserve(num_experts);
    bns_bf16.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(input_dim, output_dim).bias(use_bias));
        auto W_e = torch_weights_fp32[e].transpose(0, 1).contiguous().to(torch::kBFloat16);
        linear->weight.set_data(W_e);
        if (use_bias) linear->bias.set_data(torch_biases_fp32[e].to(torch::kBFloat16).contiguous());
        linear->to(torch::kBFloat16);
        linear->eval();
        linears_bf16.push_back(linear);

        auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(output_dim).affine(true).track_running_stats(true));
        bn->weight.set_data(torch::from_blob(h_gamma.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16).clone());
        bn->bias.set_data(torch::from_blob(h_beta.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16).clone());
        bn->running_mean.set_data(torch::from_blob(h_mean.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16).clone());
        bn->running_var.set_data(torch::from_blob(h_var.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).to(torch::kBFloat16).clone());
        bn->to(torch::kBFloat16);
        bn->eval();
        bns_bf16.push_back(bn);
    }

    cudaEvent_t start2, stop2;
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));

    CHECK_CUDA(cudaEventRecord(start2));
    for (int it = 0; it < num_iterations; ++it) {
        for (int e = 0; e < num_experts; ++e) {
            int nt = expert_tokens[e]; if (nt == 0) continue;
            auto y = linears_bf16[e]->forward(input_views_bf16[e]);
            auto bn = bns_bf16[e]->forward(y);
            auto sw = torch::silu(bn);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop2));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    float torch_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&torch_ms, start2, stop2));
    torch_ms /= num_iterations;

    std::cout << "Perf | Kernel BF16: " << kernel_ms << " ms, Torch BF16: " << torch_ms
              << " ms, Speedup: " << (torch_ms / kernel_ms) << "x" << std::endl;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Diff | Max: " << max_abs << ", Avg: " << avg_abs << std::endl;

    CHECK_CUDA(cudaEventDestroy(start1));
    CHECK_CUDA(cudaEventDestroy(stop1));
    CHECK_CUDA(cudaEventDestroy(start2));
    CHECK_CUDA(cudaEventDestroy(stop2));
    // (fold+bmm) 已移除

    // 训练态前向基线（BF16 输入）：batched BMM(BF16) + 掩码 BN(训练统计, FP32) + SiLU
    {
        auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        // 打包 BF16 输入与权重/偏置
        auto X_batched_bf16 = torch::zeros({num_experts, max_nt, input_dim}, opt_bf16);
        for (int e = 0; e < num_experts; ++e) {
            int nt = expert_tokens[e]; if (nt == 0) continue;
            X_batched_bf16[e].narrow(0, 0, nt).copy_(input_views_bf16[e]);
        }
        auto W_bf16 = torch_weights_fp32.to(torch::kBFloat16).contiguous(); // [E,K,N]
        auto b_bf16 = torch_biases_fp32.to(torch::kBFloat16).contiguous();  // [E,N]
        auto gamma_fp32 = torch::empty({num_experts, output_dim}, opt_f32);
        auto beta_fp32  = torch::empty({num_experts, output_dim}, opt_f32);
        for (int e = 0; e < num_experts; ++e) {
            gamma_fp32[e].copy_(torch::from_blob(h_gamma.data() + e * output_dim, {output_dim}, torch::kFloat32).to(device).clone());
            beta_fp32[e].copy_(torch::from_blob(h_beta.data() + e * output_dim,  {output_dim}, torch::kFloat32).to(device).clone());
        }

        // 掩码与长度
        std::vector<int> h_len = expert_tokens;
        auto len_t = torch::from_blob(h_len.data(), {num_experts}, torch::kInt32).to(device).to(torch::kFloat32).clone(); // [E]
        auto rng = torch::arange(max_nt, opt_f32); // [T]
        auto mask = (rng.unsqueeze(0).expand({num_experts, max_nt}) < len_t.unsqueeze(1)).to(torch::kFloat32); // [E,T]

        // 预热
        for (int it = 0; it < num_warmup; ++it) {
            auto Y_bf16 = torch::bmm(X_batched_bf16, W_bf16);    // [E,T,N] BF16
            Y_bf16 = Y_bf16 + b_bf16.unsqueeze(1);               // [E,1,N]
            auto Y = Y_bf16.to(torch::kFloat32);                 // upcast
            auto mean = (Y * mask.unsqueeze(-1)).sum(1) / len_t.unsqueeze(-1); // [E,N]
            auto centered = Y - mean.unsqueeze(1);               // [E,T,N]
            auto var = (centered.pow(2) * mask.unsqueeze(-1)).sum(1) / len_t.unsqueeze(-1); // [E,N]
            auto yhat = centered / torch::sqrt(var.unsqueeze(1) + eps);
            auto out = yhat * gamma_fp32.unsqueeze(1) + beta_fp32.unsqueeze(1);
            auto sw = torch::silu(out);
            (void)sw;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start4, stop4;
        CHECK_CUDA(cudaEventCreate(&start4));
        CHECK_CUDA(cudaEventCreate(&stop4));
        CHECK_CUDA(cudaEventRecord(start4));
        for (int it = 0; it < num_iterations; ++it) {
            auto Y_bf16 = torch::bmm(X_batched_bf16, W_bf16);
            Y_bf16 = Y_bf16 + b_bf16.unsqueeze(1);
            auto Y = Y_bf16.to(torch::kFloat32);
            auto mean = (Y * mask.unsqueeze(-1)).sum(1) / len_t.unsqueeze(-1);
            auto centered = Y - mean.unsqueeze(1);
            auto var = (centered.pow(2) * mask.unsqueeze(-1)).sum(1) / len_t.unsqueeze(-1);
            auto yhat = centered / torch::sqrt(var.unsqueeze(1) + eps);
            auto out = yhat * gamma_fp32.unsqueeze(1) + beta_fp32.unsqueeze(1);
            auto sw = torch::silu(out);
            (void)sw;
        }
        CHECK_CUDA(cudaEventRecord(stop4));
        CHECK_CUDA(cudaEventSynchronize(stop4));
        float torch_ms_train_fwd = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&torch_ms_train_fwd, start4, stop4));
        torch_ms_train_fwd /= num_iterations;
        std::cout << "Perf | Torch BF16 (Train-FWD, BMM+BN+SiLU): " << torch_ms_train_fwd << " ms" << std::endl;
        CHECK_CUDA(cudaEventDestroy(start4));
        CHECK_CUDA(cudaEventDestroy(stop4));
    }

    CHECK_CUDA(cudaFree(d_input_bf16));
    CHECK_CUDA(cudaFree(d_weights_bf16));
    CHECK_CUDA(cudaFree(d_biases_bf16));
    CHECK_CUDA(cudaFree(d_output_bf16));
    CHECK_CUDA(cudaFree(d_expert_offsets));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_var));

    std::cout << "=== Done ===" << std::endl;
    return 0;
}
