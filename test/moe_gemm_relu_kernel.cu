/*
 * MoE GEMM ReLU 测试内核
 * 实现基于CUTLASS的MoeFCGemm，尾处理使用ReLU激活函数
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/bfloat16.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"

#include "moe_gemm_relu_kernel.h"

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================== 设备端后处理：BatchNorm + Swish (SiLU) ==============================
template <typename T>
__global__ void broadcast_bias_rows_kernel(
    T* __restrict__ out,            // [batch_count * M, N] RowMajor, grouped按expert顺序拼接
    const T* __restrict__ bias,     // [batch_count, N]
    int M, int N, int batch_count)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (b >= batch_count || m >= M || n >= N) return;
    int64_t row_index = (int64_t)b * M + m;
    out[row_index * (int64_t)N + n] += bias[b * N + n];
}
template <typename T>
__global__ void bn_silu_epilogue_kernel_bf16(
    T* __restrict__ y,                 // [total_tokens, output_dim]
    const int64_t* __restrict__ expert_offsets, // [num_experts + 1]
    int output_dim,
    const float* __restrict__ gamma,   // [num_experts, output_dim]
    const float* __restrict__ beta,    // [num_experts, output_dim]
    const float* __restrict__ mean,    // [num_experts, output_dim]
    const float* __restrict__ var,     // [num_experts, output_dim]
    float eps,
    int num_experts)
{
    using ToFloat = cutlass::NumericConverter<float, T>;
    using ToBf16  = cutlass::NumericConverter<T, float>;

    int c = blockIdx.x * blockDim.x + threadIdx.x; // channel
    int e = blockIdx.z; // expert index

    if (e >= num_experts || c >= output_dim) return;

    int64_t start = (e == 0) ? 0 : expert_offsets[e - 1];
    int64_t end   = expert_offsets[e];
    for (int64_t r = start; r < end; ++r) {
        int64_t idx = r * (int64_t)output_dim + c;
        float v = ToFloat()(y[idx]);
        int64_t pc = (int64_t)e * output_dim + c;
        float g = gamma[pc];
        float b = beta[pc];
        float m = mean[pc];
        float s2 = var[pc];
        float xhat = (v - m) * rsqrtf(s2 + eps);
        float bn = xhat * g + b;
        float sw = bn * (1.f / (1.f + expf(-bn))); // SiLU
        y[idx] = ToBf16()(sw);
    }
}

// ============================== Epilogue操作定义 =================================

struct EpilogueOpDefault {};
struct EpilogueOpReLU {};
struct EpilogueOpSwish {};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op> 
struct Epilogue {};

constexpr auto DefaultScaleMode = cutlass::epilogue::thread::ScaleType::Default;

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefault>
{
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpReLU>
{
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpSwish>
{
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};

// ============================== MoE GEMM ReLU Kernel =================================

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void moeGemmKernelLauncherImpl(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    const int multi_processor_count)
{
    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    // For BF16, we directly use cutlass::bfloat16_t as the input type
    using ElementType = typename cutlass::platform::conditional<
        cutlass::platform::is_same<T, cutlass::half_t>::value, cutlass::half_t, T
    >::type;

    using CutlassWeightType = typename cutlass::platform::conditional<
        cutlass::platform::is_same<WeightType, cutlass::half_t>::value, cutlass::half_t, WeightType
    >::type;

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    using EpilogueOp = typename Epilogue<ElementType,
        MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    // Finally, set up the kernel.
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
        cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
        typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
        typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kGroupScheduleMode>;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
    const int threadblock_count = multi_processor_count * occupancy;

    typename EpilogueOp::Params epilogue_op(
        ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

    const int group_size = gemm_k;
    typename GemmGrouped::Arguments args(num_experts, threadblock_count, group_size, epilogue_op,
        reinterpret_cast<const ElementType*>(A),
        reinterpret_cast<const CutlassWeightType*>(B),
        nullptr, // weight_scales
        biases ? reinterpret_cast<const ElementType*>(biases) : nullptr, // ptr_C (biases)
        reinterpret_cast<ElementType*>(C), // ptr_D (output)
        total_rows_before_expert, 
        gemm_n, gemm_k);

    GemmGrouped gemm;

#ifndef HOME_MOE_SILENT
    std::cout << "Creating GEMM with " << num_experts << " experts, " 
              << num_rows << " total rows, " << gemm_n << "x" << gemm_k << std::endl;
#endif

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
#ifndef HOME_MOE_SILENT
        std::cerr << "MoE FC kernel will fail for params. Status: " << int(can_implement) << std::endl;
#endif
        return;
    } else {
#ifndef HOME_MOE_SILENT
        std::cout << "Kernel can_implement: SUCCESS" << std::endl;
#endif
    }

    auto init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
#ifndef HOME_MOE_SILENT
        std::cerr << "Failed to initialize cutlass variable batched gemm. Status: " << int(init_status) << std::endl;
#endif
        return;
    } else {
#ifndef HOME_MOE_SILENT
        std::cout << "Kernel initialize: SUCCESS" << std::endl;
#endif
    }
    
#ifndef HOME_MOE_SILENT
    std::cout << "Running GEMM kernel..." << std::endl;
#endif
    auto run_status = gemm.run();
    if (run_status != cutlass::Status::kSuccess) {
#ifndef HOME_MOE_SILENT
        std::cerr << "Failed to run cutlass variable batched gemm. Status: " << int(run_status) << std::endl;
#endif
        return;
    } else {
#ifndef HOME_MOE_SILENT
        std::cout << "Kernel run: SUCCESS" << std::endl;
#endif
    }
}


// ============================== C接口实现 =================================

extern "C" {

void launch_moe_gemm_relu_kernel(
    const float* input,           // [total_tokens, input_dim]
    const float* weights,         // [num_experts, input_dim, output_dim]
    const float* biases,          // [num_experts, output_dim] or nullptr
    float* output,                // [total_tokens, output_dim]
    const int64_t* expert_offsets, // [num_experts + 1] - cumulative token counts
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
) {
    // 获取GPU多处理器数量
    int multi_processor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

    // 启动MoE GEMM ReLU kernel
    moeGemmKernelLauncherImpl<float, float, cutlass::arch::Sm80, EpilogueOpReLU,
                              cutlass::gemm::GemmShape<128, 128, 8>,
                              cutlass::gemm::GemmShape<64, 64, 8>, 2>(
        input,
        weights,
        nullptr, // weight_scales
        use_bias ? biases : nullptr,
        output,
        const_cast<int64_t*>(expert_offsets),
        total_tokens,     // num_rows
        output_dim,       // gemm_n
        input_dim,        // gemm_k
        num_experts,
        multi_processor_count
    );

    CHECK_CUDA(cudaGetLastError());
}

void launch_moe_gemm_kernel(
    const float* input,           // [total_tokens, input_dim]
    const float* weights,         // [num_experts, input_dim, output_dim]
    const float* biases,          // [num_experts, output_dim] or nullptr
    float* output,                // [total_tokens, output_dim]
    const int64_t* expert_offsets, // [num_experts + 1] - cumulative token counts
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
) {
    // 获取GPU多处理器数量
    int multi_processor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

    // 启动MoE GEMM kernel
    moeGemmKernelLauncherImpl<float, float, cutlass::arch::Sm80, EpilogueOpDefault,
                              cutlass::gemm::GemmShape<128, 128, 8>,
                              cutlass::gemm::GemmShape<64, 64, 8>, 2>(
        input,
        weights,
        nullptr, // weight_scales
        use_bias ? biases : nullptr,
        output,
        const_cast<int64_t*>(expert_offsets),
        total_tokens,     // num_rows
        output_dim,       // gemm_n
        input_dim,        // gemm_k
        num_experts,
        multi_processor_count
    );

    CHECK_CUDA(cudaGetLastError());
}


// BF16版本的MoE GEMM kernel
void launch_moe_gemm_relu_kernel_bf16(
    const cutlass::bfloat16_t* input,           // [total_tokens, input_dim]
    const cutlass::bfloat16_t* weights,         // [num_experts, input_dim, output_dim]
    const cutlass::bfloat16_t* biases,          // [num_experts, output_dim] or nullptr
    cutlass::bfloat16_t* output,                // [total_tokens, output_dim]
    const int64_t* expert_offsets, // [num_experts + 1] - cumulative token counts
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
) {
    // 获取GPU多处理器数量
    int multi_processor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

    // 如果环境变量启用 batched，则使用 CUTLASS GemmBatched（要求 per-batch M/N/K 相同，行按 expert 均分）；否则沿用 grouped
    const char* use_batched = getenv("HOME_MOE_USE_BATCHED");
    bool enable_batched = (use_batched && std::string(use_batched) == "1");

    if (!enable_batched) {
        // 原 grouped 实现
        moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpReLU,
                                  cutlass::gemm::GemmShape<256, 128, 32>,
                                  cutlass::gemm::GemmShape<64, 64, 32>, 4>(
            input,
            weights,
            nullptr, // weight_scales
            use_bias ? biases : nullptr,
            output,
            const_cast<int64_t*>(expert_offsets),
            total_tokens,     // num_rows
            output_dim,       // gemm_n
            input_dim,        // gemm_k
            num_experts,
            multi_processor_count
        );
    } else {
        // Batched: 假设每个 expert 行数相等：rows_per_expert = total_tokens / num_experts
        int rows_per_expert = total_tokens / num_experts;
        // 校验可整除
        if (rows_per_expert * num_experts != total_tokens) {
            // 回退到 grouped
            moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpReLU,
                                      cutlass::gemm::GemmShape<256, 128, 32>,
                                      cutlass::gemm::GemmShape<64, 64, 32>, 4>(
                input, weights, nullptr, use_bias ? biases : nullptr, output,
                const_cast<int64_t*>(expert_offsets), total_tokens, output_dim, input_dim,
                num_experts, multi_processor_count);
        } else {
            using Gemm = cutlass::gemm::device::GemmBatched<
                cutlass::bfloat16_t, cutlass::layout::RowMajor,
                cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
                cutlass::bfloat16_t, cutlass::layout::RowMajor>;

            Gemm gemm_op;

            int m = rows_per_expert;
            int n = output_dim;
            int k = input_dim;
            int lda = k;           // RowMajor: leading dimension = K for A[M,K]
            int ldb = k;           // ColumnMajor: leading dimension = K for B[K,N]
            int ldc = n;           // RowMajor: leading dimension = N for C[M,N]
            long long stride_A = (long long)lda * m;
            long long stride_B = (long long)ldb * n;
            long long stride_C = (long long)ldc * m;

            auto status = gemm_op({
                {m, n, k},
                {input, lda}, stride_A,
                {weights, ldb}, stride_B,
                {output, ldc}, stride_C,
                {output, ldc}, stride_C,
                {cutlass::bfloat16_t(1.f), use_bias ? cutlass::bfloat16_t(1.f) : cutlass::bfloat16_t(0.f)},
                num_experts
            });

            if (status != cutlass::Status::kSuccess) {
                // 回退到 grouped
                moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpReLU,
                                          cutlass::gemm::GemmShape<256, 128, 32>,
                                          cutlass::gemm::GemmShape<64, 64, 32>, 4>(
                    input, weights, nullptr, use_bias ? biases : nullptr, output,
                    const_cast<int64_t*>(expert_offsets), total_tokens, output_dim, input_dim,
                    num_experts, multi_processor_count);
            } else if (use_bias) {
                // GemmBatched 不直接支持偏置广播到输出，这里补一个逐行加 bias 的 kernel
                dim3 block(128, 1, 1);
                dim3 grid((n + block.x - 1) / block.x, m, num_experts);
                broadcast_bias_rows_kernel<<<grid, block>>>(output, biases, m, n, num_experts);
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }

    CHECK_CUDA(cudaGetLastError());
}

void launch_moe_gemm_bn_silu_kernel_bf16(
    const cutlass::bfloat16_t* input,           // [total_tokens, input_dim]
    const cutlass::bfloat16_t* weights,         // [num_experts, input_dim, output_dim]
    const cutlass::bfloat16_t* biases,          // [num_experts, output_dim] or nullptr
    cutlass::bfloat16_t* output,                // [total_tokens, output_dim]
    const int64_t* expert_offsets,              // [num_experts + 1]
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    const float* bn_gamma,         // [num_experts, output_dim]
    const float* bn_beta,          // [num_experts, output_dim]
    const float* running_mean,     // [num_experts, output_dim]
    const float* running_var,      // [num_experts, output_dim]
    float eps,
    bool use_bias)
{
    // 获取GPU多处理器数量
    int multi_processor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

    // 先进行 GEMM (无激活) —— 支持 GemmBatched 或 Grouped
    const char* use_batched = getenv("HOME_MOE_USE_BATCHED");
    bool enable_batched = (use_batched && std::string(use_batched) == "1");

    if (!enable_batched) {
        // 原 Grouped GEMM，带 bias 融合（通过自定义 epilogue）
        moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpDefault,
                                  cutlass::gemm::GemmShape<256, 128, 32>,
                                  cutlass::gemm::GemmShape<64, 64, 32>, 4>(
            input, weights, nullptr, use_bias ? biases : nullptr, output,
            const_cast<int64_t*>(expert_offsets), total_tokens, output_dim, input_dim, num_experts,
            multi_processor_count);
    } else {
        // Batched GEMM 路径：要求每 expert 行数一致
        int rows_per_expert = total_tokens / num_experts;
        if (rows_per_expert * num_experts != total_tokens) {
            // 回退 Grouped
            moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpDefault,
                                      cutlass::gemm::GemmShape<256, 128, 32>,
                                      cutlass::gemm::GemmShape<64, 64, 32>, 4>(
                input, weights, nullptr, use_bias ? biases : nullptr, output,
                const_cast<int64_t*>(expert_offsets), total_tokens, output_dim, input_dim, num_experts,
                multi_processor_count);
        } else {
            using Gemm = cutlass::gemm::device::GemmBatched<
                cutlass::bfloat16_t, cutlass::layout::RowMajor,
                cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
                cutlass::bfloat16_t, cutlass::layout::RowMajor>;

            Gemm gemm_op;

            int m = rows_per_expert;
            int n = output_dim;
            int k = input_dim;
            int lda = k;  // RowMajor [M,K]
            int ldb = k;  // ColumnMajor [K,N]
            int ldc = n;  // RowMajor [M,N]
            long long stride_A = (long long)lda * m;
            long long stride_B = (long long)ldb * n;
            long long stride_C = (long long)ldc * m;

            auto status = gemm_op({
                {m, n, k},
                {input, lda}, stride_A,
                {weights, ldb}, stride_B,
                {output, ldc}, stride_C,
                {output, ldc}, stride_C,
                {cutlass::bfloat16_t(1.f), cutlass::bfloat16_t(0.f)},
                num_experts
            });

            if (status != cutlass::Status::kSuccess) {
                // 回退 Grouped
                moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpDefault,
                                          cutlass::gemm::GemmShape<256, 128, 32>,
                                          cutlass::gemm::GemmShape<64, 64, 32>, 4>(
                    input, weights, nullptr, use_bias ? biases : nullptr, output,
                    const_cast<int64_t*>(expert_offsets), total_tokens, output_dim, input_dim, num_experts,
                    multi_processor_count);
            } else if (use_bias) {
                // 补加 bias 到每行（BN+SiLU 前）
                dim3 block_bias(128, 1, 1);
                dim3 grid_bias((n + block_bias.x - 1) / block_bias.x, m, num_experts);
                broadcast_bias_rows_kernel<<<grid_bias, block_bias>>>(output, biases, m, n, num_experts);
                CHECK_CUDA(cudaGetLastError());
            }
        }
    }

    // 然后在输出上执行 BN + SiLU（使用提供的 running_mean/var 和 gamma/beta）
    dim3 block(128, 1, 1);
    dim3 grid((output_dim + block.x - 1) / block.x,
              1,
              num_experts);

    // 每个 expert 的行范围：通过 expert_offsets 控制 y 方向循环，使用二维线程分布可进一步优化
    // 这里选择在 kernel 内根据 expert_offsets 做行界限制，按列并行
    bn_silu_epilogue_kernel_bf16<cutlass::bfloat16_t><<<grid, block>>>(
        output,
        expert_offsets,
        output_dim,
        bn_gamma,
        bn_beta,
        running_mean,
        running_var,
        eps,
        num_experts);

    CHECK_CUDA(cudaGetLastError());
}

void launch_moe_gemm_kernel_bf16(
    const cutlass::bfloat16_t* input,           // [total_tokens, input_dim]
    const cutlass::bfloat16_t* weights,         // [num_experts, input_dim, output_dim]
    const cutlass::bfloat16_t* biases,          // [num_experts, output_dim] or nullptr
    cutlass::bfloat16_t* output,                // [total_tokens, output_dim]
    const int64_t* expert_offsets, // [num_experts + 1] - cumulative token counts
    int num_experts,
    int total_tokens,
    int input_dim,
    int output_dim,
    bool use_bias
) {
    // 获取GPU多处理器数量
    int multi_processor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));

    // 启动MoE GEMM kernel (BF16版本)
    moeGemmKernelLauncherImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpDefault,
                              cutlass::gemm::GemmShape<256, 128, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>, 4>(
        input,
        weights,
        nullptr, // weight_scales
        use_bias ? biases : nullptr,
        output,
        const_cast<int64_t*>(expert_offsets),
        total_tokens,     // num_rows
        output_dim,       // gemm_n
        input_dim,        // gemm_k
        num_experts,
        multi_processor_count
    );

    CHECK_CUDA(cudaGetLastError());
}

} // extern "C"
