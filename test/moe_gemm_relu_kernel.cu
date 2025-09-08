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
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
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

// ============================== Epilogue操作定义 =================================

struct EpilogueOpDefault {};
struct EpilogueOpReLU {};

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

// ============================== MoE GEMM ReLU Kernel =================================

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void moeGemmKernelLauncherImpl(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    const int multi_processor_count)
{
    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;

    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t,
            WeightType>::type;

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

    std::cout << "Creating GEMM with " << num_experts << " experts, " 
              << num_rows << " total rows, " << gemm_n << "x" << gemm_k << std::endl;

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::cerr << "MoE FC kernel will fail for params. Status: " << int(can_implement) << std::endl;
        return;
    } else {
        std::cout << "Kernel can_implement: SUCCESS" << std::endl;
    }

    auto init_status = gemm.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize cutlass variable batched gemm. Status: " << int(init_status) << std::endl;
        return;
    } else {
        std::cout << "Kernel initialize: SUCCESS" << std::endl;
    }
    
    std::cout << "Running GEMM kernel..." << std::endl;
    auto run_status = gemm.run();
    if (run_status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run cutlass variable batched gemm. Status: " << int(run_status) << std::endl;
        return;
    } else {
        std::cout << "Kernel run: SUCCESS" << std::endl;
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

// 简单的矩阵乘法 + ReLU kernel（用于对比）
__global__ void simple_gemm_relu_kernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K, bool use_bias
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        if (use_bias) {
            sum += bias[col];
        }
        
        // ReLU activation
        C[row * N + col] = fmaxf(0.0f, sum);
    }
}

void launch_simple_gemm_relu(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K, bool use_bias
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    simple_gemm_relu_kernel<<<grid, block>>>(A, B, bias, C, M, N, K, use_bias);
    CHECK_CUDA(cudaGetLastError());
}

} // extern "C"
