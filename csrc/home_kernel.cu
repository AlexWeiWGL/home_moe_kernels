/*
 * HoME (Hierarchical Mixture of Experts) CUDA Kernels - 简化版本
 * 只保留实际使用的算子
 */

#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"

#define WARP_SIZE 32
#define CEIL(a, b) ((a + b - 1) / (b))

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

// ============================== BatchNorm + SiLU 融合内核 =================================

template <typename T>
__global__ void singleExpertBatchNormSiluKernelStrided(
    T* data,
    const T* bn_weight,
    const T* bn_bias,
    const T* running_mean,
    const T* running_var,
    int batch_size,
    int hidden_dim,
    int stride,
    float epsilon = 1e-5f
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = tid % hidden_dim;
    int batch_idx = tid / hidden_dim;
    
    if (batch_idx >= batch_size || dim_idx >= hidden_dim) return;
    
    int data_idx = batch_idx * stride + dim_idx;
    
    // BatchNorm
    T normalized = (data[data_idx] - running_mean[dim_idx]) / sqrt(running_var[dim_idx] + epsilon);
    T bn_output = normalized * bn_weight[dim_idx] + bn_bias[dim_idx];
    
    // SiLU activation: x * sigmoid(x)
    T sigmoid_x = T(1.0f) / (T(1.0f) + exp(-bn_output));
    data[data_idx] = bn_output * sigmoid_x;
}

// ============================== Grouped GEMM Kernel Launcher =================================
// 从 moe_kernels.cu 引入的真正并行的 Grouped GEMM kernel
template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void genericMoeGemmKernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
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
        typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
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
        reinterpret_cast<const ElementType*>(A), reinterpret_cast<const CutlassWeightType*>(B),
        reinterpret_cast<const ElementType*>(weight_scales), reinterpret_cast<const ElementType*>(biases),
        reinterpret_cast<ElementType*>(C), total_rows_before_expert, gemm_n, gemm_k);

    GemmGrouped gemm;

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        printf("MoE FC kernel will fail for params.\n");
    }

    auto init_status = gemm.initialize(args);

    if (init_status != cutlass::Status::kSuccess) {
        printf("Failed to initialize cutlass variable batched gemm.\n");
    }
    auto run_status = gemm.run();
    if (run_status != cutlass::Status::kSuccess) {
        printf("Failed to run cutlass variable batched gemm.\n");
    }
}


// ============================== 主要接口 =================================

// HoME专家网络接口
torch::Tensor homeExpertForward(
    torch::Tensor input,                    // [batch_size, input_dim]
    torch::Tensor expert_weights_fc1,       // [num_experts, input_dim, hidden_dim]
    torch::Tensor expert_biases_fc1,        // [num_experts, hidden_dim]
    torch::Tensor expert_weights_fc2,       // [num_experts, hidden_dim, hidden_dim]
    torch::Tensor expert_biases_fc2,        // [num_experts, hidden_dim]
    torch::Tensor bn_weights,               // [num_experts, hidden_dim]
    torch::Tensor bn_biases,                // [num_experts, hidden_dim]
    torch::Tensor running_mean,             // [num_experts, hidden_dim]
    torch::Tensor running_var,              // [num_experts, hidden_dim]
    int num_experts,
    bool use_bias,
    float epsilon = 1e-5
) {
    torch::Device device(torch::kCUDA);
    
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int hidden_dim = expert_weights_fc1.size(2);
    int final_out_dim = expert_weights_fc2.size(2); // Should also be hidden_dim

    // 1. 准备 Grouped GEMM 的输入
    // 1.1. 复制输入数据，使其对每个专家都可用
    auto replicated_input = input.unsqueeze(0).expand({num_experts, batch_size, input_dim})
                                .contiguous().view({num_experts * batch_size, input_dim});

    // 1.2. 计算 total_rows_before_expert (前缀和)
    auto total_rows_before_expert_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto total_rows_before_expert_cpu = torch::zeros({num_experts + 1}, total_rows_before_expert_options);
    auto* total_rows_ptr = total_rows_before_expert_cpu.data_ptr<int64_t>();
    total_rows_ptr[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
        total_rows_ptr[i] = total_rows_ptr[i-1] + batch_size;
    }
    auto total_rows_before_expert = total_rows_before_expert_cpu.to(device);

    // 1.3. 获取GPU多处理器数量
    int multi_processor_count;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0);

    // 2. 第一次 Grouped GEMM (FC1)
    auto fc1_output = torch::zeros({num_experts * batch_size, hidden_dim}, 
                                      torch::TensorOptions().dtype(torch::kFloat).device(device));

    genericMoeGemmKernelLauncher<float, float, cutlass::arch::Sm80, EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<128, 128, 8>,
                                 cutlass::gemm::GemmShape<64, 64, 8>, 2>(
        replicated_input.data_ptr<float>(),
        expert_weights_fc1.data_ptr<float>(),
        nullptr, // weight_scales
        use_bias ? expert_biases_fc1.contiguous().view({-1}).data_ptr<float>() : nullptr,
        fc1_output.data_ptr<float>(),
        total_rows_before_expert.data_ptr<int64_t>(),
        num_experts * batch_size, // total_rows
        hidden_dim,               // gemm_n
        input_dim,                // gemm_k
        num_experts,
        multi_processor_count);

    // 3. 第二次 Grouped GEMM (FC2)
    auto fc2_output = torch::zeros({num_experts * batch_size, final_out_dim}, 
                                      torch::TensorOptions().dtype(torch::kFloat).device(device));
    
    genericMoeGemmKernelLauncher<float, float, cutlass::arch::Sm80, EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<128, 128, 8>,
                                 cutlass::gemm::GemmShape<64, 64, 8>, 2>(
        fc1_output.data_ptr<float>(),
        expert_weights_fc2.data_ptr<float>(),
        nullptr, // weight_scales
        use_bias ? expert_biases_fc2.contiguous().view({-1}).data_ptr<float>() : nullptr,
        fc2_output.data_ptr<float>(),
        total_rows_before_expert.data_ptr<int64_t>(),
        num_experts * batch_size, // total_rows
        final_out_dim,            // gemm_n
        hidden_dim,               // gemm_k
        num_experts,
        multi_processor_count);

    // 4. 将输出重新整形为期望的格式
    auto output = fc2_output.view({num_experts, batch_size, final_out_dim})
                                .transpose(0, 1).contiguous();

    // 5. 应用BatchNorm + SiLU融合计算 (这部分仍然是循环，但开销远小于GEMM)
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        float* expert_data_ptr = output.data_ptr<float>() + expert_idx * final_out_dim;
        
        dim3 block(256);
        dim3 grid(CEIL(batch_size * hidden_dim, block.x));
        
        singleExpertBatchNormSiluKernelStrided<float><<<grid, block>>>(
            expert_data_ptr,
            bn_weights[expert_idx].data_ptr<float>(),
            bn_biases[expert_idx].data_ptr<float>(),
            running_mean[expert_idx].data_ptr<float>(),
            running_var[expert_idx].data_ptr<float>(),
            batch_size,
            hidden_dim,
            num_experts * final_out_dim  // stride
        );
    }
    
    return output;
}

// ============================== Python接口 =================================

PYBIND11_MODULE(home_kernels, m) {
    m.def("home_expert_forward", torch::wrap_pybind_function(homeExpertForward), "HoME Expert Forward (All experts share same input)");
}
