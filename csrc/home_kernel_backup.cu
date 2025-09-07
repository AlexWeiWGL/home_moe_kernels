/*
 * HoME (Hierarchical Mixture of Experts) CUDA Kernels
 * 针对HoME模型结构优化的CUDA实现
 * 
 * 包含以下优化：
 * 1. 专家网络CUTLASS Group GEMM优化
 * 2. LoRA门控CUDA内核优化
 * 3. l_gate和g_gate CUDA内核优化
 * 4. BatchNorm + SiLU融合算子
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
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"

#define WARP_SIZE 32
#define MAX_BYTES_PER_LDG 16
#define CEIL(a, b) ((a + b - 1) / (b))

// HoME特定常量
constexpr int MAX_EXPERTS_PER_GROUP = 10;
constexpr int MAX_TASK_GROUPS = 3;
constexpr int MAX_TASKS = 6;
constexpr int MAX_BATCH_SIZE = 4096;

// ============================== Epilogue操作定义 =================================

struct EpilogueOpDefaultSilu {};
struct EpilogueOpDefault {};
struct EpilogueOpReLU {};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op> 
struct Epilogue {};

constexpr auto DefaultScaleMode = cutlass::epilogue::thread::ScaleType::Default;

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultSilu>
{
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};

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

// ============================== 1. HoME专家网络CUTLASS Group GEMM优化 =================================

/**
 * HoME专家网络Group GEMM计算
 * 完全参考moe_kernels.cu中的genericMoeGemmKernelLauncher实现
 */
template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void homeExpertGemmKernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
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
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,  // Stages=5
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kGroupScheduleMode>;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
    // TLLM_CHECK_WITH_INFO(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
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

/**
 * 根据专家索引重排输入数据
 * 为Group GEMM准备数据
 */
template <typename T>
__global__ void reorderInputForExpertsKernel(
    const T* input,                 // [batch_size, input_dim]
    T* reordered_input,             // [num_selected_experts, batch_size, input_dim]
    const int* expert_indices,      // [num_selected_experts]
    int batch_size,
    int input_dim,
    int num_selected_experts
) {
    const int tid = threadIdx.x;
    const int expert_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (expert_idx >= num_selected_experts || batch_idx >= batch_size) return;

    const T* src = input + batch_idx * input_dim;
    T* dst = reordered_input + (expert_idx * batch_size + batch_idx) * input_dim;

    // 向量化拷贝
    for (int i = tid; i < input_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================== 2. LoRA门控CUDA内核优化 =================================

/**
 * 融合的LoRA门控计算内核
 * 实现: Fea_Gate(v) = 2 × Sigmoid(v(BA))
 * 其中BA = A @ B，避免中间结果存储
 */
template <typename T>
__global__ void loraGateFusedKernel(
    const T* input,           // [batch_size, input_dim]
    const T* A,               // [input_dim, rank]
    const T* B,               // [rank, output_dim]
    T* output,                // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int rank,
    int output_dim
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int output_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || output_idx >= output_dim) return;

    // 计算 v(BA) 的一个元素
    // 公式: output = 2 * sigmoid(v @ A @ B)
    // 其中 v 是输入向量，A 是 [input_dim, rank]，B 是 [rank, output_dim]
    T sum = 0.0f;
    for (int i = 0; i < input_dim; ++i) {
        T v_ba_element = 0.0f;
        for (int k = 0; k < rank; ++k) {
            v_ba_element += A[i * rank + k] * B[k * output_dim + output_idx];
        }
        sum += input[batch_idx * input_dim + i] * v_ba_element;
    }

    // 应用Sigmoid并乘以2
    T sigmoid_val = 1.0f / (1.0f + expf(-sum));
    output[batch_idx * output_dim + output_idx] = 2.0f * sigmoid_val;
}

/**
 * 向量化的LoRA门控计算内核
 * 使用float4进行向量化访存
 */
template <typename T>
__global__ void loraGateVectorizedKernel(
    const T* input,           // [batch_size, input_dim]
    const T* A,               // [input_dim, rank]
    const T* B,               // [rank, output_dim]
    T* output,                // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int rank,
    int output_dim
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * input_dim;
    T* batch_output = output + batch_idx * output_dim;

    // 向量化处理输出维度
    for (int j = tid; j < output_dim; j += blockDim.x) {
        T sum = 0.0f;
        
        // 计算 v(BA) 的第j个元素
        // 公式: output = 2 * sigmoid(v @ A @ B)
        for (int i = 0; i < input_dim; ++i) {
            T v_ba_element = 0.0f;
            // 向量化计算 A[i, :] @ B[:, j]
            for (int k = 0; k < rank; k += 4) {
                if (k + 3 < rank) {
                    // 使用float4向量化
                    float4 a_vec = reinterpret_cast<const float4*>(A + i * rank + k)[0];
                    float4 b_vec = reinterpret_cast<const float4*>(B + k * output_dim + j)[0];
                    v_ba_element += a_vec.x * b_vec.x + a_vec.y * b_vec.y + 
                                   a_vec.z * b_vec.z + a_vec.w * b_vec.w;
                } else {
                    // 处理剩余元素
                    for (int kk = k; kk < rank; ++kk) {
                        v_ba_element += A[i * rank + kk] * B[kk * output_dim + j];
                    }
                }
            }
            sum += batch_input[i] * v_ba_element;
        }
        
        // 应用Sigmoid并乘以2
        T sigmoid_val = 1.0f / (1.0f + expf(-sum));
        batch_output[j] = 2.0f * sigmoid_val;
    }
}

// ============================== 3. l_gate和g_gate CUDA内核优化 =================================

/**
 * 门控权重计算内核
 * 支持不同的激活函数（Softmax/Sigmoid）
 */
template <typename T>
__global__ void gateWeightsKernel(
    const T* gate_states,        // [batch_size, gate_dim]
    const T* gate_weights,       // [gate_dim, num_experts]
    T* output,                   // [batch_size, num_experts]
    int batch_size,
    int gate_dim,
    int num_experts,
    bool use_softmax
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;

    const T* batch_gate_states = gate_states + batch_idx * gate_dim;
    T* batch_output = output + batch_idx * num_experts;

    // 计算门控权重
    for (int j = tid; j < num_experts; j += blockDim.x) {
        T sum = 0.0f;
        for (int i = 0; i < gate_dim; ++i) {
            sum += batch_gate_states[i] * gate_weights[i * num_experts + j];
        }
        batch_output[j] = sum;
    }

    // 应用激活函数
    if (use_softmax) {
        // 使用共享内存进行Softmax计算
        __shared__ T shared_max;
        __shared__ T shared_sum;
        
        // 第一步：找到最大值
        T local_max = batch_output[0];
        for (int j = tid; j < num_experts; j += blockDim.x) {
            local_max = fmaxf(local_max, batch_output[j]);
        }
        
        // 使用warp reduce找到全局最大值
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        
        if (tid == 0) {
            shared_max = local_max;
        }
        __syncthreads();
        
        // 第二步：计算exp和sum
        T local_sum = 0.0f;
        for (int j = tid; j < num_experts; j += blockDim.x) {
            T exp_val = expf(batch_output[j] - shared_max);
            batch_output[j] = exp_val;
            local_sum += exp_val;
        }
        
        // 使用warp reduce计算全局sum
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        
        if (tid == 0) {
            shared_sum = local_sum;
        }
        __syncthreads();
        
        // 第三步：归一化
        for (int j = tid; j < num_experts; j += blockDim.x) {
            batch_output[j] /= shared_sum;
        }
    } else {
        // Sigmoid激活
        for (int j = tid; j < num_experts; j += blockDim.x) {
            batch_output[j] = 1.0f / (1.0f + expf(-batch_output[j]));
        }
    }
}

/**
 * 专家输出加权求和内核
 * 支持l_gate和g_gate的加权求和
 */
template <typename T>
__global__ void expertWeightedSumKernel(
    const T* expert_outputs,     // [batch_size, dim, num_experts]
    const T* gate_weights,       // [batch_size, num_experts]
    T* output,                   // [batch_size, dim]
    int batch_size,
    int dim,
    int num_experts
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || dim_idx >= dim) return;

    const T* batch_expert_outputs = expert_outputs + (batch_idx * dim + dim_idx) * num_experts;
    const T* batch_gate_weights = gate_weights + batch_idx * num_experts;
    T* batch_output = output + batch_idx * dim + dim_idx;

    // 加权求和
            T sum = 0.0f;
    for (int j = 0; j < num_experts; ++j) {
        sum += batch_expert_outputs[j] * batch_gate_weights[j];
    }
    batch_output[0] = sum;
}

// ============================== 4. BatchNorm + SiLU融合算子 =================================

/**
 * 融合的BatchNorm + SiLU内核
 * 避免中间结果存储，提高内存效率
 */
template <typename T>
__global__ void fusedBatchNormSiluKernel(
    T* data,                    // [batch_size, num_experts, hidden_dim]
    const T* bn_weights,        // [num_experts, hidden_dim]
    const T* bn_biases,         // [num_experts, hidden_dim]
    const T* running_mean,      // [num_experts, hidden_dim]
    const T* running_var,       // [num_experts, hidden_dim]
    T epsilon,                  // BatchNorm epsilon
    int batch_size,
    int num_experts,
    int hidden_dim
) {
    const int tid = threadIdx.x;
    const int expert_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    if (expert_id >= num_experts || batch_id >= batch_size) return;

    const T* expert_bn_weights = bn_weights + expert_id * hidden_dim;
    const T* expert_bn_biases = bn_biases + expert_id * hidden_dim;
    const T* expert_running_mean = running_mean + expert_id * hidden_dim;
    const T* expert_running_var = running_var + expert_id * hidden_dim;
    
    T* expert_data = data + (batch_id * num_experts + expert_id) * hidden_dim;

    // 向量化处理
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        // BatchNorm计算
        T normalized = (expert_data[i] - expert_running_mean[i]) / 
                      sqrt(expert_running_var[i] + epsilon);
        T bn_output = expert_bn_weights[i] * normalized + expert_bn_biases[i];
        
        // SiLU激活: x * sigmoid(x)
        T sigmoid_x = 1.0f / (1.0f + expf(-bn_output));
        expert_data[i] = bn_output * sigmoid_x;
    }
}

/**
 * 单个专家的BatchNorm + SiLU内核
 * 专门用于处理单个专家的输出
 */
template <typename T>
__global__ void singleExpertBatchNormSiluKernel(
    T* expert_data,             // [batch_size, hidden_dim] - 单个专家的输出
    const T* bn_weights,        // [hidden_dim] - 当前专家的BatchNorm权重
    const T* bn_biases,         // [hidden_dim] - 当前专家的BatchNorm偏置
    const T* running_mean,      // [hidden_dim] - 当前专家的运行均值
    const T* running_var,       // [hidden_dim] - 当前专家的运行方差
    T epsilon,                  // BatchNorm epsilon
    int batch_size,
    int hidden_dim
) {
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    
    if (batch_id >= batch_size) return;

    T* batch_data = expert_data + batch_id * hidden_dim;

    // 向量化处理
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        // BatchNorm计算
        T normalized = (batch_data[i] - running_mean[i]) / 
                      sqrt(running_var[i] + epsilon);
        T bn_output = bn_weights[i] * normalized + bn_biases[i];
        
        // SiLU激活: x * sigmoid(x)
        T sigmoid_x = 1.0f / (1.0f + expf(-bn_output));
        batch_data[i] = bn_output * sigmoid_x;
    }
}

template <typename T>
__global__ void singleExpertBatchNormSiluKernelStrided(
    T* expert_data,             // 专家数据的起始位置
    const T* bn_weights,        // [hidden_dim] - 当前专家的BatchNorm权重
    const T* bn_biases,         // [hidden_dim] - 当前专家的BatchNorm偏置
    const T* running_mean,      // [hidden_dim] - 当前专家的运行均值
    const T* running_var,       // [hidden_dim] - 当前专家的运行方差
    T epsilon,                  // BatchNorm epsilon
    int batch_size,
    int hidden_dim,
    int stride                  // 批次之间的步长
) {
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    
    if (batch_id >= batch_size) return;

    T* batch_data = expert_data + batch_id * stride;

    // 向量化处理
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        // BatchNorm计算
        T normalized = (batch_data[i] - running_mean[i]) / 
                      sqrt(running_var[i] + epsilon);
        T bn_output = bn_weights[i] * normalized + bn_biases[i];
        
        // SiLU激活: x * sigmoid(x)
        T sigmoid_x = 1.0f / (1.0f + expf(-bn_output));
        batch_data[i] = bn_output * sigmoid_x;
    }
}

// ============================== 单个CUTLASS GEMM测试 =================================

/**
 * 单个CUTLASS GEMM + ReLU测试函数
 * 用于调试和验证ReLU融合是否正确工作
 */
void testSingleCutlassRelu(
    torch::Tensor input,                    // [batch_size, input_dim]
    torch::Tensor weight,                   // [input_dim, hidden_dim]
    torch::Tensor bias,                     // [hidden_dim]
    torch::Tensor output,                   // 新增: 输出张量
    bool use_bias = true
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    input = input.to(device);
    weight = weight.to(device);
    bias = bias.to(device);

    // 获取参数
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int hidden_dim = weight.size(1);

    // 分配输出内存
    // auto output = torch::zeros({batch_size, hidden_dim}, 
    //                           torch::TensorOptions().dtype(torch::kFloat).device(device));

    // 使用CUTLASS GEMM + ReLU融合
    // 这是一个简化版本，用于调试ReLU融合
    
    // 获取GPU多处理器数量
    int multi_processor_count;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0);
    
    // 实现真正的CUTLASS GEMM + ReLU融合
    // printf("CUTLASS GEMM Debug: batch_size=%d, input_dim=%d, hidden_dim=%d\n", 
    //        batch_size, input_dim, hidden_dim);
    
    // 定义CUTLASS GEMM类型
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;  
    using LayoutC = cutlass::layout::RowMajor;
    
    // 定义ReLU Epilogue
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementC,                                       // ElementOutput
        1,                                             // ElementsPerVectorAccess  
        ElementAccumulator,                            // ElementAccumulator
        ElementAccumulator,                            // ElementCompute
        cutlass::epilogue::thread::ScaleType::NoBetaScaling  // ScaleType
    >;
    
    // 定义CUTLASS GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,                    // MMA Operation
        cutlass::arch::Sm80,                          // Architecture
        cutlass::gemm::GemmShape<128, 128, 8>,        // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 8>,          // WarpShape  
        cutlass::gemm::GemmShape<1, 1, 1>,            // InstructionShape
        EpilogueOp,                                   // EpilogueOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // ThreadblockSwizzle
        2                                             // Stages
    >;
    
    // 创建问题大小
    cutlass::gemm::GemmCoord problem_size(batch_size, hidden_dim, input_dim);
    
    // 设置Epilogue参数
    typename EpilogueOp::Params epilogue_params(
        ElementAccumulator(1.0f)  // alpha
    );
    
    // 创建GEMM参数
    typename Gemm::Arguments arguments{
        problem_size,                                  // problem_size
        {input.data_ptr<float>(), input_dim},         // ref_A
        {weight.data_ptr<float>(), hidden_dim},       // ref_B  
        {use_bias ? bias.data_ptr<float>() : nullptr, 0},  // ref_C (bias)
        {output.data_ptr<float>(), hidden_dim},       // ref_D
        {epilogue_params}                             // epilogue
    };
    
    // 创建GEMM实例
    Gemm gemm_op;
    
    // 检查是否可以实现
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM cannot be implemented: %d\n", int(status));
        // 回退到PyTorch实现
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(torch::relu(linear_output));
        return;
    }
    
    // 初始化GEMM
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM initialization failed: %d\n", int(status));
        // 回退到PyTorch实现
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(torch::relu(linear_output));
        return;
    }
    
    // 执行GEMM
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM execution failed: %d\n", int(status));
        // 回退到PyTorch实现
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(torch::relu(linear_output));
        return;
    }
    
    // printf("CUTLASS GEMM executed successfully!\n");

    return;
}

// 复制并修改 testSingleCutlassRelu 为 testSingleCutlassLinear
void testSingleCutlassLinear(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    bool use_bias = true
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    input = input.to(device);
    weight = weight.to(device);
    bias = bias.to(device);

    // 获取参数
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int hidden_dim = weight.size(1);

    // 分配输出内存
    // auto output = torch::zeros({batch_size, hidden_dim}, 
    //                           torch::TensorOptions().dtype(torch::kFloat).device(device));

    // 使用CUTLASS GEMM + ReLU融合
    // 这是一个简化版本，用于调试ReLU融合
    
    // 获取GPU多处理器数量
    int multi_processor_count;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0);
    
    // 实现真正的CUTLASS GEMM + ReLU融合
    // printf("CUTLASS GEMM Debug: batch_size=%d, input_dim=%d, hidden_dim=%d\n", 
    //        batch_size, input_dim, hidden_dim);
    
    // 定义CUTLASS GEMM类型
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;  
    using LayoutC = cutlass::layout::RowMajor;
    
    // 定义 Linear Epilogue (无激活)
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,
        ElementAccumulator,
        ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;
    
    // 定义CUTLASS GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,                    // MMA Operation
        cutlass::arch::Sm80,                          // Architecture
        cutlass::gemm::GemmShape<128, 128, 8>,        // ThreadblockShape
        cutlass::gemm::GemmShape<32, 64, 8>,          // WarpShape  
        cutlass::gemm::GemmShape<1, 1, 1>,            // InstructionShape
        EpilogueOp,                                   // EpilogueOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // ThreadblockSwizzle
        2                                             // Stages
    >;
    
    // 创建问题大小
    cutlass::gemm::GemmCoord problem_size(batch_size, hidden_dim, input_dim);
    
    // 设置Epilogue参数
    typename EpilogueOp::Params epilogue_params(
        ElementAccumulator(1.0f)  // alpha
    );
    
    // 创建GEMM参数
    typename Gemm::Arguments arguments{
        problem_size,                                  // problem_size
        {input.data_ptr<float>(), input_dim},         // ref_A
        {weight.data_ptr<float>(), hidden_dim},       // ref_B  
        {use_bias ? bias.data_ptr<float>() : nullptr, 0},  // ref_C (bias)
        {output.data_ptr<float>(), hidden_dim},       // ref_D
        {epilogue_params}                             // epilogue
    };
    
    // 创建GEMM实例
    Gemm gemm_op;
    
    // 检查是否可以实现
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM cannot be implemented: %d\n", int(status));
        // 回退到PyTorch实现 (无ReLU)
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(linear_output);
        return;
    }
    
    // 初始化GEMM
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM initialization failed: %d\n", int(status));
        // 回退到PyTorch实现 (无ReLU)
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(linear_output);
        return;
    }
    
    // 执行GEMM
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM execution failed: %d\n", int(status));
        // 回退到PyTorch实现 (无ReLU)
        auto linear_output = torch::mm(input, weight);
        if (use_bias) {
            linear_output = linear_output + bias.unsqueeze(0);
        }
        output.copy_(linear_output);
        return;
    }
    
    // printf("CUTLASS GEMM executed successfully!\n");

    return;
}

// ============================== Python接口 =================================

// 1. HoME专家网络接口（使用CUTLASS Grouped GEMM）
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
    
    // 确保所有张量在GPU上
    input = input.to(device);
    expert_weights_fc1 = expert_weights_fc1.to(device);
    expert_biases_fc1 = expert_biases_fc1.to(device);
    expert_weights_fc2 = expert_weights_fc2.to(device);
    expert_biases_fc2 = expert_biases_fc2.to(device);
    bn_weights = bn_weights.to(device);
    bn_biases = bn_biases.to(device);
    running_mean = running_mean.to(device);
    running_var = running_var.to(device);

    // 获取参数
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int hidden_dim = expert_weights_fc1.size(2);
    int final_out_dim = expert_weights_fc2.size(2); // 应该是 hidden_dim

    // 准备Grouped GEMM的输入：为每个专家复制相同的输入
    auto replicated_input = input.unsqueeze(0).expand({num_experts, batch_size, input_dim})
                                 .contiguous().view({num_experts * batch_size, input_dim});
    
    // 计算total_rows_before_expert：每个专家之前的累积行数
    auto total_rows_before_expert = torch::zeros({num_experts}, 
                                                torch::TensorOptions().dtype(torch::kInt64).device(device));
    for (int i = 0; i < num_experts; ++i) {
        total_rows_before_expert[i] = i * batch_size;
    }

    // 获取GPU多处理器数量
    int multi_processor_count;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0);

    // --- FC1 + ReLU (使用循环调用精确的单GEMM Kernel) ---
    auto fc1_output = torch::zeros({num_experts * batch_size, hidden_dim}, 
                                      torch::TensorOptions().dtype(torch::kFloat).device(device));

    for (int i = 0; i < num_experts; ++i) {
        auto expert_input = replicated_input.slice(0, i * batch_size, (i + 1) * batch_size);
        auto expert_weight_fc1 = expert_weights_fc1[i];
        auto expert_bias_fc1 = expert_biases_fc1[i];
        auto expert_output_fc1 = fc1_output.slice(0, i * batch_size, (i + 1) * batch_size);
        
        testSingleCutlassRelu(expert_input, expert_weight_fc1, expert_bias_fc1, expert_output_fc1, use_bias);
    }

    // --- FC2 (No Activation) ---
    auto fc2_output = torch::zeros({num_experts * batch_size, final_out_dim}, 
                                      torch::TensorOptions().dtype(torch::kFloat).device(device));
    
    for (int i = 0; i < num_experts; ++i) {
        auto expert_input_fc2 = fc1_output.slice(0, i * batch_size, (i + 1) * batch_size);
        auto expert_weight_fc2 = expert_weights_fc2[i];
        auto expert_bias_fc2 = expert_biases_fc2[i];
        auto expert_output_fc2 = fc2_output.slice(0, i * batch_size, (i + 1) * batch_size);
        
        // 注意：FC2没有ReLU，我们需要一个新的单GEMM Kernel或修改现有Kernel
        // 为快速验证，暂时使用PyTorch matmul
        testSingleCutlassLinear(expert_input_fc2, expert_weight_fc2, expert_bias_fc2, expert_output_fc2, use_bias);
    }

    // 将输出重新整形为期望的格式
    auto output = fc2_output.view({num_experts, batch_size, final_out_dim})
                                .transpose(0, 1).contiguous();

    // 应用BatchNorm + SiLU融合计算
    // 为每个专家应用BatchNorm + SiLU
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // 获取当前专家的BatchNorm参数
        auto expert_bn_weights = bn_weights[expert_idx];
        auto expert_bn_biases = bn_biases[expert_idx];
        auto expert_running_mean = running_mean[expert_idx];
        auto expert_running_var = running_var[expert_idx];
        
        // 获取当前专家的输出数据指针
        // 输出张量形状: [batch_size, num_experts, hidden_dim]
        // 专家expert_idx的数据在output[:, expert_idx, :]
        float* expert_data_ptr = output.data_ptr<float>() + expert_idx * final_out_dim;
        
        // 应用BatchNorm + SiLU到当前专家的输出
        dim3 bn_block_dim(256);
        dim3 bn_grid_dim(batch_size);  // 每个batch一个block
        
        singleExpertBatchNormSiluKernelStrided<float><<<bn_grid_dim, bn_block_dim>>>(
            expert_data_ptr,  // 专家expert_idx的起始位置
            expert_bn_weights.data_ptr<float>(),
            expert_bn_biases.data_ptr<float>(),
            expert_running_mean.data_ptr<float>(),
            expert_running_var.data_ptr<float>(),
            epsilon,
            batch_size,
            hidden_dim,
            num_experts * final_out_dim  // stride between batches
        );
    }

    // 确保所有CUDA操作完成
    cudaDeviceSynchronize();
    
    return output;
}

// 2. LoRA门控接口
torch::Tensor loraGateForward(
    torch::Tensor input,           // [batch_size, input_dim]
    torch::Tensor A,               // [input_dim, rank]
    torch::Tensor B,               // [rank, output_dim]
    bool use_vectorized
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    input = input.to(device);
    A = A.to(device);
    B = B.to(device);

    // 获取参数
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int rank = A.size(1);
    int output_dim = B.size(1);

    // 分配输出内存
    auto output = torch::zeros({batch_size, output_dim}, 
                              torch::TensorOptions().dtype(torch::kFloat).device(device));

    if (use_vectorized) {
        // 使用向量化内核实现
        dim3 block_dim(256);
        dim3 grid_dim(batch_size);
        loraGateVectorizedKernel<float><<<grid_dim, block_dim>>>(
            input.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, input_dim, rank, output_dim
        );
    } else {
        // 使用融合内核实现
        dim3 block_dim(256);
        dim3 grid_dim(batch_size, output_dim);
        loraGateFusedKernel<float><<<grid_dim, block_dim>>>(
            input.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, input_dim, rank, output_dim
        );
    }

    return output;
}

// 3. 门控权重接口
torch::Tensor gateWeightsForward(
    torch::Tensor gate_states,       // [batch_size, gate_dim]
    torch::Tensor gate_weights,      // [gate_dim, num_experts]
    bool use_softmax
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    gate_states = gate_states.to(device);
    gate_weights = gate_weights.to(device);

    // 获取参数
    int batch_size = gate_states.size(0);
    int gate_dim = gate_states.size(1);
    int num_experts = gate_weights.size(1);

    // 分配输出内存
    auto output = torch::zeros({batch_size, num_experts}, 
                                   torch::TensorOptions().dtype(torch::kFloat).device(device));
    
    // 执行门控权重计算
    dim3 block_dim(256);
    dim3 grid_dim(batch_size);
    gateWeightsKernel<float><<<grid_dim, block_dim>>>(
        gate_states.data_ptr<float>(),
        gate_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, gate_dim, num_experts, use_softmax
    );

    return output;
}

// 4. 专家输出加权求和接口
torch::Tensor expertWeightedSumForward(
    torch::Tensor expert_outputs,    // [batch_size, dim, num_experts]
    torch::Tensor gate_weights       // [batch_size, num_experts]
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    expert_outputs = expert_outputs.to(device);
    gate_weights = gate_weights.to(device);

    // 获取参数
    int batch_size = expert_outputs.size(0);
    int dim = expert_outputs.size(1);
    int num_experts = expert_outputs.size(2);

    // 分配输出内存
    auto output = torch::zeros({batch_size, dim}, 
                              torch::TensorOptions().dtype(torch::kFloat).device(device));

    // 执行专家输出加权求和
    dim3 block_dim(256);
    dim3 grid_dim(batch_size, dim);
    expertWeightedSumKernel<float><<<grid_dim, block_dim>>>(
        expert_outputs.data_ptr<float>(),
        gate_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim, num_experts
    );

    return output;
}

// 5. 融合算子接口
torch::Tensor fusedBatchNormSiluForward(
    torch::Tensor data,             // [batch_size, num_experts, hidden_dim]
    torch::Tensor bn_weights,       // [num_experts, hidden_dim]
    torch::Tensor bn_biases,        // [num_experts, hidden_dim]
    torch::Tensor running_mean,     // [num_experts, hidden_dim]
    torch::Tensor running_var,      // [num_experts, hidden_dim]
    float epsilon = 1e-5
) {
    torch::Device device(torch::kCUDA);
    
    // 确保所有张量在GPU上
    data = data.to(device);
    bn_weights = bn_weights.to(device);
    bn_biases = bn_biases.to(device);
    running_mean = running_mean.to(device);
    running_var = running_var.to(device);

    // 获取参数
    int batch_size = data.size(0);
    int num_experts = data.size(1);
    int hidden_dim = data.size(2);

    // 执行融合计算
    dim3 block_dim(256);
    dim3 grid_dim(num_experts, batch_size);
    fusedBatchNormSiluKernel<float><<<grid_dim, block_dim>>>(
        data.data_ptr<float>(),
        bn_weights.data_ptr<float>(),
        bn_biases.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        epsilon,
        batch_size, num_experts, hidden_dim
    );

    return data;
}

// ============================== 模块注册 =================================

PYBIND11_MODULE(home_kernels, m) {
    m.def("home_expert_forward", torch::wrap_pybind_function(homeExpertForward), "HoME Expert Forward (All experts share same input)");
    m.def("test_single_cutlass_relu", torch::wrap_pybind_function(testSingleCutlassRelu), "Test Single CUTLASS GEMM + ReLU");
    m.def("lora_gate_forward", torch::wrap_pybind_function(loraGateForward), "LoRA Gate Forward");
    m.def("gate_weights_forward", torch::wrap_pybind_function(gateWeightsForward), "Gate Weights Forward");
    m.def("expert_weighted_sum_forward", torch::wrap_pybind_function(expertWeightedSumForward), "Expert Weighted Sum Forward");
    m.def("fused_batch_norm_silu_forward", torch::wrap_pybind_function(fusedBatchNormSiluForward), "Fused BatchNorm SiLU Forward");
}
