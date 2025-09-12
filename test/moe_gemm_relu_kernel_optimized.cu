/*
 * MoE GEMM ReLU 优化内核 - 共享输入版本
 * 避免输入数据重复复制，实现真正的内存和带宽优化
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
 
 // 将偏置向量按行广播到矩阵的 kernel：C[row, col] = bias[col]
template <typename T>
__global__ void expand_bias_rows_kernel(const T* __restrict__ bias, T* __restrict__ C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[row * N + col] = bias[col];
    }
}

// 行主 [K, N] -> 列主 [K, N] 的权重转换 kernel（按 expert 维 z 轴并行）
template <typename T>
__global__ void convert_row_to_col_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                          int K, int N, int expert_stride) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int e = blockIdx.z;
    if (k < K && n < N) {
        const T* src_e = src + e * expert_stride;
        T* dst_e = dst + e * expert_stride;
        // 行主: src[k*N + n]  -> 列主: dst[n*K + k]
        dst_e[n * K + k] = src_e[k * N + n];
    }
}
 
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
 
 // ============================== 共享输入MoE GEMM Kernel =================================
 
 template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
     typename WarpShape, int Stages>
 void moeGemmKernelLauncherSharedInputImpl(const T* shared_input, const WeightType* weights, const T* biases, T* output,
     const int64_t* expert_offsets, int64_t batch_size, int64_t gemm_n, int64_t gemm_k, int num_experts,
     const int multi_processor_count)
 {
     // The cutlass type for the input elements
     using ElementType = typename cutlass::platform::conditional<
         cutlass::platform::is_same<T, cutlass::half_t>::value, cutlass::half_t, T
     >::type;
 
     using CutlassWeightType = typename cutlass::platform::conditional<
         cutlass::platform::is_same<WeightType, cutlass::half_t>::value, cutlass::half_t, WeightType
     >::type;
 
     using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
     using ElementAccumulator = typename MixedGemmArchTraits::AccType;
 
     using EpilogueOp = typename Epilogue<ElementType,
         MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;
 
     // 使用标准的Grouped GEMM而不是MoeFCGemm，因为我们要手动处理共享输入
     using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor,
         cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
         cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
         MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
         typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
         typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
         cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
         cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;
 
     using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel_>;
 
     // 先用硬件最大占用估算一波线程块，再与 tile 数取最小
     int max_active = std::max(1, GemmGrouped::maximum_active_blocks());
     int wave_blocks = multi_processor_count * max_active;
 
     typename EpilogueOp::Params epilogue_op(
         ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));
 
     // 准备Grouped GEMM的参数
     // 每个expert处理相同的输入，但使用不同的权重
     std::vector<cutlass::gemm::GemmCoord> problem_sizes;
     std::vector<const ElementType*> ptr_A;
     std::vector<const CutlassWeightType*> ptr_B;
     std::vector<ElementType*> ptr_C;
     std::vector<ElementType*> ptr_D;
     std::vector<int64_t> lda, ldb, ldc, ldd;
     std::vector<ElementAccumulator> alpha, beta;
 
     problem_sizes.reserve(num_experts);
     ptr_A.reserve(num_experts);
     ptr_B.reserve(num_experts);
     ptr_C.reserve(num_experts);
     ptr_D.reserve(num_experts);
     lda.reserve(num_experts);
     ldb.reserve(num_experts);
     ldc.reserve(num_experts);
     ldd.reserve(num_experts);
     alpha.reserve(num_experts);
     beta.reserve(num_experts);
 
     // 如果使用 bias，则为每个 expert 扩展一份按行广播的 bias 矩阵（带缓存）
    static std::vector<ElementType*> expanded_bias_C_cache;
    static int cached_num_experts = 0;
    static int64_t cached_batch_size = 0;
    static int64_t cached_gemm_n = 0;
    bool need_recreate_bias = false;
    if (biases) {
        if (cached_num_experts != num_experts || cached_batch_size != batch_size || cached_gemm_n != gemm_n) {
            // 释放旧缓存
            for (auto ptr : expanded_bias_C_cache) { if (ptr) cudaFree(ptr); }
            expanded_bias_C_cache.clear();
            expanded_bias_C_cache.resize(num_experts, nullptr);
            cached_num_experts = num_experts;
            cached_batch_size = batch_size;
            cached_gemm_n = gemm_n;
            need_recreate_bias = true;
        }
        if (need_recreate_bias && num_experts > 0) {
            dim3 block(32, 8);
            dim3 grid((gemm_n + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
            for (int e = 0; e < num_experts; ++e) {
                CHECK_CUDA(cudaMalloc(&expanded_bias_C_cache[e], sizeof(ElementType) * batch_size * gemm_n));
                const ElementType* bias_e = reinterpret_cast<const ElementType*>(biases) + e * gemm_n;
                expand_bias_rows_kernel<ElementType><<<grid, block>>>(bias_e, expanded_bias_C_cache[e], (int)batch_size, (int)gemm_n);
            }
            // 不进行设备同步，后续 GEMM 会在同一流上消费这些数据
        }
    }

    // (4) 暂不使用权重列主缓存，直接使用原始权重指针
 
     for (int e = 0; e < num_experts; ++e) {
         // 每个expert处理相同的输入
         problem_sizes.push_back(cutlass::gemm::GemmCoord(batch_size, gemm_n, gemm_k));
         
         // 所有expert使用相同的输入指针
         ptr_A.push_back(reinterpret_cast<const ElementType*>(shared_input));
         // 使用原始权重指针
         ptr_B.push_back(reinterpret_cast<const CutlassWeightType*>(weights) + e * gemm_k * gemm_n);
         // 当提供 biases 时，使用已扩展的按行广播矩阵；否则不传 C
         ptr_C.push_back(biases ? expanded_bias_C_cache[e] : nullptr);
         ptr_D.push_back(reinterpret_cast<ElementType*>(output) + e * batch_size * gemm_n);
         
         // 设置leading dimensions
         lda.push_back(gemm_k);
         ldb.push_back(gemm_k);
         // 使用行主步长
         ldc.push_back(gemm_n);
         ldd.push_back(gemm_n);
         
         // 设置alpha和beta
         alpha.push_back(ElementAccumulator(1.0));
         beta.push_back(ElementAccumulator(biases ? 1.0f : 0.0f));
     }
 
     // 为非const正确的CUTLASS API创建非const指针数组
     std::vector<ElementType*> ptr_A_nonconst;
     ptr_A_nonconst.reserve(num_experts);
     for (const auto& p : ptr_A) {
         ptr_A_nonconst.push_back(const_cast<ElementType*>(p));
     }
 
     std::vector<CutlassWeightType*> ptr_B_nonconst;
     ptr_B_nonconst.reserve(num_experts);
     for (const auto& p : ptr_B) {
         ptr_B_nonconst.push_back(const_cast<CutlassWeightType*>(p));
     }
 
     // (3) 设备参数数组缓存
     static cutlass::gemm::GemmCoord* d_problem_sizes = nullptr;
     static ElementType** d_ptr_A = nullptr;
     static CutlassWeightType** d_ptr_B = nullptr;
     static ElementType** d_ptr_C = nullptr;
     static ElementType** d_ptr_D = nullptr;
     static typename cutlass::layout::RowMajor::Stride::LongIndex* d_lda = nullptr;
     static typename cutlass::layout::RowMajor::Stride::LongIndex* d_ldb = nullptr;
     static typename cutlass::layout::RowMajor::Stride::LongIndex* d_ldc = nullptr;
     static typename cutlass::layout::RowMajor::Stride::LongIndex* d_ldd = nullptr;
     static int cached_params_experts = 0;

     if (cached_params_experts != num_experts) {
         if (d_problem_sizes) { cudaFree(d_problem_sizes); }
         if (d_ptr_A) { cudaFree(d_ptr_A); }
         if (d_ptr_B) { cudaFree(d_ptr_B); }
         if (d_ptr_C) { cudaFree(d_ptr_C); }
         if (d_ptr_D) { cudaFree(d_ptr_D); }
         if (d_lda) { cudaFree(d_lda); }
         if (d_ldb) { cudaFree(d_ldb); }
         if (d_ldc) { cudaFree(d_ldc); }
         if (d_ldd) { cudaFree(d_ldd); }
         CHECK_CUDA(cudaMalloc(&d_problem_sizes, sizeof(cutlass::gemm::GemmCoord) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ptr_A, sizeof(ElementType*) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ptr_B, sizeof(CutlassWeightType*) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ptr_C, sizeof(ElementType*) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ptr_D, sizeof(ElementType*) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_lda, sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ldb, sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ldc, sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts));
         CHECK_CUDA(cudaMalloc(&d_ldd, sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts));
         cached_params_experts = num_experts;
     }
 
     // 拷贝到 device
     CHECK_CUDA(cudaMemcpy(d_problem_sizes, problem_sizes.data(), sizeof(cutlass::gemm::GemmCoord) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ptr_A, ptr_A_nonconst.data(), sizeof(ElementType*) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ptr_B, ptr_B_nonconst.data(), sizeof(CutlassWeightType*) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ptr_C, ptr_C.data(), sizeof(ElementType*) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ptr_D, ptr_D.data(), sizeof(ElementType*) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_lda, lda.data(), sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ldb, ldb.data(), sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ldc, ldc.data(), sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts, cudaMemcpyHostToDevice));
     CHECK_CUDA(cudaMemcpy(d_ldd, ldd.data(), sizeof(typename cutlass::layout::RowMajor::Stride::LongIndex) * num_experts, cudaMemcpyHostToDevice));
 
     // 先用 wave_blocks 构造临时 args，计算总 tiles
     typename GemmGrouped::Arguments args_tmp(
         d_problem_sizes,                // problem_sizes (device)
         num_experts,                    // problem_count
         wave_blocks,                    // provisional threadblock_count
         epilogue_op,                    // output_op
         d_ptr_A,                        // ptr_A (device)
         d_ptr_B,                        // ptr_B (device)
         d_ptr_C,                        // ptr_C (device)
         d_ptr_D,                        // ptr_D (device)
         d_lda,                          // lda (device)
         d_ldb,                          // ldb (device)
         d_ldc,                          // ldc (device)
         d_ldd,                          // ldd (device)
         problem_sizes.data()            // host_problem_sizes (host)
     );

     int tiles = GemmGrouped::group_tile_count(args_tmp);
     int threadblock_count = (tiles > 0) ? std::min(tiles, wave_blocks) : wave_blocks;

     typename GemmGrouped::Arguments args(
         d_problem_sizes,                // problem_sizes (device)
         num_experts,                    // problem_count
         threadblock_count,              // final threadblock_count
         epilogue_op,                    // output_op
         d_ptr_A,                        // ptr_A (device)
         d_ptr_B,                        // ptr_B (device)
         d_ptr_C,                        // ptr_C (device)
         d_ptr_D,                        // ptr_D (device)
         d_lda,                          // lda (device)
         d_ldb,                          // ldb (device)
         d_ldc,                          // ldc (device)
         d_ldd,                          // ldd (device)
         problem_sizes.data()            // host_problem_sizes (host)
     );
 
     GemmGrouped gemm;
 
     std::cout << "Creating optimized GEMM with " << num_experts << " experts, " 
               << batch_size << " batch size, " << gemm_n << "x" << gemm_k << std::endl;
     std::cout << "Shared input optimization: NO input replication!" << std::endl;
 
     auto can_implement = gemm.can_implement(args);
     if (can_implement != cutlass::Status::kSuccess) {
         std::cerr << "Optimized MoE kernel will fail for params. Status: " << int(can_implement) << std::endl;
         return;
     } else {
         std::cout << "Kernel can_implement: SUCCESS" << std::endl;
     }
 
     auto init_status = gemm.initialize(args);
     if (init_status != cutlass::Status::kSuccess) {
         std::cerr << "Failed to initialize optimized cutlass grouped gemm. Status: " << int(init_status) << std::endl;
         return;
     } else {
         std::cout << "Kernel initialize: SUCCESS" << std::endl;
     }
     
     std::cout << "Running optimized GEMM kernel..." << std::endl;
     auto run_status = gemm.run();
     if (run_status != cutlass::Status::kSuccess) {
         std::cerr << "Failed to run optimized cutlass grouped gemm. Status: " << int(run_status) << std::endl;
         // 参数数组采用缓存，不释放
         // 不释放 bias 缓存，复用以减少后续调用开销
         return;
     } else {
         std::cout << "Kernel run: SUCCESS" << std::endl;
     }
 
     // 参数数组采用缓存，不释放
     // 保留 bias 缓存供后续调用复用
 }
 
 // ============================== 优化的C接口实现 =================================
 
 extern "C" {
 
 void launch_moe_gemm_relu_kernel_optimized(
     const float* shared_input,      // [batch_size, input_dim] - 共享输入，不重复
     const float* weights,           // [num_experts, input_dim, output_dim]
     const float* biases,            // [num_experts, output_dim] or nullptr
     float* output,                  // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     // 获取GPU多处理器数量
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     // 创建expert_offsets数组（虽然在这个优化版本中不使用，但保持接口兼容性）
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     // 启动优化的MoE GEMM ReLU kernel
    moeGemmKernelLauncherSharedInputImpl<float, float, cutlass::arch::Sm89, EpilogueOpReLU,
                                        cutlass::gemm::GemmShape<64, 64, 8>,
                                        cutlass::gemm::GemmShape<32, 32, 8>, 2>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 void launch_moe_gemm_kernel_optimized(
     const float* shared_input,      // [batch_size, input_dim] - 共享输入，不重复
     const float* weights,           // [num_experts, input_dim, output_dim]
     const float* biases,            // [num_experts, output_dim] or nullptr
     float* output,                  // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     // 获取GPU多处理器数量
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     // 创建expert_offsets数组
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     // 启动优化的MoE GEMM kernel
    moeGemmKernelLauncherSharedInputImpl<float, float, cutlass::arch::Sm89, EpilogueOpDefault,
                                        cutlass::gemm::GemmShape<64, 64, 8>,
                                        cutlass::gemm::GemmShape<32, 32, 8>, 2>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 // FP16版本的优化MoE GEMM kernel
 void launch_moe_gemm_relu_kernel_optimized_fp16(
     const cutlass::half_t* shared_input,  // [batch_size, input_dim] - 共享输入，不重复
     const cutlass::half_t* weights,       // [num_experts, input_dim, output_dim]
     const cutlass::half_t* biases,        // [num_experts, output_dim] or nullptr
     cutlass::half_t* output,              // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     moeGemmKernelLauncherSharedInputImpl<cutlass::half_t, cutlass::half_t, cutlass::arch::Sm80, EpilogueOpReLU,
                                         cutlass::gemm::GemmShape<256, 128, 32>,
                                         cutlass::gemm::GemmShape<64, 64, 32>, 4>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 void launch_moe_gemm_kernel_optimized_fp16(
     const cutlass::half_t* shared_input,  // [batch_size, input_dim] - 共享输入，不重复
     const cutlass::half_t* weights,       // [num_experts, input_dim, output_dim]
     const cutlass::half_t* biases,        // [num_experts, output_dim] or nullptr
     cutlass::half_t* output,              // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     moeGemmKernelLauncherSharedInputImpl<cutlass::half_t, cutlass::half_t, cutlass::arch::Sm80, EpilogueOpDefault,
                                         cutlass::gemm::GemmShape<256, 128, 32>,
                                         cutlass::gemm::GemmShape<64, 64, 32>, 4>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 // BF16版本的优化MoE GEMM kernel
 void launch_moe_gemm_relu_kernel_optimized_bf16(
     const cutlass::bfloat16_t* shared_input,  // [batch_size, input_dim] - 共享输入，不重复
     const cutlass::bfloat16_t* weights,       // [num_experts, input_dim, output_dim]
     const cutlass::bfloat16_t* biases,        // [num_experts, output_dim] or nullptr
     cutlass::bfloat16_t* output,              // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     moeGemmKernelLauncherSharedInputImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpReLU,
                                         cutlass::gemm::GemmShape<256, 128, 32>,
                                         cutlass::gemm::GemmShape<64, 64, 32>, 4>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 void launch_moe_gemm_kernel_optimized_bf16(
     const cutlass::bfloat16_t* shared_input,  // [batch_size, input_dim] - 共享输入，不重复
     const cutlass::bfloat16_t* weights,       // [num_experts, input_dim, output_dim]
     const cutlass::bfloat16_t* biases,        // [num_experts, output_dim] or nullptr
     cutlass::bfloat16_t* output,              // [num_experts, batch_size, output_dim]
     int num_experts,
     int batch_size,
     int input_dim,
     int output_dim,
     bool use_bias
 ) {
     int multi_processor_count;
     CHECK_CUDA(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0));
 
     std::vector<int64_t> expert_offsets(num_experts);
     for (int i = 0; i < num_experts; ++i) {
         expert_offsets[i] = (i + 1) * batch_size;
     }
 
     moeGemmKernelLauncherSharedInputImpl<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm80, EpilogueOpDefault,
                                         cutlass::gemm::GemmShape<256, 128, 32>,
                                         cutlass::gemm::GemmShape<64, 64, 32>, 4>(
         shared_input,
         weights,
         use_bias ? biases : nullptr,
         output,
         expert_offsets.data(),
         batch_size,
         output_dim,
         input_dim,
         num_experts,
         multi_processor_count
     );
 
     CHECK_CUDA(cudaGetLastError());
 }
 
 } // extern "C"
 