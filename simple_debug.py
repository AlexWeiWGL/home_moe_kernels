#!/usr/bin/env python3
"""
简化的调试测试 - 验证CUDA kernel和HoME实现的计算差异
"""

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入HoME相关模块
try:
    from MLP import MLPLayer
    from Dense import DenseLayer
    HOME_AVAILABLE = True
    print("✓ HoME模块导入成功")
except ImportError as e:
    print(f"⚠️  HoME模块导入失败: {e}")
    HOME_AVAILABLE = False

# 尝试导入CUDA kernel模块
try:
    import home_kernels
    CUDA_KERNELS_AVAILABLE = True
    print("✓ CUDA kernels模块导入成功")
except ImportError as e:
    print(f"⚠️  CUDA kernels模块导入失败: {e}")
    CUDA_KERNELS_AVAILABLE = False
    home_kernels = None

def test_simple_linear():
    """测试简单的线性计算，不包含BatchNorm和SiLU"""
    print("\n测试简单线性计算...")
    
    # 测试参数
    batch_size = 2
    dim = 4
    meta_output_dim = dim * 2  # 8
    expert_output_dim = dim    # 4
    num_experts = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    input_data = torch.randn(batch_size, meta_output_dim, device=device)
    expert_mlp_weights = torch.randn(num_experts, meta_output_dim, expert_output_dim, device=device)
    expert_biases = torch.randn(num_experts, expert_output_dim, device=device)
    expert_indices = torch.arange(num_experts, dtype=torch.int32, device=device)
    
    print(f"输入数据:")
    print(f"  input_data: {input_data}")
    print(f"  expert_mlp_weights: {expert_mlp_weights}")
    print(f"  expert_biases: {expert_biases}")
    
    # HoME实现 - 只做线性计算
    print(f"\nHoME实现:")
    home_outputs = []
    for i, expert_idx in enumerate(expert_indices):
        expert_idx = expert_idx.item()
        expert_weight = expert_mlp_weights[expert_idx]  # [meta_output_dim, expert_output_dim]
        expert_bias = expert_biases[expert_idx]  # [expert_output_dim]
        
        # 线性计算: input @ weight + bias
        expert_output = torch.matmul(input_data, expert_weight) + expert_bias
        home_outputs.append(expert_output)
        print(f"  专家{expert_idx}: {expert_output}")
    
    home_output = torch.stack(home_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]
    print(f"  HoME最终输出: {home_output}")
    
    # CUDA实现
    if CUDA_KERNELS_AVAILABLE:
        print(f"\nCUDA实现:")
        try:
            # 创建BatchNorm参数（设置为恒等变换）
            bn_weights = torch.ones(num_experts, expert_output_dim, device=device)
            bn_biases = torch.zeros(num_experts, expert_output_dim, device=device)
            running_mean = torch.zeros(num_experts, expert_output_dim, device=device)
            running_var = torch.ones(num_experts, expert_output_dim, device=device)
            
            print(f"  BatchNorm参数:")
            print(f"    bn_weights: {bn_weights}")
            print(f"    bn_biases: {bn_biases}")
            print(f"    running_mean: {running_mean}")
            print(f"    running_var: {running_var}")
            
            with torch.no_grad():
                cuda_output = home_kernels.home_expert_forward(
                    input_data, expert_mlp_weights, expert_biases, expert_indices, 
                    bn_weights, bn_biases, running_mean, running_var,
                    num_experts, True, 1e-5
                )
            
            print(f"  CUDA输出: {cuda_output}")
            print(f"  CUDA输出形状: {cuda_output.shape}")
            print(f"  CUDA输出是否全零: {torch.all(cuda_output == 0)}")
            
            # 检查CUDA kernel是否真的被调用了
            if torch.all(cuda_output == 0):
                print(f"  ⚠️  CUDA kernel输出全零，可能存在问题")
                
                # 尝试不使用BatchNorm，直接测试GEMM
                print(f"\n  尝试不使用BatchNorm...")
                # 设置BatchNorm为恒等变换：weight=1, bias=0, mean=0, var=1
                # 这样 BatchNorm(x) = 1 * (x - 0) / sqrt(1 + eps) + 0 = x / sqrt(1 + eps) ≈ x
                
                # 让我们尝试不同的BatchNorm参数
                bn_weights_test = torch.ones(num_experts, expert_output_dim, device=device)
                bn_biases_test = torch.zeros(num_experts, expert_output_dim, device=device)
                running_mean_test = torch.zeros(num_experts, expert_output_dim, device=device)
                running_var_test = torch.ones(num_experts, expert_output_dim, device=device) * 100  # 增大方差
            
            # 比较结果
            diff = torch.abs(home_output - cuda_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"\n精度对比:")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"  ✓ 线性计算精度验证通过")
            else:
                print(f"  ⚠️  线性计算精度差异较大")
                print(f"  HoME输出: {home_output}")
                print(f"  CUDA输出: {cuda_output}")
                print(f"  差异: {diff}")
            
        except Exception as e:
            print(f"✗ CUDA实现失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  CUDA kernels不可用，跳过测试")

def main():
    """主函数"""
    print("简单线性计算调试测试")
    print("=" * 40)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 运行测试
    test_simple_linear()

if __name__ == "__main__":
    main()
