#!/usr/bin/env python3
"""
测试不使用SiLU激活的情况 - 只做线性计算和BatchNorm
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

def test_without_silu():
    """测试不使用SiLU的情况"""
    print("\n测试不使用SiLU激活...")
    
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
    print(f"  expert_mlp_weights[0]: {expert_mlp_weights[0]}")
    print(f"  expert_biases[0]: {expert_biases[0]}")
    
    # HoME实现 - 只做线性计算，不使用SiLU
    print(f"\nHoME实现（不使用SiLU）:")
    home_outputs = []
    for i, expert_idx in enumerate(expert_indices):
        expert_idx = expert_idx.item()
        expert_weight = expert_mlp_weights[expert_idx]  # [meta_output_dim, expert_output_dim]
        expert_bias = expert_biases[expert_idx]  # [expert_output_dim]
        
        # 线性计算: input @ weight + bias
        expert_output = torch.matmul(input_data, expert_weight) + expert_bias
        
        # BatchNorm恒等变换：(x - 0) / sqrt(1 + 1e-5) * 1 + 0 ≈ x
        expert_output = expert_output / torch.sqrt(torch.tensor(1.0 + 1e-5))
        
        home_outputs.append(expert_output)
        print(f"  专家{expert_idx} 线性输出: {expert_output}")
    
    home_output = torch.stack(home_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]
    print(f"  HoME最终输出: {home_output}")
    
    # 手动验证第一个专家的计算
    print(f"\n手动验证专家0的计算:")
    manual_output = torch.matmul(input_data, expert_mlp_weights[0]) + expert_biases[0]
    print(f"  手动计算结果: {manual_output}")
    print(f"  与HoME输出差异: {torch.abs(manual_output - home_outputs[0]).max().item()}")
    
    # CUDA实现
    if CUDA_KERNELS_AVAILABLE:
        print(f"\nCUDA实现:")
        try:
            # 创建BatchNorm参数（恒等变换，但不使用SiLU）
            bn_weights = torch.ones(num_experts, expert_output_dim, device=device)
            bn_biases = torch.zeros(num_experts, expert_output_dim, device=device)
            running_mean = torch.zeros(num_experts, expert_output_dim, device=device)
            running_var = torch.ones(num_experts, expert_output_dim, device=device)
            
            with torch.no_grad():
                cuda_output = home_kernels.home_expert_forward(
                    input_data, expert_mlp_weights, expert_biases, expert_indices, 
                    bn_weights, bn_biases, running_mean, running_var,
                    num_experts, True, 1e-5
                )
            
            print(f"  CUDA输出: {cuda_output}")
            print(f"  CUDA输出形状: {cuda_output.shape}")
            
            # 分析CUDA输出
            print(f"\n分析CUDA输出:")
            for i in range(num_experts):
                for b in range(batch_size):
                    cuda_expert_batch = cuda_output[b, i, :]
                    home_expert_batch = home_output[b, i, :]
                    
                    print(f"  专家{i}, 批次{b}:")
                    print(f"    CUDA: {cuda_expert_batch}")
                    print(f"    HoME: {home_expert_batch}")
                    print(f"    差异: {torch.abs(cuda_expert_batch - home_expert_batch)}")
            
            # 比较结果
            diff = torch.abs(home_output - cuda_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"\n精度对比:")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            
        except Exception as e:
            print(f"✗ CUDA实现失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  CUDA kernels不可用，跳过测试")

def main():
    """主函数"""
    print("测试不使用SiLU激活的情况")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 运行测试
    test_without_silu()

if __name__ == "__main__":
    main()
