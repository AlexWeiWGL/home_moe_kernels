#!/usr/bin/env python3
"""
调试CUTLASS Grouped GEMM输出布局
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import home_kernels
    print("✓ CUDA kernels模块导入成功")
except ImportError as e:
    print(f"⚠️  CUDA kernels模块导入失败: {e}")
    exit(1)

def test_cutlass_layout():
    """测试CUTLASS输出布局"""
    print("测试CUTLASS Grouped GEMM输出布局...")
    
    # 简单的测试参数
    batch_size = 2
    input_dim = 3
    hidden_dim = 2
    num_experts = 2
    
    device = torch.device('cuda')
    
    # 创建简单的测试数据
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, input_dim, device=device)
    expert_weights = torch.randn(num_experts, input_dim, hidden_dim, device=device)
    expert_biases = torch.randn(num_experts, hidden_dim, device=device)
    
    print(f"输入数据:")
    print(f"  input: {input_data}")
    print(f"  expert_weights: {expert_weights}")
    print(f"  expert_biases: {expert_biases}")
    
    # PyTorch参考实现
    print(f"\nPyTorch参考实现:")
    ref_outputs = []
    for expert_idx in range(num_experts):
        expert_output = torch.mm(input_data, expert_weights[expert_idx]) + expert_biases[expert_idx]
        ref_outputs.append(expert_output)
        print(f"  专家{expert_idx}: {expert_output}")
    
    ref_output = torch.stack(ref_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
    print(f"  参考输出: {ref_output}")
    print(f"  参考输出形状: {ref_output.shape}")
    
    # 创建恒等BatchNorm参数（不改变数值）
    bn_weights = torch.ones(num_experts, hidden_dim, device=device)
    bn_biases = torch.zeros(num_experts, hidden_dim, device=device)
    running_mean = torch.zeros(num_experts, hidden_dim, device=device)
    running_var = torch.ones(num_experts, hidden_dim, device=device)
    
    # CUDA实现
    print(f"\nCUDA实现:")
    try:
        cuda_output = home_kernels.home_expert_forward(
            input_data, expert_weights, expert_biases,
            bn_weights, bn_biases, running_mean, running_var,
            num_experts, True, 1e-5
        )
        
        print(f"  CUDA输出: {cuda_output}")
        print(f"  CUDA输出形状: {cuda_output.shape}")
        
        # 比较结果
        diff = torch.abs(ref_output - cuda_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n精度对比:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"  ✓ 精度验证通过")
        else:
            print(f"  ⚠️  精度差异较大")
            print(f"  差异详情: {diff}")
            
    except Exception as e:
        print(f"✗ CUDA实现失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cutlass_layout()
