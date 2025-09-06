#!/usr/bin/env python3
"""
调试BatchNorm的精度问题
"""

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import home_kernels
    from MLP import MLPLayer
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"⚠️  模块导入失败: {e}")
    exit(1)

def test_batchnorm_precision():
    """测试BatchNorm的精度匹配"""
    print("测试BatchNorm精度匹配...")
    
    # 简单测试参数
    batch_size = 2
    meta_output_dim = 6  # dim * 2
    expert_output_dim = 3  # dim
    num_experts = 2
    
    device = torch.device('cuda')
    
    # 创建简单的测试数据
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, meta_output_dim, device=device)
    expert_mlp_weights = torch.randn(num_experts, meta_output_dim, expert_output_dim, device=device)
    expert_biases = torch.randn(num_experts, expert_output_dim, device=device)
    
    print(f"输入数据:")
    print(f"  input_data: {input_data}")
    
    # 1. 先计算线性部分
    linear_outputs = []
    for expert_idx in range(num_experts):
        weight = expert_mlp_weights[expert_idx]
        bias = expert_biases[expert_idx]
        output = torch.mm(input_data, weight) + bias
        linear_outputs.append(output)
        print(f"  专家{expert_idx}线性输出: {output}")
    
    # 2. 创建并训练BatchNorm层，获取真实的running stats
    print(f"\n训练BatchNorm层获取真实统计信息:")
    bn_layers = []
    for expert_idx in range(num_experts):
        bn = nn.BatchNorm1d(expert_output_dim).to(device)
        bn.train()  # 训练模式，会更新running stats
        
        # 用线性输出训练BatchNorm
        with torch.no_grad():
            for _ in range(10):  # 多次训练更新统计信息
                _ = bn(linear_outputs[expert_idx])
        
        bn.eval()  # 切换到评估模式
        bn_layers.append(bn)
        
        print(f"  专家{expert_idx} BatchNorm统计:")
        print(f"    weight: {bn.weight.data}")
        print(f"    bias: {bn.bias.data}")
        print(f"    running_mean: {bn.running_mean.data}")
        print(f"    running_var: {bn.running_var.data}")
    
    # 3. 使用训练好的BatchNorm参数
    bn_weights = torch.stack([bn.weight.data for bn in bn_layers])
    bn_biases = torch.stack([bn.bias.data for bn in bn_layers])
    running_mean = torch.stack([bn.running_mean.data for bn in bn_layers])
    running_var = torch.stack([bn.running_var.data for bn in bn_layers])
    
    # 4. HoME.py风格的实现
    print(f"\nHoME.py风格实现:")
    home_outputs = []
    for expert_idx in range(num_experts):
        # MLPLayer
        expert_mlp = MLPLayer(meta_output_dim, [expert_output_dim], activate="relu")
        with torch.no_grad():
            expert_mlp.linears[0].linear.weight.data.copy_(expert_mlp_weights[expert_idx].t())
            expert_mlp.linears[0].linear.bias.data.copy_(expert_biases[expert_idx])
        expert_mlp = expert_mlp.to(device)
        expert_mlp.eval()
        
        # 完整的处理流程
        mlp_output = expert_mlp(input_data)
        bn_output = bn_layers[expert_idx](mlp_output)
        silu_output = torch.nn.functional.silu(bn_output)
        
        home_outputs.append(silu_output)
        print(f"  专家{expert_idx}:")
        print(f"    MLP输出: {mlp_output}")
        print(f"    BatchNorm输出: {bn_output}")
        print(f"    SiLU输出: {silu_output}")
    
    home_output = torch.stack(home_outputs, dim=1)  # [batch_size, num_experts, dim]
    print(f"  HoME最终输出: {home_output}")
    
    # 5. CUDA实现
    print(f"\nCUDA实现:")
    try:
        cuda_output = home_kernels.home_expert_forward(
            input_data, expert_mlp_weights, expert_biases,
            bn_weights, bn_biases, running_mean, running_var,
            num_experts, True, 1e-5
        )
        print(f"  CUDA输出: {cuda_output}")
        
        # 比较结果
        diff = torch.abs(home_output - cuda_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n精度对比:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print(f"  ✓ 精度验证通过")
        else:
            print(f"  ⚠️  精度差异较大")
            print(f"  差异详情: {diff}")
            
    except Exception as e:
        print(f"  CUDA实现失败: {e}")

if __name__ == "__main__":
    test_batchnorm_precision()
