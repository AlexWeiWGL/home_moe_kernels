#!/usr/bin/env python3
"""
准确的专家前向传播对比测试

完全复刻HoME.py中的专家计算逻辑，对比：
1. CUDA Kernel实现 (home_kernels.home_expert_forward)
2. HoME.py中的实际for循环实现 (Sequential(MLPLayer, BatchNorm, SiLU))

HoME模型结构理解：
- 专家输入维度: meta_output_dim = dim * 2
- 专家输出维度: dim
- 专家网络: MLPLayer(meta_output_dim, [dim], activate="relu")
- 完整结构: Sequential(MLPLayer, BatchNorm1d, SiLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
from typing import List

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入CUDA kernel模块
try:
    import home_kernels
    CUDA_KERNELS_AVAILABLE = True
    print("✓ CUDA kernels模块导入成功")
except ImportError as e:
    print(f"⚠️  CUDA kernels模块导入失败: {e}")
    CUDA_KERNELS_AVAILABLE = False
    home_kernels = None

# 导入HoME相关模块
try:
    from MLP import MLPLayer
    from Dense import DenseLayer
    HOME_AVAILABLE = True
    print("✓ HoME模块导入成功")
except ImportError as e:
    print(f"⚠️  HoME模块导入失败: {e}")
    HOME_AVAILABLE = False


def create_home_expert_sequential(input_data, expert_mlp_weights, expert_biases, expert_indices, 
                                meta_output_dim, dim, num_experts, use_bias=True):
    """
    完全复刻HoME.py中的专家计算逻辑
    使用Sequential(MLPLayer, BatchNorm, SiLU)结构
    
    Args:
        input_data: [batch_size, meta_output_dim] - 专家输入（来自Meta层）
        expert_mlp_weights: [num_experts, meta_output_dim, dim] - MLP层权重
        expert_biases: [num_experts, dim] - MLP层偏置
        expert_indices: [num_selected_experts] - 选中的专家索引
        meta_output_dim: Meta层输出维度 = dim * 2
        dim: 专家输出维度
        num_experts: 专家总数
    """
    batch_size = input_data.size(0)
    num_selected_experts = expert_indices.size(0)
    
    # 分配输出内存
    output = torch.zeros(batch_size, num_selected_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    # 完全复刻HoME.py中的专家计算逻辑
    for i, expert_idx in enumerate(expert_indices):
        expert_idx = expert_idx.item()
        if expert_idx < num_experts:
            # 创建专家网络，完全按照HoME.py中的结构
            # 1. MLPLayer(meta_output_dim, [dim], activate="relu")
            expert_mlp = MLPLayer(meta_output_dim, [dim], activate="relu", name=f"expert_{expert_idx}")
            
            # 2. BatchNorm1d
            batch_norm = nn.BatchNorm1d(dim)
            
            # 3. SiLU激活函数
            silu_activation = nn.SiLU()
            
            # 设置MLP层权重和偏置
            with torch.no_grad():
                # expert_mlp_weights[expert_idx] 的形状是 [meta_output_dim, dim]
                # MLPLayer内部只有一个Linear层，权重形状是 [dim, meta_output_dim]
                expert_mlp.linears[0].linear.weight.data.copy_(expert_mlp_weights[expert_idx].t())
                if use_bias and expert_biases is not None:
                    expert_mlp.linears[0].linear.bias.data.copy_(expert_biases[expert_idx])
            
            # 移动到正确的设备
            expert_mlp = expert_mlp.to(input_data.device)
            batch_norm = batch_norm.to(input_data.device)
            
            # 设置为评估模式，确保使用running_mean和running_var
            batch_norm.eval()
            
            # 按照HoME.py中的顺序执行：MLP -> BatchNorm -> SiLU
            expert_output = expert_mlp(input_data)  # MLPLayer
            expert_output = batch_norm(expert_output)  # BatchNorm (使用running stats)
            expert_output = silu_activation(expert_output)  # SiLU激活
            
            # 存储到输出张量中
            output[:, i, :] = expert_output
    
    return output


def create_home_expert_sequential_with_bn(input_data, 
                                        expert_mlp_weights_fc1, expert_biases_fc1,
                                        expert_mlp_weights_fc2, expert_biases_fc2,
                                        bn_weights, bn_biases, running_mean, running_var, 
                                        meta_output_dim, dim, num_experts, use_bias=True, epsilon=1e-5):
    """
    HoME架构的专家计算逻辑，使用指定的BatchNorm参数
    使用两层网络结构: Linear+ReLU -> Linear -> BatchNorm -> SiLU
    """
    batch_size = input_data.size(0)
    hidden_dim = expert_mlp_weights_fc1.shape[2] # FC1的输出维度
    
    output = torch.zeros(batch_size, num_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    for expert_idx in range(num_experts):
        fc1 = DenseLayer(meta_output_dim, hidden_dim, activation="relu", name=f"expert_fc1_{expert_idx}")
        fc2 = DenseLayer(hidden_dim, dim, activation=None, name=f"expert_fc2_{expert_idx}")
        batch_norm = nn.BatchNorm1d(dim)
        silu_activation = nn.SiLU()
        
        with torch.no_grad():
            fc1.linear.weight.data.copy_(expert_mlp_weights_fc1[expert_idx].t())
            if use_bias: fc1.linear.bias.data.copy_(expert_biases_fc1[expert_idx])
            fc2.linear.weight.data.copy_(expert_mlp_weights_fc2[expert_idx].t())
            if use_bias: fc2.linear.bias.data.copy_(expert_biases_fc2[expert_idx])
            batch_norm.weight.data.copy_(bn_weights[expert_idx])
            batch_norm.bias.data.copy_(bn_biases[expert_idx])
            batch_norm.running_mean.data.copy_(running_mean[expert_idx])
            batch_norm.running_var.data.copy_(running_var[expert_idx])
        
        fc1 = fc1.to(input_data.device)
        fc2 = fc2.to(input_data.device)
        batch_norm = batch_norm.to(input_data.device)
        batch_norm.eval()
        
        expert_output = fc1(input_data)
        expert_output = fc2(expert_output)
        expert_output = batch_norm(expert_output)
        expert_output = silu_activation(expert_output)
        
        output[:, expert_idx, :] = expert_output
    
    return output

# torch.compile 优化版本 - 包含完整开销
@torch.compile
def create_home_expert_sequential_with_bn_compiled(input_data, 
                                                 expert_mlp_weights_fc1, expert_biases_fc1,
                                                 expert_mlp_weights_fc2, expert_biases_fc2,
                                                 bn_weights, bn_biases, running_mean, running_var, 
                                                 meta_output_dim, dim, num_experts, use_bias=True, epsilon=1e-5):
    """
    HoME架构的专家计算逻辑，使用指定的BatchNorm参数 - torch.compile优化版本
    使用两层网络结构: Linear+ReLU -> Linear -> BatchNorm -> SiLU
    注意：包含网络层创建和数据拷贝的完整开销
    """
    batch_size = input_data.size(0)
    hidden_dim = expert_mlp_weights_fc1.shape[2] # FC1的输出维度
    
    output = torch.zeros(batch_size, num_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    for expert_idx in range(num_experts):
        fc1 = DenseLayer(meta_output_dim, hidden_dim, activation="relu", name=f"expert_fc1_{expert_idx}")
        fc2 = DenseLayer(hidden_dim, dim, activation=None, name=f"expert_fc2_{expert_idx}")
        batch_norm = nn.BatchNorm1d(dim)
        silu_activation = nn.SiLU()
        
        with torch.no_grad():
            fc1.linear.weight.data.copy_(expert_mlp_weights_fc1[expert_idx].t())
            if use_bias: fc1.linear.bias.data.copy_(expert_biases_fc1[expert_idx])
            fc2.linear.weight.data.copy_(expert_mlp_weights_fc2[expert_idx].t())
            if use_bias: fc2.linear.bias.data.copy_(expert_biases_fc2[expert_idx])
            batch_norm.weight.data.copy_(bn_weights[expert_idx])
            batch_norm.bias.data.copy_(bn_biases[expert_idx])
            batch_norm.running_mean.data.copy_(running_mean[expert_idx])
            batch_norm.running_var.data.copy_(running_var[expert_idx])
        
        fc1 = fc1.to(input_data.device)
        fc2 = fc2.to(input_data.device)
        batch_norm = batch_norm.to(input_data.device)
        batch_norm.eval()
        
        expert_output = fc1(input_data)
        expert_output = fc2(expert_output)
        expert_output = batch_norm(expert_output)
        expert_output = silu_activation(expert_output)
        
        output[:, expert_idx, :] = expert_output
    
    return output


# 纯计算版本 - 预先创建网络层，只测试前向计算
def create_home_expert_pure_computation(input_data, expert_networks):
    """
    只测试纯前向计算，不包含网络层创建和数据拷贝的开销
    """
    batch_size = input_data.size(0)
    num_experts = len(expert_networks)
    dim = expert_networks[0]['fc2'].linear.out_features  # DenseLayer的输出维度在linear属性中
    
    output = torch.zeros(batch_size, num_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    for expert_idx, expert_net in enumerate(expert_networks):
        fc1 = expert_net['fc1']
        fc2 = expert_net['fc2']
        batch_norm = expert_net['batch_norm']
        silu_activation = expert_net['silu']
        
        expert_output = fc1(input_data)
        expert_output = fc2(expert_output)
        expert_output = batch_norm(expert_output)
        expert_output = silu_activation(expert_output)
        
        output[:, expert_idx, :] = expert_output
    
    return output


# torch.compile 纯计算版本
@torch.compile
def create_home_expert_pure_computation_compiled(input_data, expert_networks):
    """
    torch.compile优化的纯前向计算版本
    """
    batch_size = input_data.size(0)
    num_experts = len(expert_networks)
    dim = expert_networks[0]['fc2'].linear.out_features  # DenseLayer的输出维度在linear属性中
    
    output = torch.zeros(batch_size, num_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    for expert_idx, expert_net in enumerate(expert_networks):
        fc1 = expert_net['fc1']
        fc2 = expert_net['fc2']
        batch_norm = expert_net['batch_norm']
        silu_activation = expert_net['silu']
        
        expert_output = fc1(input_data)
        expert_output = fc2(expert_output)
        expert_output = batch_norm(expert_output)
        expert_output = silu_activation(expert_output)
        
        output[:, expert_idx, :] = expert_output
    
    return output


def prepare_expert_networks(expert_mlp_weights_fc1, expert_biases_fc1,
                           expert_mlp_weights_fc2, expert_biases_fc2,
                           bn_weights, bn_biases, running_mean, running_var,
                           meta_output_dim, dim, num_experts, device, use_bias=True):
    """
    预先创建和配置所有专家网络，用于纯计算测试
    """
    expert_networks = []
    
    for expert_idx in range(num_experts):
        hidden_dim = expert_mlp_weights_fc1.shape[2]
        
        # 创建网络层
        fc1 = DenseLayer(meta_output_dim, hidden_dim, activation="relu", name=f"expert_fc1_{expert_idx}")
        fc2 = DenseLayer(hidden_dim, dim, activation=None, name=f"expert_fc2_{expert_idx}")
        batch_norm = nn.BatchNorm1d(dim)
        silu_activation = nn.SiLU()
        
        # 设置权重和偏置
        with torch.no_grad():
            fc1.linear.weight.data.copy_(expert_mlp_weights_fc1[expert_idx].t())
            if use_bias: fc1.linear.bias.data.copy_(expert_biases_fc1[expert_idx])
            fc2.linear.weight.data.copy_(expert_mlp_weights_fc2[expert_idx].t())
            if use_bias: fc2.linear.bias.data.copy_(expert_biases_fc2[expert_idx])
            batch_norm.weight.data.copy_(bn_weights[expert_idx])
            batch_norm.bias.data.copy_(bn_biases[expert_idx])
            batch_norm.running_mean.data.copy_(running_mean[expert_idx])
            batch_norm.running_var.data.copy_(running_var[expert_idx])
        
        # 移动到设备并设置为评估模式
        fc1 = fc1.to(device)
        fc2 = fc2.to(device)
        batch_norm = batch_norm.to(device)
        batch_norm.eval()
        
        expert_networks.append({
            'fc1': fc1,
            'fc2': fc2,
            'batch_norm': batch_norm,
            'silu': silu_activation
        })
    
    return expert_networks


def create_simple_linear_expert(input_data, expert_mlp_weights, expert_biases, expert_indices, 
                                meta_output_dim, dim, num_experts, use_bias=True):
    """
    简化的线性专家实现（仅用于对比）
    只使用Linear层，不包含BatchNorm和SiLU
    
    Args:
        input_data: [batch_size, meta_output_dim] - 专家输入（来自Meta层）
        expert_mlp_weights: [num_experts, meta_output_dim, dim] - MLP层权重
        expert_biases: [num_experts, dim] - MLP层偏置
        expert_indices: [num_selected_experts] - 选中的专家索引
        meta_output_dim: Meta层输出维度 = dim * 2
        dim: 专家输出维度
        num_experts: 专家总数
    """
    batch_size = input_data.size(0)
    num_selected_experts = expert_indices.size(0)
    
    # 分配输出内存
    output = torch.zeros(batch_size, num_selected_experts, dim, 
                       device=input_data.device, dtype=input_data.dtype)
    
    # 简化的线性计算
    for i, expert_idx in enumerate(expert_indices):
        expert_idx = expert_idx.item()
        if expert_idx < num_experts:
            # 获取当前专家的权重和偏置
            expert_weight = expert_mlp_weights[expert_idx]  # [meta_output_dim, dim]
            expert_bias = expert_biases[expert_idx] if use_bias else None  # [dim]
            
            # 计算专家输出: input @ weight + bias
            expert_output = torch.matmul(input_data, expert_weight)  # [batch_size, dim]
            
            if use_bias and expert_bias is not None:
                expert_output = expert_output + expert_bias.unsqueeze(0)  # 广播偏置
            
            # 存储到输出张量中
            output[:, i, :] = expert_output
    
    return output


def run_performance_test(func, num_runs=10, warmup_runs=3):
    """运行性能测试"""
    times = []
    
    # 预热
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = func()
    
    # 测试
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计信息
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'times': times
    }


def run_comparison(batch_size=4096, dim=700, num_experts=5, num_runs=10):
    """运行对比测试 - 高吞吐量测试用例"""
    print(f"\n{'='*80}")
    print(f"HoME vs CUDA 专家前向传播对比测试")
    print(f"配置: batch_size={batch_size}, dim={dim}, num_experts={num_experts}")
    print(f"HoME模型结构: meta_output_dim={dim*2}, expert_output_dim={dim}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # HoME模型参数
    meta_output_dim = dim * 2  # Meta层输出维度 = 1400
    expert_output_dim = dim    # 专家输出维度 = 700
    
    # 创建测试数据 - 使用固定种子确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 专家输入来自Meta层，维度为meta_output_dim
    input_data = torch.randn(batch_size, meta_output_dim, device=device)
    
    # FC1权重和偏置
    expert_mlp_weights_fc1 = torch.randn(num_experts, meta_output_dim, expert_output_dim, device=device)
    expert_biases_fc1 = torch.randn(num_experts, expert_output_dim, device=device)
    
    # FC2权重和偏置
    expert_mlp_weights_fc2 = torch.randn(num_experts, expert_output_dim, expert_output_dim, device=device)
    expert_biases_fc2 = torch.randn(num_experts, expert_output_dim, device=device)
    
    expert_indices = torch.arange(num_experts, dtype=torch.int32, device=device)
    
    # BatchNorm参数 (维度与FC1输出维度一致)
    bn_weights = torch.randn(num_experts, expert_output_dim, device=device)
    bn_biases = torch.randn(num_experts, expert_output_dim, device=device)
    running_mean = torch.randn(num_experts, expert_output_dim, device=device)
    running_var = torch.ones(num_experts, expert_output_dim, device=device)
    
    print(f"数据形状验证:")
    print(f"  input_data: {input_data.shape}")
    print(f"  expert_mlp_weights_fc1: {expert_mlp_weights_fc1.shape}")
    print(f"  expert_mlp_weights_fc2: {expert_mlp_weights_fc2.shape}")
    print(f"  expert_biases_fc1: {expert_biases_fc1.shape}")
    print(f"  expert_biases_fc2: {expert_biases_fc2.shape}")
    print(f"  expert_indices: {expert_indices.shape}")
    print(f"  bn_weights: {bn_weights.shape}")
    print(f"  bn_biases: {bn_biases.shape}")
    print(f"  running_mean: {running_mean.shape}")
    print(f"  running_var: {running_var.shape}")
    
    results = {}
    
    # 1. CUDA Kernel实现
    if CUDA_KERNELS_AVAILABLE:
        print(f"\n1. CUDA Kernel实现 (home_kernels.home_expert_forward)")
        print("-" * 60)
        
        try:
            # --- 调试: 只测试 FC1 + ReLU ---
            with torch.no_grad():
                cuda_output = home_kernels.home_expert_forward(
                    input_data, 
                    expert_mlp_weights_fc1, expert_biases_fc1,
                    # 传入虚拟张量以满足函数签名
                    expert_mlp_weights_fc2, expert_biases_fc2,
                    bn_weights, bn_biases, running_mean, running_var,
                    num_experts, True, 1e-5
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            
            # 性能测试
            def cuda_func():
                return home_kernels.home_expert_forward(
                    input_data, 
                    expert_mlp_weights_fc1, expert_biases_fc1,
                    expert_mlp_weights_fc2, expert_biases_fc2,
                    bn_weights, bn_biases, running_mean, running_var,
                    num_experts, True, 1e-5
                )
            
            cuda_perf = run_performance_test(cuda_func, num_runs)
            print(f"  性能统计:")
            print(f"    平均时间: {cuda_perf['avg_time']:.4f} ms")
            print(f"    最小时间: {cuda_perf['min_time']:.4f} ms")
            print(f"    最大时间: {cuda_perf['max_time']:.4f} ms")
            print(f"    标准差: {cuda_perf['std_time']:.4f} ms")
            
            results['cuda'] = {
                'output': cuda_output,
                'performance': cuda_perf
            }
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            results['cuda'] = None
    
    # 2. HoME.py中的实际实现 (Sequential结构)
    print(f"\n2. HoME.py实际实现 (Sequential(MLPLayer, BatchNorm, SiLU))")
    print("-" * 60)
    
    try:
        with torch.no_grad():
            home_output = create_home_expert_sequential_with_bn(
                input_data, 
                expert_mlp_weights_fc1, expert_biases_fc1,
                expert_mlp_weights_fc2, expert_biases_fc2,
                bn_weights, bn_biases, running_mean, running_var, 
                meta_output_dim, expert_output_dim, num_experts, True
            )
        
        print(f"✓ HoME.py实际实现执行成功")
        print(f"  输出形状: {home_output.shape}")
        
        # 性能测试
        def home_func():
            return create_home_expert_sequential_with_bn(
                input_data, 
                expert_mlp_weights_fc1, expert_biases_fc1,
                expert_mlp_weights_fc2, expert_biases_fc2,
                bn_weights, bn_biases, running_mean, running_var, 
                meta_output_dim, expert_output_dim, num_experts, True
            )
        
        home_perf = run_performance_test(home_func, num_runs)
        print(f"  性能统计:")
        print(f"    平均时间: {home_perf['avg_time']:.4f} ms")
        print(f"    最小时间: {home_perf['min_time']:.4f} ms")
        print(f"    最大时间: {home_perf['max_time']:.4f} ms")
        print(f"    标准差: {home_perf['std_time']:.4f} ms")
        
        results['home'] = {
            'output': home_output,
            'performance': home_perf
        }
        
    except Exception as e:
        print(f"✗ HoME.py实际实现执行失败: {e}")
        results['home'] = None

    # 3. HoME.py中的实际实现 (torch.compile优化版本)
    print(f"\n3. HoME.py实际实现 (torch.compile优化版本)")
    print("-" * 60)
    
    try:
        with torch.no_grad():
            home_compiled_output = create_home_expert_sequential_with_bn_compiled(
                input_data, 
                expert_mlp_weights_fc1, expert_biases_fc1,
                expert_mlp_weights_fc2, expert_biases_fc2,
                bn_weights, bn_biases, running_mean, running_var, 
                meta_output_dim, expert_output_dim, num_experts, True
            )
        
        print(f"✓ HoME.py torch.compile版本执行成功")
        print(f"  输出形状: {home_compiled_output.shape}")
        
        # 性能测试
        def home_compiled_func():
            return create_home_expert_sequential_with_bn_compiled(
                input_data, 
                expert_mlp_weights_fc1, expert_biases_fc1,
                expert_mlp_weights_fc2, expert_biases_fc2,
                bn_weights, bn_biases, running_mean, running_var, 
                meta_output_dim, expert_output_dim, num_experts, True
            )
        
        home_compiled_perf = run_performance_test(home_compiled_func, num_runs)
        print(f"  性能统计:")
        print(f"    平均时间: {home_compiled_perf['avg_time']:.4f} ms")
        print(f"    最小时间: {home_compiled_perf['min_time']:.4f} ms")
        print(f"    最大时间: {home_compiled_perf['max_time']:.4f} ms")
        print(f"    标准差: {home_compiled_perf['std_time']:.4f} ms")
        
        results['home_compiled'] = {
            'output': home_compiled_output,
            'performance': home_compiled_perf
        }
        
    except Exception as e:
        print(f"✗ HoME.py torch.compile版本执行失败: {e}")
        results['home_compiled'] = None

    # 4. HoME.py纯计算版本测试 (预先创建网络层)
    print(f"\n4. HoME.py纯计算版本 (预先创建网络层)")
    print("-" * 60)
    
    try:
        # 预先创建专家网络（这部分不计入性能测试）
        print("  正在预先创建专家网络...")
        expert_networks = prepare_expert_networks(
            expert_mlp_weights_fc1, expert_biases_fc1,
            expert_mlp_weights_fc2, expert_biases_fc2,
            bn_weights, bn_biases, running_mean, running_var,
            meta_output_dim, expert_output_dim, num_experts, device, True
        )
        
        # 测试纯计算版本
        with torch.no_grad():
            home_pure_output = create_home_expert_pure_computation(input_data, expert_networks)
        
        print(f"✓ HoME.py纯计算版本执行成功")
        print(f"  输出形状: {home_pure_output.shape}")
        
        # 性能测试（只测试纯前向计算）
        def home_pure_func():
            return create_home_expert_pure_computation(input_data, expert_networks)
        
        home_pure_perf = run_performance_test(home_pure_func, num_runs)
        print(f"  性能统计:")
        print(f"    平均时间: {home_pure_perf['avg_time']:.4f} ms")
        print(f"    最小时间: {home_pure_perf['min_time']:.4f} ms")
        print(f"    最大时间: {home_pure_perf['max_time']:.4f} ms")
        print(f"    标准差: {home_pure_perf['std_time']:.4f} ms")
        
        results['home_pure'] = {
            'output': home_pure_output,
            'performance': home_pure_perf
        }
        
    except Exception as e:
        print(f"✗ HoME.py纯计算版本执行失败: {e}")
        results['home_pure'] = None

    # 5. HoME.py纯计算版本 + torch.compile
    print(f"\n5. HoME.py纯计算版本 + torch.compile")
    print("-" * 60)
    
    try:
        if 'expert_networks' in locals():
            # 测试torch.compile纯计算版本
            with torch.no_grad():
                home_pure_compiled_output = create_home_expert_pure_computation_compiled(input_data, expert_networks)
            
            print(f"✓ HoME.py纯计算+torch.compile版本执行成功")
            print(f"  输出形状: {home_pure_compiled_output.shape}")
            
            # 性能测试（只测试纯前向计算）
            def home_pure_compiled_func():
                return create_home_expert_pure_computation_compiled(input_data, expert_networks)
            
            home_pure_compiled_perf = run_performance_test(home_pure_compiled_func, num_runs)
            print(f"  性能统计:")
            print(f"    平均时间: {home_pure_compiled_perf['avg_time']:.4f} ms")
            print(f"    最小时间: {home_pure_compiled_perf['min_time']:.4f} ms")
            print(f"    最大时间: {home_pure_compiled_perf['max_time']:.4f} ms")
            print(f"    标准差: {home_pure_compiled_perf['std_time']:.4f} ms")
            
            results['home_pure_compiled'] = {
                'output': home_pure_compiled_output,
                'performance': home_pure_compiled_perf
            }
        else:
            print(f"✗ 专家网络未创建，跳过测试")
            results['home_pure_compiled'] = None
        
    except Exception as e:
        print(f"✗ HoME.py纯计算+torch.compile版本执行失败: {e}")
        results['home_pure_compiled'] = None
    
    
    # 6. 精度对比分析
    print(f"\n6. 精度对比分析")
    print("-" * 60)
    
    if results.get('cuda') and results.get('home'):
        cuda_output = results['cuda']['output']
        home_output = results['home']['output']
        
        # 计算数值差异
        diff = torch.abs(cuda_output - home_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"CUDA Kernel vs HoME.py实际实现:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"  ✓ 数值精度验证通过")
        else:
            print(f"  ⚠️  数值精度可能存在差异")
    else:
        print("⚠️  无法进行CUDA vs HoME精度对比，某些实现失败")
    
    # 验证纯计算版本的精度
    if results.get('home') and results.get('home_pure'):
        home_output = results['home']['output']
        home_pure_output = results['home_pure']['output']
        
        # 计算数值差异
        diff = torch.abs(home_output - home_pure_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\nHoME.py完整版 vs 纯计算版:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"  ✓ 纯计算版本数值精度验证通过")
        else:
            print(f"  ⚠️  纯计算版本可能存在数值差异")
    
    # 验证torch.compile纯计算版本的精度
    if results.get('home_pure') and results.get('home_pure_compiled'):
        home_pure_output = results['home_pure']['output']
        home_pure_compiled_output = results['home_pure_compiled']['output']
        
        # 计算数值差异
        diff = torch.abs(home_pure_output - home_pure_compiled_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\n纯计算版 vs 纯计算+torch.compile版:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"  ✓ torch.compile纯计算版本数值精度验证通过")
        else:
            print(f"  ⚠️  torch.compile纯计算版本可能存在数值差异")
    
    # 7. 性能对比分析
    print(f"\n7. 性能对比分析")
    print("-" * 60)
    
    # 完整版本对比
    if results.get('cuda') and results.get('home'):
        cuda_time = results['cuda']['performance']['avg_time']
        home_time = results['home']['performance']['avg_time']
        speedup = home_time / cuda_time
        
        print(f"CUDA Kernel vs HoME.py完整版本:")
        print(f"  CUDA Kernel: {cuda_time:.4f} ms")
        print(f"  HoME.py完整版本: {home_time:.4f} ms")
        print(f"  加速比: {speedup:.2f}x {'(CUDA更快)' if speedup > 1 else '(HoME.py更快)'}")
    else:
        print("⚠️  无法进行CUDA vs HoME完整版性能对比，某些实现失败")
    
    # 纯计算版本对比
    if results.get('cuda') and results.get('home_pure'):
        cuda_time = results['cuda']['performance']['avg_time']
        home_pure_time = results['home_pure']['performance']['avg_time']
        pure_speedup = home_pure_time / cuda_time
        
        print(f"\nCUDA Kernel vs HoME.py纯计算版本:")
        print(f"  CUDA Kernel: {cuda_time:.4f} ms")
        print(f"  HoME.py纯计算版本: {home_pure_time:.4f} ms")
        print(f"  加速比: {pure_speedup:.2f}x {'(CUDA更快)' if pure_speedup > 1 else '(纯计算版更快)'}")
    
    # torch.compile纯计算版本对比
    if results.get('cuda') and results.get('home_pure_compiled'):
        cuda_time = results['cuda']['performance']['avg_time']
        home_pure_compiled_time = results['home_pure_compiled']['performance']['avg_time']
        pure_compile_speedup = home_pure_compiled_time / cuda_time
        
        print(f"\nCUDA Kernel vs HoME.py纯计算+torch.compile版本:")
        print(f"  CUDA Kernel: {cuda_time:.4f} ms")
        print(f"  纯计算+torch.compile版本: {home_pure_compiled_time:.4f} ms")
        print(f"  加速比: {pure_compile_speedup:.2f}x {'(CUDA更快)' if pure_compile_speedup > 1 else '(compile版更快)'}")
    
    # 数据拷贝开销分析
    if results.get('home') and results.get('home_pure'):
        home_time = results['home']['performance']['avg_time']
        home_pure_time = results['home_pure']['performance']['avg_time']
        copy_overhead = home_time - home_pure_time
        copy_overhead_ratio = copy_overhead / home_time * 100
        
        print(f"\n数据拷贝开销分析:")
        print(f"  完整版本时间: {home_time:.4f} ms")
        print(f"  纯计算时间: {home_pure_time:.4f} ms")
        print(f"  数据拷贝开销: {copy_overhead:.4f} ms ({copy_overhead_ratio:.1f}%)")
    
    # torch.compile效果对比
    if results.get('home_pure') and results.get('home_pure_compiled'):
        home_pure_time = results['home_pure']['performance']['avg_time']
        home_pure_compiled_time = results['home_pure_compiled']['performance']['avg_time']
        compile_speedup = home_pure_time / home_pure_compiled_time
        
        print(f"\ntorch.compile效果分析（纯计算）:")
        print(f"  PyTorch原版: {home_pure_time:.4f} ms")
        print(f"  torch.compile版: {home_pure_compiled_time:.4f} ms")
        print(f"  torch.compile加速比: {compile_speedup:.2f}x {'(compile更快)' if compile_speedup > 1 else '(原版更快)'}")
    
    # 8. GPU内存使用
    if torch.cuda.is_available():
        print(f"\n8. GPU内存使用")
        print("-" * 60)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"已分配: {allocated:.2f} GB")
        print(f"已缓存: {reserved:.2f} GB")
    
    return results


def main():
    """主函数 - 高吞吐量测试用例"""
    print("HoME vs CUDA 专家前向传播对比测试")
    print("=" * 60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 运行高吞吐量测试用例
    print(f"\n{'='*80}")
    print(f"高吞吐量测试用例")
    print(f"{'='*80}")
    
    results = run_comparison(batch_size=4096, dim=700, num_experts=5, num_runs=10)
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ 对比测试完成!")


if __name__ == "__main__":
    main()
