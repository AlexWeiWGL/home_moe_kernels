#!/usr/bin/env python3
"""
简化的专家前向传播性能对比测试
只对比纯计算部分，不包含数据拷贝开销

对比项目：
1. CUDA Kernel实现
2. PyTorch纯计算版本 
3. PyTorch纯计算版本 + torch.compile
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
    from Dense import DenseLayer
    HOME_AVAILABLE = True
    print("✓ HoME模块导入成功")
except ImportError as e:
    print(f"⚠️  HoME模块导入失败: {e}")
    HOME_AVAILABLE = False


# PyTorch纯计算版本
def pytorch_expert_forward(input_data, expert_networks):
    """
    PyTorch纯计算版本 - 预先创建好网络层，只测试前向计算
    """
    batch_size = input_data.size(0)
    num_experts = len(expert_networks)
    dim = expert_networks[0]['fc2'].linear.out_features
    
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


# PyTorch纯计算版本 + torch.compile
@torch.compile
def pytorch_expert_forward_compiled(input_data, expert_networks):
    """
    PyTorch纯计算版本 + torch.compile优化
    """
    batch_size = input_data.size(0)
    num_experts = len(expert_networks)
    dim = expert_networks[0]['fc2'].linear.out_features
    
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
    预先创建和配置所有专家网络（这部分不计入性能测试）
    """
    print("  正在预先创建专家网络...")
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
    
    print("  ✓ 专家网络创建完成")
    return expert_networks


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
    """运行简化的性能对比测试 - 只对比纯计算部分"""
    print(f"\n{'='*80}")
    print(f"专家前向传播纯计算性能对比测试")
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
    
    # 创建权重和偏置
    expert_mlp_weights_fc1 = torch.randn(num_experts, meta_output_dim, expert_output_dim, device=device)
    expert_biases_fc1 = torch.randn(num_experts, expert_output_dim, device=device)
    expert_mlp_weights_fc2 = torch.randn(num_experts, expert_output_dim, expert_output_dim, device=device)
    expert_biases_fc2 = torch.randn(num_experts, expert_output_dim, device=device)
    
    # BatchNorm参数
    bn_weights = torch.randn(num_experts, expert_output_dim, device=device)
    bn_biases = torch.randn(num_experts, expert_output_dim, device=device)
    running_mean = torch.randn(num_experts, expert_output_dim, device=device)
    running_var = torch.ones(num_experts, expert_output_dim, device=device)
    
    print(f"\n数据形状验证:")
    print(f"  input_data: {input_data.shape}")
    print(f"  expert_mlp_weights_fc1: {expert_mlp_weights_fc1.shape}")
    print(f"  expert_mlp_weights_fc2: {expert_mlp_weights_fc2.shape}")
    
    results = {}
    
    # 预先创建专家网络（不计入性能测试）
    print(f"\n准备阶段:")
    expert_networks = prepare_expert_networks(
        expert_mlp_weights_fc1, expert_biases_fc1,
        expert_mlp_weights_fc2, expert_biases_fc2,
        bn_weights, bn_biases, running_mean, running_var,
        meta_output_dim, expert_output_dim, num_experts, device, True
    )
    
    # 1. CUDA Kernel实现
    if CUDA_KERNELS_AVAILABLE:
        print(f"\n1. CUDA Kernel实现")
        print("-" * 60)
        
        try:
            with torch.no_grad():
                cuda_output = home_kernels.home_expert_forward(
                    input_data, 
                    expert_mlp_weights_fc1, expert_biases_fc1,
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

    # 2. PyTorch纯计算版本
    print(f"\n2. PyTorch纯计算版本")
    print("-" * 60)
    
    try:
        with torch.no_grad():
            pytorch_output = pytorch_expert_forward(input_data, expert_networks)
        
        print(f"✓ PyTorch纯计算版本执行成功")
        print(f"  输出形状: {pytorch_output.shape}")
        
        # 性能测试
        def pytorch_func():
            return pytorch_expert_forward(input_data, expert_networks)
        
        pytorch_perf = run_performance_test(pytorch_func, num_runs)
        print(f"  性能统计:")
        print(f"    平均时间: {pytorch_perf['avg_time']:.4f} ms")
        print(f"    最小时间: {pytorch_perf['min_time']:.4f} ms")
        print(f"    最大时间: {pytorch_perf['max_time']:.4f} ms")
        print(f"    标准差: {pytorch_perf['std_time']:.4f} ms")
        
        results['pytorch'] = {
            'output': pytorch_output,
            'performance': pytorch_perf
        }
        
    except Exception as e:
        print(f"✗ PyTorch纯计算版本执行失败: {e}")
        results['pytorch'] = None

    # 3. PyTorch纯计算版本 + torch.compile
    print(f"\n3. PyTorch纯计算版本 + torch.compile")
    print("-" * 60)
    
    try:
        with torch.no_grad():
            pytorch_compiled_output = pytorch_expert_forward_compiled(input_data, expert_networks)
        
        print(f"✓ PyTorch+torch.compile版本执行成功")
        print(f"  输出形状: {pytorch_compiled_output.shape}")
        
        # 性能测试
        def pytorch_compiled_func():
            return pytorch_expert_forward_compiled(input_data, expert_networks)
        
        pytorch_compiled_perf = run_performance_test(pytorch_compiled_func, num_runs)
        print(f"  性能统计:")
        print(f"    平均时间: {pytorch_compiled_perf['avg_time']:.4f} ms")
        print(f"    最小时间: {pytorch_compiled_perf['min_time']:.4f} ms")
        print(f"    最大时间: {pytorch_compiled_perf['max_time']:.4f} ms")
        print(f"    标准差: {pytorch_compiled_perf['std_time']:.4f} ms")
        
        results['pytorch_compiled'] = {
            'output': pytorch_compiled_output,
            'performance': pytorch_compiled_perf
        }
        
    except Exception as e:
        print(f"✗ PyTorch+torch.compile版本执行失败: {e}")
        results['pytorch_compiled'] = None
    
    # 4. 精度对比分析
    print(f"\n4. 精度对比分析")
    print("-" * 60)
    
    if results.get('cuda') and results.get('pytorch'):
        cuda_output = results['cuda']['output']
        pytorch_output = results['pytorch']['output']
        
        # 计算数值差异
        diff = torch.abs(cuda_output - pytorch_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"CUDA Kernel vs PyTorch:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"  ✓ 数值精度验证通过")
        else:
            print(f"  ⚠️  数值精度可能存在差异")
    
    # 验证torch.compile版本的精度
    if results.get('pytorch') and results.get('pytorch_compiled'):
        pytorch_output = results['pytorch']['output']
        pytorch_compiled_output = results['pytorch_compiled']['output']
        
        # 计算数值差异
        diff = torch.abs(pytorch_output - pytorch_compiled_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\nPyTorch vs PyTorch+torch.compile:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print(f"  ✓ torch.compile版本数值精度验证通过")
        else:
            print(f"  ⚠️  torch.compile版本可能存在数值差异")
    
    # 5. 性能对比分析
    print(f"\n5. 性能对比分析")
    print("-" * 60)
    
    # 收集所有性能数据
    perf_data = []
    if results.get('cuda'):
        perf_data.append(('CUDA Kernel', results['cuda']['performance']['avg_time']))
    if results.get('pytorch'):
        perf_data.append(('PyTorch', results['pytorch']['performance']['avg_time']))
    if results.get('pytorch_compiled'):
        perf_data.append(('PyTorch+torch.compile', results['pytorch_compiled']['performance']['avg_time']))
    
    # 排序并显示
    perf_data.sort(key=lambda x: x[1])
    
    print("性能排名（从快到慢）:")
    for i, (name, time_ms) in enumerate(perf_data, 1):
        fastest_time = perf_data[0][1]
        speedup = time_ms / fastest_time
        print(f"  {i}. {name}: {time_ms:.4f} ms ({speedup:.2f}x)")
    
    # 详细对比
    if len(perf_data) >= 2:
        print(f"\n详细对比:")
        fastest_name, fastest_time = perf_data[0]
        for name, time_ms in perf_data[1:]:
            speedup = time_ms / fastest_time
            print(f"  {fastest_name} vs {name}: {speedup:.2f}x 加速")
    
    # 6. GPU内存使用
    if torch.cuda.is_available():
        print(f"\n6. GPU内存使用")
        print("-" * 60)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"已分配: {allocated:.2f} GB")
        print(f"已缓存: {reserved:.2f} GB")
    
    return results


def main():
    """主函数"""
    print("专家前向传播纯计算性能对比测试")
    print("=" * 60)
    print("注意：此测试只对比纯计算部分，不包含数据拷贝开销")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 运行对比测试
    results = run_comparison(batch_size=4096, dim=700, num_experts=5, num_runs=10)
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ 纯计算性能对比测试完成!")
    print("\n总结:")
    print("- 此测试排除了数据拷贝开销，只对比实际计算性能")
    print("- 结果更能反映各种实现的真实计算效率")
    print("- 适合用于算法优化和性能调优的参考")


if __name__ == "__main__":
    main()
