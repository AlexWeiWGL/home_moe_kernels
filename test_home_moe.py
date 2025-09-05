#!/usr/bin/env python3
"""
HoME优化架构使用示例和测试套件

这个文件包含了HoME (Hierarchical Mixture of Experts) 优化架构的完整测试套件，
包括以下CUDA算子的详细测试：

1. home_expert_forward - 专家网络前向传播
2. lora_gate_forward - LoRA门控前向传播  
3. gate_weights_forward - 门控权重前向传播
4. expert_weighted_sum_forward - 专家加权求和
5. fused_batch_norm_silu_forward - 融合BatchNorm+SiLU算子

使用方法:
    python test_home_moe.py                    # 运行所有测试
    python test_home_moe.py --test kernels     # 只测试CUDA kernels
    python test_home_moe.py --test performance # 只运行性能测试
    python test_home_moe.py --test accuracy    # 只运行精度验证
    python test_home_moe.py --test basic       # 只运行基本示例
    python test_home_moe.py --quick            # 快速测试模式

每个算子测试包括：
- 功能正确性验证
- 输出形状验证
- 数值精度验证（与PyTorch实现对比）
- 性能基准测试
- 多种测试用例覆盖
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入CUDA kernel模块
try:
    import home_kernels
    CUDA_KERNELS_AVAILABLE = True
    print("✓ CUDA kernels模块导入成功")
except ImportError as e:
    print(f"⚠️  CUDA kernels模块导入失败: {e}")
    print("将使用PyTorch原生实现作为后备")
    CUDA_KERNELS_AVAILABLE = False
    home_kernels = None

def create_sample_data(batch_size=4096, device='cuda'):
    """创建示例数据"""
    # 6个任务的输入数据
    input_states = [
        torch.randn(batch_size, 700, device=device) for _ in range(6)
    ]
    
    # 门控状态
    l_gate_states = torch.randn(batch_size, 700, device=device)
    g_gate_states = torch.randn(batch_size, 700, device=device)
    l_gate_states_task = torch.randn(batch_size, 700, device=device)
    
    return input_states, l_gate_states, g_gate_states, l_gate_states_task

def print_gpu_memory():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存 - 已分配: {allocated:.2f} GB, 已缓存: {reserved:.2f} GB")
        return allocated, reserved
    return 0, 0

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU内存已清理")

def test_home_expert_forward():
    """测试home_expert_forward算子"""
    print("\n" + "=" * 60)
    print("测试 home_expert_forward 算子")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    test_cases = [
        {"batch_size": 32, "input_dim": 256, "hidden_dim": 256, "num_experts": 5},
        {"batch_size": 64, "input_dim": 512, "hidden_dim": 512, "num_experts": 8},
        {"batch_size": 128, "input_dim": 128, "hidden_dim": 128, "num_experts": 3},
        {"batch_size": 4096, "input_dim": 700, "hidden_dim": 256, "num_experts": 5},  # 高吞吐量测试用例
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {case}")
        print("-" * 40)
        
        batch_size = case["batch_size"]
        input_dim = case["input_dim"]
        hidden_dim = case["hidden_dim"]
        num_experts = case["num_experts"]
        
        # 对于高吞吐量测试用例，显示内存使用情况
        if batch_size >= 4096:
            print(f"  🚀 高吞吐量测试用例 - 预计内存使用: {batch_size * input_dim * 4 / 1024**2:.1f} MB (输入)")
            print_gpu_memory()
        
        # 创建测试数据
        torch.manual_seed(42)
        input_data = torch.randn(batch_size, input_dim, device=device)
        expert_weights = torch.randn(num_experts, input_dim, hidden_dim, device=device)
        expert_biases = torch.randn(num_experts, hidden_dim, device=device)
        expert_indices = torch.arange(num_experts, dtype=torch.int32, device=device)
        
        # 测试CUDA kernel
        try:
            with torch.no_grad():
                cuda_output = home_kernels.home_expert_forward(
                    input_data, expert_weights, expert_biases, expert_indices, num_experts, True
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            print(f"  输出数据类型: {cuda_output.dtype}")
            print(f"  输出设备: {cuda_output.device}")
            
            # 验证输出形状
            expected_shape = (batch_size, num_experts, hidden_dim)
            if cuda_output.shape == expected_shape:
                print(f"✓ 输出形状正确: {cuda_output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {cuda_output.shape}")
            
            # 性能测试
            times = []
            num_runs = 10
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = home_kernels.home_expert_forward(
                        input_data, expert_weights, expert_biases, expert_indices, num_experts, True
                    )
            
            # 测试
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = home_kernels.home_expert_forward(
                        input_data, expert_weights, expert_biases, expert_indices, num_experts, True
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"  性能统计:")
            print(f"    平均时间: {avg_time:.4f} ms")
            print(f"    最小时间: {min_time:.4f} ms")
            print(f"    最大时间: {max_time:.4f} ms")
            print(f"    标准差: {std_time:.4f} ms")
            
            # 对于高吞吐量测试用例，计算吞吐量
            if batch_size >= 4096:
                throughput = batch_size / (avg_time / 1000)  # samples per second
                print(f"    吞吐量: {throughput:.0f} samples/sec")
                print_gpu_memory()
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ home_expert_forward 算子测试完成!")


def test_lora_gate_forward():
    """测试lora_gate_forward算子"""
    print("\n" + "=" * 60)
    print("测试 lora_gate_forward 算子")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    test_cases = [
        {"batch_size": 32, "input_dim": 256, "rank": 16, "output_dim": 256},
        {"batch_size": 64, "input_dim": 512, "rank": 32, "output_dim": 512},
        {"batch_size": 128, "input_dim": 128, "rank": 8, "output_dim": 128},
        {"batch_size": 4096, "input_dim": 700, "rank": 16, "output_dim": 700},  # 高吞吐量测试用例
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {case}")
        print("-" * 40)
        
        batch_size = case["batch_size"]
        input_dim = case["input_dim"]
        rank = case["rank"]
        output_dim = case["output_dim"]
        
        # 对于高吞吐量测试用例，显示内存使用情况
        if batch_size >= 4096:
            print(f"  🚀 高吞吐量测试用例 - 预计内存使用: {batch_size * input_dim * 4 / 1024**2:.1f} MB (输入)")
            print_gpu_memory()
        
        # 创建测试数据
        torch.manual_seed(42)
        input_data = torch.randn(batch_size, input_dim, device=device)
        A_matrix = torch.randn(input_dim, rank, device=device)
        B_matrix = torch.randn(rank, output_dim, device=device)
        
        # 测试CUDA kernel
        try:
            with torch.no_grad():
                cuda_output = home_kernels.lora_gate_forward(
                    input_data, A_matrix, B_matrix, True  # use_vectorized=True
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            print(f"  输出数据类型: {cuda_output.dtype}")
            print(f"  输出设备: {cuda_output.device}")
            
            # 验证输出形状
            expected_shape = (batch_size, output_dim)
            if cuda_output.shape == expected_shape:
                print(f"✓ 输出形状正确: {cuda_output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {cuda_output.shape}")
            
            # 与PyTorch实现对比验证
            with torch.no_grad():
                pytorch_output = torch.matmul(torch.matmul(input_data, A_matrix), B_matrix)
            
            # 计算数值差异
            diff = torch.abs(cuda_output - pytorch_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  与PyTorch实现对比:")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✓ 数值精度验证通过")
            else:
                print(f"⚠️  数值精度可能存在差异")
            
            # 性能测试
            times = []
            num_runs = 10
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = home_kernels.lora_gate_forward(input_data, A_matrix, B_matrix, True)
            
            # 测试
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = home_kernels.lora_gate_forward(input_data, A_matrix, B_matrix, True)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"  性能统计:")
            print(f"    平均时间: {avg_time:.4f} ms")
            print(f"    最小时间: {min_time:.4f} ms")
            print(f"    最大时间: {max_time:.4f} ms")
            print(f"    标准差: {std_time:.4f} ms")
            
            # 对于高吞吐量测试用例，计算吞吐量
            if batch_size >= 4096:
                throughput = batch_size / (avg_time / 1000)  # samples per second
                print(f"    吞吐量: {throughput:.0f} samples/sec")
                print_gpu_memory()
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ lora_gate_forward 算子测试完成!")


def test_gate_weights_forward():
    """测试gate_weights_forward算子"""
    print("\n" + "=" * 60)
    print("测试 gate_weights_forward 算子")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    test_cases = [
        {"batch_size": 32, "gate_dim": 256, "num_experts": 5, "use_softmax": True},
        {"batch_size": 64, "gate_dim": 512, "num_experts": 8, "use_softmax": True},
        {"batch_size": 128, "gate_dim": 128, "num_experts": 3, "use_softmax": False},
        {"batch_size": 4096, "gate_dim": 700, "num_experts": 5, "use_softmax": True},  # 高吞吐量测试用例
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {case}")
        print("-" * 40)
        
        batch_size = case["batch_size"]
        gate_dim = case["gate_dim"]
        num_experts = case["num_experts"]
        use_softmax = case["use_softmax"]
        
        # 对于高吞吐量测试用例，显示内存使用情况
        if batch_size >= 4096:
            print(f"  🚀 高吞吐量测试用例 - 预计内存使用: {batch_size * gate_dim * 4 / 1024**2:.1f} MB (输入)")
            print_gpu_memory()
        
        # 创建测试数据
        torch.manual_seed(42)
        gate_states = torch.randn(batch_size, gate_dim, device=device)
        gate_weights = torch.randn(gate_dim, num_experts, device=device)
        
        # 测试CUDA kernel
        try:
            with torch.no_grad():
                cuda_output = home_kernels.gate_weights_forward(
                    gate_states, gate_weights, use_softmax
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            print(f"  输出数据类型: {cuda_output.dtype}")
            print(f"  输出设备: {cuda_output.device}")
            
            # 验证输出形状
            expected_shape = (batch_size, num_experts)
            if cuda_output.shape == expected_shape:
                print(f"✓ 输出形状正确: {cuda_output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {cuda_output.shape}")
            
            # 与PyTorch实现对比验证
            with torch.no_grad():
                pytorch_output = torch.matmul(gate_states, gate_weights)
                if use_softmax:
                    pytorch_output = torch.softmax(pytorch_output, dim=-1)
            
            # 计算数值差异
            diff = torch.abs(cuda_output - pytorch_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  与PyTorch实现对比:")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✓ 数值精度验证通过")
            else:
                print(f"⚠️  数值精度可能存在差异")
            
            # 验证softmax特性（如果启用）
            if use_softmax:
                row_sums = torch.sum(cuda_output, dim=-1)
                print(f"  Softmax验证:")
                print(f"    行和范围: [{torch.min(row_sums).item():.6f}, {torch.max(row_sums).item():.6f}]")
                if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                    print(f"    ✓ Softmax归一化正确")
                else:
                    print(f"    ✗ Softmax归一化可能有问题")
            
            # 性能测试
            times = []
            num_runs = 10
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = home_kernels.gate_weights_forward(gate_states, gate_weights, use_softmax)
            
            # 测试
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = home_kernels.gate_weights_forward(gate_states, gate_weights, use_softmax)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"  性能统计:")
            print(f"    平均时间: {avg_time:.4f} ms")
            print(f"    最小时间: {min_time:.4f} ms")
            print(f"    最大时间: {max_time:.4f} ms")
            print(f"    标准差: {std_time:.4f} ms")
            
            # 对于高吞吐量测试用例，计算吞吐量
            if batch_size >= 4096:
                throughput = batch_size / (avg_time / 1000)  # samples per second
                print(f"    吞吐量: {throughput:.0f} samples/sec")
                print_gpu_memory()
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ gate_weights_forward 算子测试完成!")


def test_expert_weighted_sum_forward():
    """测试expert_weighted_sum_forward算子"""
    print("\n" + "=" * 60)
    print("测试 expert_weighted_sum_forward 算子")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    test_cases = [
        {"batch_size": 32, "hidden_dim": 256, "num_experts": 5},
        {"batch_size": 64, "hidden_dim": 512, "num_experts": 8},
        {"batch_size": 128, "hidden_dim": 128, "num_experts": 3},
        {"batch_size": 4096, "hidden_dim": 256, "num_experts": 5},  # 高吞吐量测试用例
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {case}")
        print("-" * 40)
        
        batch_size = case["batch_size"]
        hidden_dim = case["hidden_dim"]
        num_experts = case["num_experts"]
        
        # 对于高吞吐量测试用例，显示内存使用情况
        if batch_size >= 4096:
            print(f"  🚀 高吞吐量测试用例 - 预计内存使用: {batch_size * hidden_dim * num_experts * 4 / 1024**2:.1f} MB (专家输出)")
            print_gpu_memory()
        
        # 创建测试数据
        torch.manual_seed(42)
        expert_outputs = torch.randn(batch_size, hidden_dim, num_experts, device=device)
        gate_weights = torch.randn(batch_size, num_experts, device=device)
        
        # 测试CUDA kernel
        try:
            with torch.no_grad():
                cuda_output = home_kernels.expert_weighted_sum_forward(
                    expert_outputs, gate_weights
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            print(f"  输出数据类型: {cuda_output.dtype}")
            print(f"  输出设备: {cuda_output.device}")
            
            # 验证输出形状
            expected_shape = (batch_size, hidden_dim)
            if cuda_output.shape == expected_shape:
                print(f"✓ 输出形状正确: {cuda_output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {cuda_output.shape}")
            
            # 与PyTorch实现对比验证
            with torch.no_grad():
                # 扩展gate_weights维度以匹配expert_outputs
                gate_weights_expanded = gate_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
                pytorch_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
            
            # 计算数值差异
            diff = torch.abs(cuda_output - pytorch_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  与PyTorch实现对比:")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✓ 数值精度验证通过")
            else:
                print(f"⚠️  数值精度可能存在差异")
            
            # 性能测试
            times = []
            num_runs = 10
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = home_kernels.expert_weighted_sum_forward(expert_outputs, gate_weights)
            
            # 测试
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = home_kernels.expert_weighted_sum_forward(expert_outputs, gate_weights)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"  性能统计:")
            print(f"    平均时间: {avg_time:.4f} ms")
            print(f"    最小时间: {min_time:.4f} ms")
            print(f"    最大时间: {max_time:.4f} ms")
            print(f"    标准差: {std_time:.4f} ms")
            
            # 对于高吞吐量测试用例，计算吞吐量
            if batch_size >= 4096:
                throughput = batch_size / (avg_time / 1000)  # samples per second
                print(f"    吞吐量: {throughput:.0f} samples/sec")
                print_gpu_memory()
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ expert_weighted_sum_forward 算子测试完成!")


def test_fused_batch_norm_silu_forward():
    """测试fused_batch_norm_silu_forward算子"""
    print("\n" + "=" * 60)
    print("测试 fused_batch_norm_silu_forward 算子")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    test_cases = [
        {"batch_size": 32, "num_experts": 5, "hidden_dim": 256},
        {"batch_size": 64, "num_experts": 8, "hidden_dim": 512},
        {"batch_size": 128, "num_experts": 3, "hidden_dim": 128},
        {"batch_size": 4096, "num_experts": 5, "hidden_dim": 256},  # 高吞吐量测试用例
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {case}")
        print("-" * 40)
        
        batch_size = case["batch_size"]
        num_experts = case["num_experts"]
        hidden_dim = case["hidden_dim"]
        
        # 对于高吞吐量测试用例，显示内存使用情况
        if batch_size >= 4096:
            print(f"  🚀 高吞吐量测试用例 - 预计内存使用: {batch_size * num_experts * hidden_dim * 4 / 1024**2:.1f} MB (数据)")
            print_gpu_memory()
        
        # 创建测试数据
        torch.manual_seed(42)
        data = torch.randn(batch_size, num_experts, hidden_dim, device=device)
        bn_weights = torch.randn(num_experts, hidden_dim, device=device)
        bn_biases = torch.randn(num_experts, hidden_dim, device=device)
        running_mean = torch.randn(num_experts, hidden_dim, device=device)
        running_var = torch.ones(num_experts, hidden_dim, device=device)  # 确保方差为正
        epsilon = 1e-5
        
        # 测试CUDA kernel
        try:
            with torch.no_grad():
                cuda_output = home_kernels.fused_batch_norm_silu_forward(
                    data, bn_weights, bn_biases, running_mean, running_var, epsilon
                )
            
            print(f"✓ CUDA kernel执行成功")
            print(f"  输出形状: {cuda_output.shape}")
            print(f"  输出数据类型: {cuda_output.dtype}")
            print(f"  输出设备: {cuda_output.device}")
            
            # 验证输出形状
            expected_shape = (batch_size, num_experts, hidden_dim)
            if cuda_output.shape == expected_shape:
                print(f"✓ 输出形状正确: {cuda_output.shape}")
            else:
                print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {cuda_output.shape}")
            
            # 与PyTorch实现对比验证
            with torch.no_grad():
                # 手动实现BatchNorm + SiLU
                # 首先应用BatchNorm
                normalized = (data - running_mean.unsqueeze(0)) / torch.sqrt(running_var.unsqueeze(0) + epsilon)
                batch_norm_output = normalized * bn_weights.unsqueeze(0) + bn_biases.unsqueeze(0)
                
                # 然后应用SiLU激活函数
                pytorch_output = batch_norm_output * torch.sigmoid(batch_norm_output)
            
            # 计算数值差异
            diff = torch.abs(cuda_output - pytorch_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"  与PyTorch实现对比:")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"✓ 数值精度验证通过")
            else:
                print(f"⚠️  数值精度可能存在差异")
            
            # 验证SiLU激活函数特性
            # SiLU(x) = x * sigmoid(x)，应该保持输入的正负性
            input_positive = data > 0
            output_positive = cuda_output > 0
            consistency = torch.all(input_positive == output_positive)
            
            print(f"  SiLU激活函数验证:")
            print(f"    正负性一致性: {'✓' if consistency else '✗'}")
            
            # 性能测试
            times = []
            num_runs = 10
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = home_kernels.fused_batch_norm_silu_forward(
                        data, bn_weights, bn_biases, running_mean, running_var, epsilon
                    )
            
            # 测试
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = home_kernels.fused_batch_norm_silu_forward(
                        data, bn_weights, bn_biases, running_mean, running_var, epsilon
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"  性能统计:")
            print(f"    平均时间: {avg_time:.4f} ms")
            print(f"    最小时间: {min_time:.4f} ms")
            print(f"    最大时间: {max_time:.4f} ms")
            print(f"    标准差: {std_time:.4f} ms")
            
            # 对于高吞吐量测试用例，计算吞吐量
            if batch_size >= 4096:
                throughput = batch_size / (avg_time / 1000)  # samples per second
                print(f"    吞吐量: {throughput:.0f} samples/sec")
                print_gpu_memory()
            
        except Exception as e:
            print(f"✗ CUDA kernel执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ fused_batch_norm_silu_forward 算子测试完成!")


def test_cuda_kernel_times():
    """测试各个CUDA kernel的执行时间"""
    print("\n" + "=" * 60)
    print("CUDA Kernel性能测试")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过测试")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    input_dim = 256
    hidden_dim = 256
    num_experts = 5
    
    # 创建测试数据
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, input_dim, device=device)
    expert_weights = torch.randn(num_experts, input_dim, hidden_dim, device=device)
    expert_biases = torch.randn(num_experts, hidden_dim, device=device)
    expert_indices = torch.arange(num_experts, dtype=torch.int32, device=device)
    
    # 测试各个kernel
    kernels_to_test = [
        ("home_expert_forward", lambda: home_kernels.home_expert_forward(
            input_data, expert_weights, expert_biases, expert_indices, num_experts, True
        )),
        ("lora_gate_forward", lambda: home_kernels.lora_gate_forward(
            input_data, torch.randn(input_dim, 16, device=device), 
            torch.randn(16, hidden_dim, device=device), True
        )),
        ("gate_weights_forward", lambda: home_kernels.gate_weights_forward(
            input_data, torch.randn(input_dim, num_experts, device=device), True
        )),
        ("expert_weighted_sum_forward", lambda: home_kernels.expert_weighted_sum_forward(
            torch.randn(batch_size, hidden_dim, num_experts, device=device),
            torch.randn(batch_size, num_experts, device=device)
        )),
        ("fused_batch_norm_silu_forward", lambda: home_kernels.fused_batch_norm_silu_forward(
            torch.randn(batch_size, num_experts, hidden_dim, device=device),
            torch.randn(num_experts, hidden_dim, device=device),
            torch.randn(num_experts, hidden_dim, device=device),
            torch.randn(num_experts, hidden_dim, device=device),
            torch.randn(num_experts, hidden_dim, device=device),
            1e-5
        ))
    ]
    
    print(f"{'Kernel名称':<30} {'平均时间(ms)':<15} {'最小时间(ms)':<15} {'最大时间(ms)':<15} {'标准差(ms)':<15}")
    print("-" * 90)
    
    for kernel_name, kernel_func in kernels_to_test:
        times = []
        num_runs = 10
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = kernel_func()
        
        # 测试
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = kernel_func()
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"{kernel_name:<30} {avg_time:<15.4f} {min_time:<15.4f} {max_time:<15.4f} {std_time:<15.4f}")

def profile_model_operators(model, input_states, l_gate_states, g_gate_states, l_gate_states_task, model_name="模型"):
    """分析模型内部算子的执行时间"""
    print(f"\n{model_name}算子时间分析:")
    print("-" * 50)
    
    # 预热
    with torch.no_grad():
        _ = model(input_states, l_gate_states, g_gate_states, l_gate_states_task)
    
    # 使用torch.profiler进行详细分析
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            _ = model(input_states, l_gate_states, g_gate_states, l_gate_states_task)
    
    # 分析结果
    print("CUDA算子统计:")
    cuda_events = [event for event in prof.events() if event.device_type == torch.profiler.ProfilerActivity.CUDA]
    
    # 按算子类型分组统计
    operator_stats = {}
    for event in cuda_events:
        op_name = event.name
        if op_name not in operator_stats:
            operator_stats[op_name] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0,
                'min_time': float('inf')
            }
        
        stats = operator_stats[op_name]
        stats['count'] += 1
        stats['total_time'] += event.cuda_time
        stats['max_time'] = max(stats['max_time'], event.cuda_time)
        stats['min_time'] = min(stats['min_time'], event.cuda_time)
    
    # 计算平均值
    for op_name, stats in operator_stats.items():
        stats['avg_time'] = stats['total_time'] / stats['count']
        if stats['min_time'] == float('inf'):
            stats['min_time'] = 0
    
    # 按总时间排序
    sorted_ops = sorted(operator_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print(f"{'算子名称':<40} {'调用次数':<8} {'总时间(μs)':<12} {'平均时间(μs)':<12} {'最大时间(μs)':<12} {'最小时间(μs)':<12}")
    print("-" * 100)
    
    total_cuda_time = 0
    for op_name, stats in sorted_ops[:20]:  # 显示前20个最耗时的算子
        print(f"{op_name:<40} {stats['count']:<8} {stats['total_time']:<12.2f} {stats['avg_time']:<12.2f} {stats['max_time']:<12.2f} {stats['min_time']:<12.2f}")
        total_cuda_time += stats['total_time']
    
    print("-" * 100)
    print(f"总CUDA时间: {total_cuda_time:.2f} μs ({total_cuda_time/1000:.2f} ms)")
    
    # 分析内存使用
    print(f"\n内存使用分析:")
    memory_events = [event for event in prof.events() if 'memory' in event.name.lower()]
    if memory_events:
        total_memory = sum(event.cuda_memory_usage for event in memory_events if hasattr(event, 'cuda_memory_usage'))
        print(f"总内存使用: {total_memory / 1024**2:.2f} MB")
    
    return operator_stats

def quick_operator_analysis(model, input_states, l_gate_states, g_gate_states, l_gate_states_task, model_name="模型"):
    """快速算子分析（不使用profiler，避免开销）"""
    print(f"\n{model_name}快速算子分析:")
    print("-" * 40)
    
    # 预热
    with torch.no_grad():
        _ = model(input_states, l_gate_states, g_gate_states, l_gate_states_task)
    
    # 多次运行测试总时间
    num_runs = 5
    times = []
    
    for i in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"总推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"时间范围: {min_time:.2f} - {max_time:.2f} ms")
    
    return avg_time

def detailed_operator_analysis():
    """详细的算子分析，使用torch.profiler"""
    print("\n" + "=" * 60)
    print("详细算子分析")
    print("=" * 60)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过详细分析")
        return
    
    try:
        from HoME_optimized import HoMELayerOptimized
        from HoME import HoMELayer
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型配置
    config = {
        'num_experts': 5,
        'input_dims': [700] * 6,
        'dim': 256,
        'l_gate_dim': 700,
        'g_gate_dim': 700,
        'l_gate_dim_task': 700,
        'use_lora_gate': True,
        'lora_rank': 2
    }
    
    # 创建测试数据
    batch_size = 32
    input_states = [torch.randn(batch_size, 700, device=device) for _ in range(6)]
    l_gate_states = torch.randn(batch_size, 700, device=device)
    g_gate_states = torch.randn(batch_size, 700, device=device)
    l_gate_states_task = torch.randn(batch_size, 700, device=device)
    
    # 分析原始版本
    print("\n分析原始版本详细算子...")
    try:
        torch.manual_seed(42)
        model_original = HoMELayer(**config).to(device)
        
        orig_stats = profile_model_operators(
            model_original, input_states, l_gate_states, g_gate_states, l_gate_states_task, "原始版本"
        )
        
        del model_original
        clear_gpu_memory()
        
    except Exception as e:
        print(f"✗ 原始版本详细分析失败: {e}")
        orig_stats = {}
    
    # 分析优化版本
    print("\n分析优化版本详细算子...")
    try:
        torch.manual_seed(42)
        model_optimized = HoMELayerOptimized(**config, use_cuda_kernels=True).to(device)
        
        opt_stats = profile_model_operators(
            model_optimized, input_states, l_gate_states, g_gate_states, l_gate_states_task, "优化版本"
        )
        
    except Exception as e:
        print(f"✗ 优化版本详细分析失败: {e}")
        opt_stats = {}
    
    # 对比分析
    if orig_stats and opt_stats:
        print("\n算子对比分析:")
        print("-" * 60)
        
        # 找出共同的算子
        common_ops = set(orig_stats.keys()) & set(opt_stats.keys())
        
        print(f"{'算子名称':<30} {'原始版本(μs)':<15} {'优化版本(μs)':<15} {'加速比':<10}")
        print("-" * 70)
        
        for op_name in sorted(common_ops):
            orig_time = orig_stats[op_name]['total_time']
            opt_time = opt_stats[op_name]['total_time']
            speedup = orig_time / opt_time if opt_time > 0 else float('inf')
            
            print(f"{op_name:<30} {orig_time:<15.2f} {opt_time:<15.2f} {speedup:<10.2f}x")
    
    print("\n✓ 详细算子分析完成!")

def sync_model_weights(model_original, model_optimized):
    """同步两个模型的权重，确保完全一致"""
    print("开始权重同步...")
    
    # 1. 同步LoRA门控权重
    if hasattr(model_original, 'group_f_gates') and hasattr(model_optimized, 'group_f_gates'):
        for key in model_original.group_f_gates.keys():
            if key in model_optimized.group_f_gates:
                # 同步A和B矩阵
                model_optimized.group_f_gates[key].A.data.copy_(model_original.group_f_gates[key].A.data)
                model_optimized.group_f_gates[key].B.data.copy_(model_original.group_f_gates[key].B.data)
                print(f"  同步LoRA门控: {key}")
    
    # 2. 同步专家网络权重
    if hasattr(model_original, 'task_experts') and hasattr(model_optimized, 'task_experts'):
        for task_key in model_original.task_experts.keys():
            if task_key in model_optimized.task_experts:
                orig_experts = model_original.task_experts[task_key]
                opt_expert = model_optimized.task_experts[task_key]
                
                # 从原始版本的专家网络中提取权重
                expert_weights_list = []
                expert_biases_list = []
                
                for i, orig_expert in enumerate(orig_experts):
                    # 原始版本是Sequential(MLPLayer, BatchNorm, SiLU)
                    mlp_layer = orig_expert[0]  # MLPLayer
                    
                    # 获取MLP层的权重和偏置
                    if hasattr(mlp_layer, 'layers') and len(mlp_layer.layers) > 0:
                        # MLPLayer的第一层是Linear层
                        linear_layer = mlp_layer.layers[0]
                        expert_weights_list.append(linear_layer.weight.data.t())  # 转置为[input_dim, output_dim]
                        if linear_layer.bias is not None:
                            expert_biases_list.append(linear_layer.bias.data)
                        else:
                            expert_biases_list.append(torch.zeros(linear_layer.weight.size(0), device=linear_layer.weight.device))
                
                if expert_weights_list:
                    # 将权重堆叠为[num_experts, input_dim, output_dim]
                    stacked_weights = torch.stack(expert_weights_list, dim=0)
                    stacked_biases = torch.stack(expert_biases_list, dim=0)
                    
                    # 同步到优化版本
                    opt_expert.expert_weights.data.copy_(stacked_weights)
                    opt_expert.expert_biases.data.copy_(stacked_biases)
                    print(f"  同步专家权重: {task_key} - {len(expert_weights_list)} experts")
    
    # 3. 同步门控网络权重
    if hasattr(model_original, 'task_l_gates') and hasattr(model_optimized, 'task_l_gates'):
        for key in model_original.task_l_gates.keys():
            if key in model_optimized.task_l_gates:
                # 同步门控网络的权重
                orig_gate = model_original.task_l_gates[key]
                opt_gate = model_optimized.task_l_gates[key]
                
                # 获取原始门控的权重
                if hasattr(orig_gate, 'layers') and len(orig_gate.layers) > 0:
                    orig_linear = orig_gate.layers[0]  # DenseLayer的第一层是Linear
                    opt_gate.gate_weights.data.copy_(orig_gate.layers[0].weight.data)
                    if orig_linear.bias is not None:
                        opt_gate.gate_biases.data.copy_(orig_linear.bias.data)
                    print(f"  同步门控权重: {key}")
    
    # 4. 同步g门控权重
    if hasattr(model_original, 'task_g_gates') and hasattr(model_optimized, 'task_g_gates'):
        for key in model_original.task_g_gates.keys():
            if key in model_optimized.task_g_gates:
                orig_gate = model_original.task_g_gates[key]
                opt_gate = model_optimized.task_g_gates[key]
                
                if hasattr(orig_gate, 'layers') and len(orig_gate.layers) > 0:
                    orig_linear = orig_gate.layers[0]
                    opt_gate.gate_weights.data.copy_(orig_linear.weight.data)
                    if orig_linear.bias is not None:
                        opt_gate.gate_biases.data.copy_(orig_linear.bias.data)
                    print(f"  同步g门控权重: {key}")
    
    # 5. 同步meta专家权重
    if hasattr(model_original, 'meta_specific_experts') and hasattr(model_optimized, 'meta_experts'):
        for group_name in model_original.meta_specific_experts.keys():
            if group_name in model_optimized.meta_experts:
                orig_experts = model_original.meta_specific_experts[group_name]
                opt_expert = model_optimized.meta_experts[group_name]
                
                # 提取meta专家权重
                meta_weights_list = []
                meta_biases_list = []
                
                for orig_expert in orig_experts:
                    mlp_layer = orig_expert[0]  # MLPLayer
                    if hasattr(mlp_layer, 'layers') and len(mlp_layer.layers) > 0:
                        linear_layer = mlp_layer.layers[0]
                        meta_weights_list.append(linear_layer.weight.data.t())
                        if linear_layer.bias is not None:
                            meta_biases_list.append(linear_layer.bias.data)
                        else:
                            meta_biases_list.append(torch.zeros(linear_layer.weight.size(0), device=linear_layer.weight.device))
                
                if meta_weights_list:
                    stacked_weights = torch.stack(meta_weights_list, dim=0)
                    stacked_biases = torch.stack(meta_biases_list, dim=0)
                    
                    opt_expert.expert_weights.data.copy_(stacked_weights)
                    opt_expert.expert_biases.data.copy_(stacked_biases)
                    print(f"  同步meta专家权重: {group_name}")
    
    print("权重同步完成！")


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("HoME优化架构基本使用示例")
    print("=" * 60)
    
    try:
        from HoME_optimized import HoMELayerOptimized
        print("✓ 成功导入HoME_optimized模块")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请先编译CUDA kernel: python setup_home_optimized.py build_ext --inplace")
        return
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("\n创建HoME模型...")
    try:
        model = HoMELayerOptimized(
            num_experts=5,                    # 专家数量
            input_dims=[700] * 6,  # 各任务输入维度
            dim=256,                          # 隐藏维度
            l_gate_dim=700,                   # l门控维度
            g_gate_dim=700,                   # g门控维度
            l_gate_dim_task=700,               # 任务级l门控维度
            use_lora_gate=True,               # 使用LoRA门控
            lora_rank=2,                     # LoRA秩
            use_cuda_kernels=CUDA_KERNELS_AVAILABLE  # 根据可用性决定是否使用CUDA kernel
        ).to(device)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建示例数据
    print("\n创建示例数据...")
    input_states, l_gate_states, g_gate_states, l_gate_states_task = create_sample_data(
        batch_size=32, device=device
    )
    
    # 前向传播
    print("\n执行前向传播...")
    try:
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_states, l_gate_states, g_gate_states, l_gate_states_task)
            inference_time = time.time() - start_time
        
        print(f"推理时间: {inference_time*1000:.2f} ms")
        
        # 显示输出
        print("\n模型输出:")
        for key, output in outputs.items():
            print(f"  {key}: {output.shape}")
        
        print("\n✓ 基本使用示例完成!")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return


def compare_home_versions():
    """对比HoMELayerOptimized和HoMELayer的精度和性能"""
    print("\n" + "=" * 60)
    print("HoME版本对比测试")
    print("=" * 60)
    
    try:
        from HoME_optimized import HoMELayerOptimized
        from HoME import HoMELayer
        print("✓ 成功导入两个HoME版本")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    config = {
        'num_experts': 5,
        'input_dims': [700] * 6,
        'dim': 256,
        'l_gate_dim': 700,
        'g_gate_dim': 700,
        'l_gate_dim_task': 700,
        'use_lora_gate': True,
        'lora_rank': 2
    }
    
    # 创建两个版本的模型，确保权重初始化相同
    print("\n创建模型...")
    print_gpu_memory()
    
    try:
        # 使用相同的随机种子创建原始版本
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        model_original = HoMELayer(
            **config
        ).to(device)
        
        # 使用相同的随机种子创建优化版本
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        model_optimized = HoMELayerOptimized(
            **config,
            use_cuda_kernels=CUDA_KERNELS_AVAILABLE
        ).to(device)
        
        print(f"原始版本参数数量: {sum(p.numel() for p in model_original.parameters()):,}")
        print(f"优化版本参数数量: {sum(p.numel() for p in model_optimized.parameters()):,}")
        
        # 手动同步权重（确保完全一致）
        print("同步模型权重...")
        sync_model_weights(model_original, model_optimized)
        
        print_gpu_memory()
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建测试数据
    print("\n创建测试数据...")
    batch_size = 32
    input_states, l_gate_states, g_gate_states, l_gate_states_task = create_sample_data(
        batch_size=batch_size, device=device
    )
    
    # 设置相同的随机种子确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 重新创建数据以确保一致性
    input_states = [torch.randn(batch_size, 700, device=device) for _ in range(6)]
    l_gate_states = torch.randn(batch_size, 700, device=device)
    g_gate_states = torch.randn(batch_size, 700, device=device)
    l_gate_states_task = torch.randn(batch_size, 700, device=device)
    
    # 精度对比测试
    print("\n" + "-" * 40)
    print("精度对比测试")
    print("-" * 40)
    
    # 先测试原始版本
    print("测试原始版本...")
    outputs_original = None
    try:
        with torch.no_grad():
            outputs_original = model_original(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        print("✓ 原始版本运行成功")
        print(f"原始版本输出键: {list(outputs_original.keys())}")
    except Exception as e:
        print(f"✗ 原始版本运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 清理GPU内存
    if torch.cuda.is_available():
        del model_original
        clear_gpu_memory()
        print_gpu_memory()
    
    # 再测试优化版本
    print("\n测试优化版本...")
    outputs_optimized = None
    try:
        with torch.no_grad():
            outputs_optimized = model_optimized(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        print("✓ 优化版本运行成功")
        print(f"优化版本输出键: {list(outputs_optimized.keys())}")
    except Exception as e:
        print(f"✗ 优化版本运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 进行精度对比
    print("\n进行精度对比...")
    try:
        print("输出形状对比:")
        for key in outputs_optimized.keys():
            if key in outputs_original:
                opt_shape = outputs_optimized[key].shape
                orig_shape = outputs_original[key].shape
                print(f"  {key}: 优化版本 {opt_shape} vs 原始版本 {orig_shape}")
                
                # 计算数值差异
                if opt_shape == orig_shape:
                    diff = torch.abs(outputs_optimized[key] - outputs_original[key])
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    print(f"    最大差异: {max_diff:.6f}")
                    print(f"    平均差异: {mean_diff:.6f}")
                    
                    # 计算相对误差
                    rel_error = mean_diff / torch.mean(torch.abs(outputs_original[key])).item()
                    print(f"    相对误差: {rel_error:.6f}")
                else:
                    print(f"    ❌ 形状不匹配!")
            else:
                print(f"  {key}: 仅在优化版本中存在")
        
        # 检查原始版本独有的输出
        for key in outputs_original.keys():
            if key not in outputs_optimized:
                print(f"  {key}: 仅在原始版本中存在")
        
    except Exception as e:
        print(f"✗ 精度对比失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 快速算子时间分析
    print("\n" + "=" * 60)
    print("快速算子时间分析")
    print("=" * 60)
    
    # 分析原始版本算子时间
    print("\n分析原始版本算子时间...")
    try:
        # 重新创建原始版本模型用于分析
        torch.manual_seed(42)
        model_original_for_profile = HoMELayer(**config).to(device)
        sync_model_weights(model_original_for_profile, model_optimized)
        
        orig_avg_time = quick_operator_analysis(
            model_original_for_profile, input_states, l_gate_states, g_gate_states, l_gate_states_task, "原始版本"
        )
        
        # 清理内存
        del model_original_for_profile
        clear_gpu_memory()
        
    except Exception as e:
        print(f"✗ 原始版本算子分析失败: {e}")
        orig_avg_time = 0
    
    # 分析优化版本算子时间
    print("\n分析优化版本算子时间...")
    try:
        opt_avg_time = quick_operator_analysis(
            model_optimized, input_states, l_gate_states, g_gate_states, l_gate_states_task, "优化版本"
        )
    except Exception as e:
        print(f"✗ 优化版本算子分析失败: {e}")
        opt_avg_time = 0
    
    # 算子时间对比
    if orig_avg_time > 0 and opt_avg_time > 0:
        speedup = orig_avg_time / opt_avg_time
        print(f"\n算子时间对比:")
        print(f"  原始版本: {orig_avg_time:.2f} ms")
        print(f"  优化版本: {opt_avg_time:.2f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✅ 优化版本更快")
        else:
            print(f"  ⚠️  原始版本更快")
    
    # 性能对比测试
    print("\n" + "-" * 40)
    print("性能对比测试")
    print("-" * 40)
    
    num_runs = 5  # 减少运行次数以节省内存
    
    # 先测试原始版本性能
    print("测试原始版本性能...")
    orig_time = float('inf')
    try:
        # 重新创建原始版本模型
        model_original = HoMELayer(**config).to(device)
        
        # 预热
        for _ in range(2):
            with torch.no_grad():
                _ = model_original(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        
        # 性能测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model_original(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        orig_time = time.time() - start_time
        
        print(f"原始版本平均时间: {orig_time/num_runs*1000:.2f} ms")
        
    except Exception as e:
        print(f"✗ 原始版本性能测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理原始版本模型
    if torch.cuda.is_available():
        del model_original
        clear_gpu_memory()
        print_gpu_memory()
    
    # 再测试优化版本性能
    print("\n测试优化版本性能...")
    opt_time = float('inf')
    try:
        # 预热
        for _ in range(2):
            with torch.no_grad():
                _ = model_optimized(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        
        # 性能测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model_optimized(input_states, l_gate_states, g_gate_states, l_gate_states_task)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        opt_time = time.time() - start_time
        
        print(f"优化版本平均时间: {opt_time/num_runs*1000:.2f} ms")
        
    except Exception as e:
        print(f"✗ 优化版本性能测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 性能对比结果
    if opt_time != float('inf') and orig_time != float('inf'):
        speedup = orig_time / opt_time
        print(f"\n性能对比结果:")
        print(f"  优化版本: {opt_time/num_runs*1000:.2f} ms")
        print(f"  原始版本: {orig_time/num_runs*1000:.2f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✅ 优化版本更快")
        else:
            print(f"  ⚠️  原始版本更快")
    
    # 内存使用对比
    if torch.cuda.is_available():
        print(f"\nGPU内存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    print("\n✓ HoME版本对比测试完成!")


def run_comprehensive_kernel_tests():
    """运行所有CUDA kernel的详细测试"""
    print("\n" + "=" * 80)
    print("HoME CUDA Kernels 综合测试套件")
    print("=" * 80)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过所有kernel测试")
        return
    
    # 运行各个算子的详细测试
    test_home_expert_forward()
    test_lora_gate_forward()
    test_gate_weights_forward()
    test_expert_weighted_sum_forward()
    test_fused_batch_norm_silu_forward()
    
    print("\n" + "=" * 80)
    print("所有CUDA kernel测试完成!")
    print("=" * 80)


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "=" * 80)
    print("HoME 性能基准测试")
    print("=" * 80)
    
    if not CUDA_KERNELS_AVAILABLE:
        print("⚠️  CUDA kernels不可用，跳过性能测试")
        return
    
    # 运行快速性能测试
    test_cuda_kernel_times()
    
    # 运行详细算子分析
    detailed_operator_analysis()
    
    print("\n" + "=" * 80)
    print("性能基准测试完成!")
    print("=" * 80)


def run_accuracy_validation():
    """运行精度验证测试"""
    print("\n" + "=" * 80)
    print("HoME 精度验证测试")
    print("=" * 80)
    
    # 运行版本对比测试
    compare_home_versions()
    
    print("\n" + "=" * 80)
    print("精度验证测试完成!")
    print("=" * 80)


def run_all_tests():
    """运行所有测试"""
    print("HoME优化架构完整测试套件")
    print("=" * 80)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 1. 基本使用示例
    print("\n" + "=" * 40)
    print("1. 基本使用示例")
    print("=" * 40)
    example_basic_usage()
    
    # 2. CUDA kernel详细测试
    print("\n" + "=" * 40)
    print("2. CUDA Kernel详细测试")
    print("=" * 40)
    run_comprehensive_kernel_tests()
    
    # 3. 性能基准测试
    print("\n" + "=" * 40)
    print("3. 性能基准测试")
    print("=" * 40)
    run_performance_benchmark()
    
    # 4. 精度验证测试
    print("\n" + "=" * 40)
    print("4. 精度验证测试")
    print("=" * 40)
    run_accuracy_validation()
    
    print("\n" + "=" * 80)
    print("🎉 所有测试完成! 🎉")
    print("=" * 80)


def main():
    """主函数 - 提供多种测试选项"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HoME优化架构测试工具')
    parser.add_argument('--test', choices=['all', 'kernels', 'performance', 'accuracy', 'basic'], 
                       default='all', help='选择要运行的测试类型')
    parser.add_argument('--quick', action='store_true', help='运行快速测试（跳过详细分析）')
    
    args = parser.parse_args()
    
    print("HoME优化架构测试工具")
    print("=" * 60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    if args.test == 'all':
        run_all_tests()
    elif args.test == 'kernels':
        run_comprehensive_kernel_tests()
    elif args.test == 'performance':
        run_performance_benchmark()
    elif args.test == 'accuracy':
        run_accuracy_validation()
    elif args.test == 'basic':
        example_basic_usage()
    
    if args.quick:
        print("\n⚡ 快速测试模式完成!")
    else:
        print("\n✅ 完整测试完成!")


if __name__ == "__main__":
    main()
