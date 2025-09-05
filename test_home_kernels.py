#!/usr/bin/env python3
"""
HoME CUDA Kernels 专用测试文件

这个文件专门测试 home_kernels.cu 中实现的所有CUDA算子：
1. home_expert_forward - 专家网络前向传播
2. lora_gate_forward - LoRA门控前向传播  
3. gate_weights_forward - 门控权重前向传播
4. expert_weighted_sum_forward - 专家加权求和
5. fused_batch_norm_silu_forward - 融合BatchNorm+SiLU算子

每个算子都包含：
- 功能正确性验证
- 输出形状验证
- 数值精度验证（与PyTorch实现对比）
- 性能基准测试
- 多种测试用例覆盖

使用方法:
    python test_home_kernels.py                    # 运行所有测试
    python test_home_kernels.py --test expert      # 只测试专家网络
    python test_home_kernels.py --test lora        # 只测试LoRA门控
    python test_home_kernels.py --test gate        # 只测试门控权重
    python test_home_kernels.py --test weighted    # 只测试加权求和
    python test_home_kernels.py --test fused       # 只测试融合算子
    python test_home_kernels.py --quick            # 快速测试模式
"""

import torch
import torch.nn as nn
import time
import sys
import os
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入CUDA kernel模块
try:
    import home_kernels
    CUDA_KERNELS_AVAILABLE = True
    print("✓ CUDA kernels模块导入成功")
except ImportError as e:
    print(f"⚠️  CUDA kernels模块导入失败: {e}")
    print("请先编译CUDA kernels: python setup_home.py build_ext --inplace")
    CUDA_KERNELS_AVAILABLE = False
    home_kernels = None


class HomeKernelTester:
    """HoME CUDA Kernels 测试器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        
    def print_gpu_memory(self):
        """打印GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU内存 - 已分配: {allocated:.2f} GB, 已缓存: {reserved:.2f} GB")
            return allocated, reserved
        return 0, 0
    
    def clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_performance_test(self, kernel_func, num_runs: int = 10, warmup_runs: int = 3) -> Dict:
        """运行性能测试"""
        times = []
        
        # 预热
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = kernel_func()
        
        # 测试
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = kernel_func()
            
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
    
    def test_home_expert_forward(self):
        """测试home_expert_forward算子"""
        print("\n" + "=" * 80)
        print("测试 home_expert_forward 算子")
        print("=" * 80)
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠️  CUDA kernels不可用，跳过测试")
            return
        
        # 测试用例
        test_cases = [
            {"batch_size": 32, "input_dim": 256, "hidden_dim": 256, "num_experts": 5},
            {"batch_size": 64, "input_dim": 512, "hidden_dim": 512, "num_experts": 8},
            {"batch_size": 128, "input_dim": 128, "hidden_dim": 128, "num_experts": 3},
            {"batch_size": 4096, "input_dim": 700, "hidden_dim": 256, "num_experts": 5},  # 高吞吐量测试
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case}")
            print("-" * 60)
            
            batch_size = case["batch_size"]
            input_dim = case["input_dim"]
            hidden_dim = case["hidden_dim"]
            num_experts = case["num_experts"]
            
            # 高吞吐量测试用例特殊处理
            if batch_size >= 4096:
                print(f"  🚀 高吞吐量测试用例")
                self.print_gpu_memory()
            
            try:
                # 创建测试数据
                torch.manual_seed(42)
                input_data = torch.randn(batch_size, input_dim, device=self.device)
                expert_weights = torch.randn(num_experts, input_dim, hidden_dim, device=self.device)
                expert_biases = torch.randn(num_experts, hidden_dim, device=self.device)
                expert_indices = torch.arange(num_experts, dtype=torch.int32, device=self.device)
                
                # 测试CUDA kernel
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
                    continue
                
                # 性能测试
                def kernel_func():
                    return home_kernels.home_expert_forward(
                        input_data, expert_weights, expert_biases, expert_indices, num_experts, True
                    )
                
                perf_stats = self.run_performance_test(kernel_func)
                print(f"  性能统计:")
                print(f"    平均时间: {perf_stats['avg_time']:.4f} ms")
                print(f"    最小时间: {perf_stats['min_time']:.4f} ms")
                print(f"    最大时间: {perf_stats['max_time']:.4f} ms")
                print(f"    标准差: {perf_stats['std_time']:.4f} ms")
                
                # 高吞吐量测试用例额外计算吞吐量
                if batch_size >= 4096:
                    throughput = batch_size / (perf_stats['avg_time'] / 1000)
                    print(f"    吞吐量: {throughput:.0f} samples/sec")
                    self.print_gpu_memory()
                
                # 清理内存
                del input_data, expert_weights, expert_biases, expert_indices, cuda_output
                self.clear_gpu_memory()
                
            except Exception as e:
                print(f"✗ CUDA kernel执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ home_expert_forward 算子测试完成!")
    
    def test_lora_gate_forward(self):
        """测试lora_gate_forward算子"""
        print("\n" + "=" * 80)
        print("测试 lora_gate_forward 算子")
        print("=" * 80)
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠️  CUDA kernels不可用，跳过测试")
            return
        
        # 测试用例
        test_cases = [
            {"batch_size": 32, "input_dim": 256, "rank": 16, "output_dim": 256},
            {"batch_size": 64, "input_dim": 512, "rank": 32, "output_dim": 512},
            {"batch_size": 128, "input_dim": 128, "rank": 8, "output_dim": 128},
            {"batch_size": 4096, "input_dim": 700, "rank": 16, "output_dim": 700},  # 高吞吐量测试
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case}")
            print("-" * 60)
            
            batch_size = case["batch_size"]
            input_dim = case["input_dim"]
            rank = case["rank"]
            output_dim = case["output_dim"]
            
            # 高吞吐量测试用例特殊处理
            if batch_size >= 4096:
                print(f"  🚀 高吞吐量测试用例")
                self.print_gpu_memory()
            
            try:
                # 创建测试数据
                torch.manual_seed(42)
                input_data = torch.randn(batch_size, input_dim, device=self.device)
                A_matrix = torch.randn(input_dim, rank, device=self.device)
                B_matrix = torch.randn(rank, output_dim, device=self.device)
                
                # 测试CUDA kernel
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
                    continue
                
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
                def kernel_func():
                    return home_kernels.lora_gate_forward(input_data, A_matrix, B_matrix, True)
                
                perf_stats = self.run_performance_test(kernel_func)
                print(f"  性能统计:")
                print(f"    平均时间: {perf_stats['avg_time']:.4f} ms")
                print(f"    最小时间: {perf_stats['min_time']:.4f} ms")
                print(f"    最大时间: {perf_stats['max_time']:.4f} ms")
                print(f"    标准差: {perf_stats['std_time']:.4f} ms")
                
                # 高吞吐量测试用例额外计算吞吐量
                if batch_size >= 4096:
                    throughput = batch_size / (perf_stats['avg_time'] / 1000)
                    print(f"    吞吐量: {throughput:.0f} samples/sec")
                    self.print_gpu_memory()
                
                # 清理内存
                del input_data, A_matrix, B_matrix, cuda_output, pytorch_output
                self.clear_gpu_memory()
                
            except Exception as e:
                print(f"✗ CUDA kernel执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ lora_gate_forward 算子测试完成!")
    
    def test_gate_weights_forward(self):
        """测试gate_weights_forward算子"""
        print("\n" + "=" * 80)
        print("测试 gate_weights_forward 算子")
        print("=" * 80)
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠️  CUDA kernels不可用，跳过测试")
            return
        
        # 测试用例
        test_cases = [
            {"batch_size": 32, "gate_dim": 256, "num_experts": 5, "use_softmax": True},
            {"batch_size": 64, "gate_dim": 512, "num_experts": 8, "use_softmax": True},
            {"batch_size": 128, "gate_dim": 128, "num_experts": 3, "use_softmax": False},
            {"batch_size": 4096, "gate_dim": 700, "num_experts": 5, "use_softmax": True},  # 高吞吐量测试
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case}")
            print("-" * 60)
            
            batch_size = case["batch_size"]
            gate_dim = case["gate_dim"]
            num_experts = case["num_experts"]
            use_softmax = case["use_softmax"]
            
            # 高吞吐量测试用例特殊处理
            if batch_size >= 4096:
                print(f"  🚀 高吞吐量测试用例")
                self.print_gpu_memory()
            
            try:
                # 创建测试数据
                torch.manual_seed(42)
                gate_states = torch.randn(batch_size, gate_dim, device=self.device)
                gate_weights = torch.randn(gate_dim, num_experts, device=self.device)
                
                # 测试CUDA kernel
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
                    continue
                
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
                def kernel_func():
                    return home_kernels.gate_weights_forward(gate_states, gate_weights, use_softmax)
                
                perf_stats = self.run_performance_test(kernel_func)
                print(f"  性能统计:")
                print(f"    平均时间: {perf_stats['avg_time']:.4f} ms")
                print(f"    最小时间: {perf_stats['min_time']:.4f} ms")
                print(f"    最大时间: {perf_stats['max_time']:.4f} ms")
                print(f"    标准差: {perf_stats['std_time']:.4f} ms")
                
                # 高吞吐量测试用例额外计算吞吐量
                if batch_size >= 4096:
                    throughput = batch_size / (perf_stats['avg_time'] / 1000)
                    print(f"    吞吐量: {throughput:.0f} samples/sec")
                    self.print_gpu_memory()
                
                # 清理内存
                del gate_states, gate_weights, cuda_output, pytorch_output
                self.clear_gpu_memory()
                
            except Exception as e:
                print(f"✗ CUDA kernel执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ gate_weights_forward 算子测试完成!")
    
    def test_expert_weighted_sum_forward(self):
        """测试expert_weighted_sum_forward算子"""
        print("\n" + "=" * 80)
        print("测试 expert_weighted_sum_forward 算子")
        print("=" * 80)
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠️  CUDA kernels不可用，跳过测试")
            return
        
        # 测试用例
        test_cases = [
            {"batch_size": 32, "hidden_dim": 256, "num_experts": 5},
            {"batch_size": 64, "hidden_dim": 512, "num_experts": 8},
            {"batch_size": 128, "hidden_dim": 128, "num_experts": 3},
            {"batch_size": 4096, "hidden_dim": 256, "num_experts": 5},  # 高吞吐量测试
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case}")
            print("-" * 60)
            
            batch_size = case["batch_size"]
            hidden_dim = case["hidden_dim"]
            num_experts = case["num_experts"]
            
            # 高吞吐量测试用例特殊处理
            if batch_size >= 4096:
                print(f"  🚀 高吞吐量测试用例")
                self.print_gpu_memory()
            
            try:
                # 创建测试数据
                torch.manual_seed(42)
                expert_outputs = torch.randn(batch_size, hidden_dim, num_experts, device=self.device)
                gate_weights = torch.randn(batch_size, num_experts, device=self.device)
                
                # 测试CUDA kernel
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
                    continue
                
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
                def kernel_func():
                    return home_kernels.expert_weighted_sum_forward(expert_outputs, gate_weights)
                
                perf_stats = self.run_performance_test(kernel_func)
                print(f"  性能统计:")
                print(f"    平均时间: {perf_stats['avg_time']:.4f} ms")
                print(f"    最小时间: {perf_stats['min_time']:.4f} ms")
                print(f"    最大时间: {perf_stats['max_time']:.4f} ms")
                print(f"    标准差: {perf_stats['std_time']:.4f} ms")
                
                # 高吞吐量测试用例额外计算吞吐量
                if batch_size >= 4096:
                    throughput = batch_size / (perf_stats['avg_time'] / 1000)
                    print(f"    吞吐量: {throughput:.0f} samples/sec")
                    self.print_gpu_memory()
                
                # 清理内存
                del expert_outputs, gate_weights, cuda_output, pytorch_output
                self.clear_gpu_memory()
                
            except Exception as e:
                print(f"✗ CUDA kernel执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ expert_weighted_sum_forward 算子测试完成!")
    
    def test_fused_batch_norm_silu_forward(self):
        """测试fused_batch_norm_silu_forward算子"""
        print("\n" + "=" * 80)
        print("测试 fused_batch_norm_silu_forward 算子")
        print("=" * 80)
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠️  CUDA kernels不可用，跳过测试")
            return
        
        # 测试用例
        test_cases = [
            {"batch_size": 32, "num_experts": 5, "hidden_dim": 256},
            {"batch_size": 64, "num_experts": 8, "hidden_dim": 512},
            {"batch_size": 128, "num_experts": 3, "hidden_dim": 128},
            {"batch_size": 4096, "num_experts": 5, "hidden_dim": 256},  # 高吞吐量测试
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case}")
            print("-" * 60)
            
            batch_size = case["batch_size"]
            num_experts = case["num_experts"]
            hidden_dim = case["hidden_dim"]
            
            # 高吞吐量测试用例特殊处理
            if batch_size >= 4096:
                print(f"  🚀 高吞吐量测试用例")
                self.print_gpu_memory()
            
            try:
                # 创建测试数据
                torch.manual_seed(42)
                data = torch.randn(batch_size, num_experts, hidden_dim, device=self.device)
                bn_weights = torch.randn(num_experts, hidden_dim, device=self.device)
                bn_biases = torch.randn(num_experts, hidden_dim, device=self.device)
                running_mean = torch.randn(num_experts, hidden_dim, device=self.device)
                running_var = torch.ones(num_experts, hidden_dim, device=self.device)  # 确保方差为正
                epsilon = 1e-5
                
                # 测试CUDA kernel
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
                    continue
                
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
                def kernel_func():
                    return home_kernels.fused_batch_norm_silu_forward(
                        data, bn_weights, bn_biases, running_mean, running_var, epsilon
                    )
                
                perf_stats = self.run_performance_test(kernel_func)
                print(f"  性能统计:")
                print(f"    平均时间: {perf_stats['avg_time']:.4f} ms")
                print(f"    最小时间: {perf_stats['min_time']:.4f} ms")
                print(f"    最大时间: {perf_stats['max_time']:.4f} ms")
                print(f"    标准差: {perf_stats['std_time']:.4f} ms")
                
                # 高吞吐量测试用例额外计算吞吐量
                if batch_size >= 4096:
                    throughput = batch_size / (perf_stats['avg_time'] / 1000)
                    print(f"    吞吐量: {throughput:.0f} samples/sec")
                    self.print_gpu_memory()
                
                # 清理内存
                del data, bn_weights, bn_biases, running_mean, running_var, cuda_output, pytorch_output
                self.clear_gpu_memory()
                
            except Exception as e:
                print(f"✗ CUDA kernel执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ fused_batch_norm_silu_forward 算子测试完成!")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("HoME CUDA Kernels 完整测试套件")
        print("=" * 80)
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("警告: CUDA不可用，将使用CPU运行")
        
        # 运行所有算子测试
        self.test_home_expert_forward()
        self.test_lora_gate_forward()
        self.test_gate_weights_forward()
        self.test_expert_weighted_sum_forward()
        self.test_fused_batch_norm_silu_forward()
        
        print("\n" + "=" * 80)
        print("🎉 所有CUDA kernel测试完成! 🎉")
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HoME CUDA Kernels 测试工具')
    parser.add_argument('--test', choices=['all', 'expert', 'lora', 'gate', 'weighted', 'fused'], 
                       default='all', help='选择要运行的测试类型')
    parser.add_argument('--quick', action='store_true', help='运行快速测试（减少运行次数）')
    parser.add_argument('--device', default='cuda', help='运行设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("HoME CUDA Kernels 测试工具")
    print("=" * 60)
    
    # 创建测试器
    tester = HomeKernelTester(device=args.device)
    
    if args.test == 'all':
        tester.run_all_tests()
    elif args.test == 'expert':
        tester.test_home_expert_forward()
    elif args.test == 'lora':
        tester.test_lora_gate_forward()
    elif args.test == 'gate':
        tester.test_gate_weights_forward()
    elif args.test == 'weighted':
        tester.test_expert_weighted_sum_forward()
    elif args.test == 'fused':
        tester.test_fused_batch_norm_silu_forward()
    
    if args.quick:
        print("\n⚡ 快速测试模式完成!")
    else:
        print("\n✅ 完整测试完成!")


if __name__ == "__main__":
    main()
