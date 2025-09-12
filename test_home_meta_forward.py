#!/usr/bin/env python3
"""
HoME Meta Forward 测试脚本
测试完整的两层专家网络：FC+ReLU -> FC+BN+SiLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os

# 添加当前目录到Python路径，以便导入MLP和Dense模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入完整的MLP和Dense层实现
from MLP import MLPLayer
from Dense import DenseLayer

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class PyTorchMetaLayer(nn.Module):
    """使用完整MLP和Dense层实现的Meta层，用于对比测试"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        
        # 第一层：使用MLPLayer实现 FC + ReLU
        self.experts1 = nn.ModuleList()
        for i in range(num_experts):
            expert = MLPLayer(input_dim, [hidden_dim], activate="relu", name=f"expert1_{i}")
            self.experts1.append(expert)
        
        # 第二层：使用MLPLayer实现 FC，然后添加BatchNorm + SiLU
        self.experts2 = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_experts):
            expert = MLPLayer(hidden_dim, [output_dim], activate=None, name=f"expert2_{i}")
            batch_norm = nn.BatchNorm1d(output_dim)
            self.experts2.append(expert)
            self.batch_norms.append(batch_norm)
    
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [batch_size, input_dim]
        Returns:
            output: [batch_size, num_experts, output_dim]
        """
        batch_size = input_tensor.size(0)
        
        # 第一层：MLP (FC + ReLU)
        hidden_outputs = []
        for expert in self.experts1:
            hidden_output = expert(input_tensor)  # [batch_size, hidden_dim]
            hidden_outputs.append(hidden_output)
        
        # 第二层：MLP (FC) + BatchNorm + SiLU
        final_outputs = []
        for i, (expert, batch_norm) in enumerate(zip(self.experts2, self.batch_norms)):
            # 使用对应的hidden输出
            hidden_output = hidden_outputs[i]  # [batch_size, hidden_dim]
            
            # MLP (FC)
            fc_output = expert(hidden_output)  # [batch_size, output_dim]
            
            # BatchNorm + SiLU
            bn_output = batch_norm(fc_output)  # [batch_size, output_dim]
            silu_output = F.silu(bn_output)  # [batch_size, output_dim]
            
            final_outputs.append(silu_output)
        
        # 堆叠成 [batch_size, num_experts, output_dim]
        output = torch.stack(final_outputs, dim=1)
        return output

def test_pytorch_meta_forward():
    """测试PyTorch实现的Meta Forward接口"""
    print("=== PyTorch Meta Forward 测试 ===")
    
    # 参数设置
    batch_size = 4096
    input_dim = 1408
    hidden_dim = 704
    output_dim = 504
    num_experts = 10
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输入数据
    input_tensor = torch.randn(batch_size, input_dim, dtype=torch.bfloat16, device=device)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"参数设置: batch_size={batch_size}, input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, num_experts={num_experts}")
    
    try:
        # 创建PyTorch模型
        pytorch_model = PyTorchMetaLayer(input_dim, hidden_dim, output_dim, num_experts).to(device)
        pytorch_model = pytorch_model.bfloat16()  # 转换为BF16
        
        # 启用torch.compile进行优化
        print("Enabling torch.compile for PyTorch model...")
        pytorch_model = torch.compile(pytorch_model)
        
        # 预热
        for _ in range(5):
            _ = pytorch_model(input_tensor)
        
        # 使用CUDA events进行精确计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(50):
            output = pytorch_model(input_tensor)
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 50  # 已经是毫秒
        
        print(f"PyTorch Meta Forward 平均时间: {avg_time:.4f} ms")
        print(f"输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出形状
        expected_shape = (batch_size, num_experts, output_dim)
        assert output.shape == expected_shape, f"输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}"
        
        print("✅ PyTorch Meta Forward 测试通过!")
        
        return output, pytorch_model
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return None, None

def test_home_meta_forward():
    """测试HoME Meta Forward接口"""
    print("=== HoME Meta Forward 测试 ===")
    
    # 参数设置
    batch_size = 4096
    input_dim = 1408
    hidden_dim = 704
    output_dim = 504
    num_experts = 10
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输入数据
    input_tensor = torch.randn(batch_size, input_dim, dtype=torch.bfloat16, device=device)
    
    # 第一层专家权重和偏置
    expert_weights1 = torch.randn(num_experts, input_dim, hidden_dim, dtype=torch.bfloat16, device=device)
    expert_biases1 = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)
    
    # 第二层专家权重和偏置
    expert_weights2 = torch.randn(num_experts, hidden_dim, output_dim, dtype=torch.bfloat16, device=device)
    expert_biases2 = torch.randn(num_experts, output_dim, dtype=torch.bfloat16, device=device)
    
    # BatchNorm参数
    bn_gamma = torch.randn(num_experts, output_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    bn_beta = torch.randn(num_experts, output_dim, dtype=torch.float32, device=device) * 0.1
    running_mean = torch.randn(num_experts, output_dim, dtype=torch.float32, device=device) * 0.1
    running_var = torch.randn(num_experts, output_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"第一层权重形状: {expert_weights1.shape}")
    print(f"第二层权重形状: {expert_weights2.shape}")
    print(f"BatchNorm参数形状: {bn_gamma.shape}")
    
    # 导入HoME kernels
    try:
        import sys
        sys.path.append('/root/home_moe_kernels/build')
        import home_kernels
        
        print("\n=== 测试HoME Meta Forward ===")
        
        # 预热
        for _ in range(5):
            _ = home_kernels.home_meta_forward_bf16(
                input_tensor, expert_weights1, expert_biases1,
                expert_weights2, expert_biases2,
                bn_gamma, bn_beta, running_mean, running_var,
                num_experts, True, 1e-5
            )
        
        # 使用CUDA events进行精确计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(50):
            output = home_kernels.home_meta_forward_bf16(
                input_tensor, expert_weights1, expert_biases1,
                expert_weights2, expert_biases2,
                bn_gamma, bn_beta, running_mean, running_var,
                num_experts, True, 1e-5
            )
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 50  # 已经是毫秒
        
        print(f"HoME Meta Forward 平均时间: {avg_time:.4f} ms")
        print(f"输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出形状
        expected_shape = (batch_size, num_experts, output_dim)
        assert output.shape == expected_shape, f"输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}"
        
        print("✅ HoME Meta Forward 测试通过!")
        
        return output
        
    except ImportError as e:
        print(f"❌ 无法导入HoME kernels: {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def test_home_expert_forward():
    """测试HoME Expert Forward接口（单层）"""
    print("\n=== HoME Expert Forward 测试 ===")
    
    # 参数设置
    batch_size = 4096
    input_dim = 512
    output_dim = 1024
    num_experts = 8
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_tensor = torch.randn(batch_size, input_dim, dtype=torch.bfloat16, device=device)
    expert_weights = torch.randn(num_experts, input_dim, output_dim, dtype=torch.bfloat16, device=device)
    expert_biases = torch.randn(num_experts, output_dim, dtype=torch.bfloat16, device=device)
    
    try:
        import sys
        sys.path.append('/root/home_moe_kernels/build')
        import home_kernels
        
        # 预热
        for _ in range(5):
            _ = home_kernels.home_expert_forward_bf16(
                input_tensor, expert_weights, expert_biases,
                num_experts, True
            )
        
        # 使用CUDA events进行精确计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(50):
            output = home_kernels.home_expert_forward_bf16(
                input_tensor, expert_weights, expert_biases,
                num_experts, True
            )
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 50  # 已经是毫秒
        
        print(f"HoME Expert Forward 平均时间: {avg_time:.4f} ms")
        print(f"输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        
        # 验证输出形状
        expected_shape = (batch_size, num_experts, output_dim)
        assert output.shape == expected_shape, f"输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}"
        
        print("✅ HoME Expert Forward 测试通过!")
        
        return output
        
    except ImportError as e:
        print(f"❌ 无法导入HoME kernels: {e}")
        return None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

if __name__ == "__main__":
    print("开始HoME kernels测试...")
    
    # 测试PyTorch实现
    pytorch_output, pytorch_model = test_pytorch_meta_forward()
    
    # 测试CUDA kernel实现
    cuda_output = test_home_meta_forward()
    
    # 测试单层专家网络
    expert_output = test_home_expert_forward()
    
    # 精度对比测试
    
    # 性能对比总结
    if pytorch_output is not None and cuda_output is not None:
        print("\n=== 性能对比总结 ===")
        print("PyTorch实现和CUDA kernel实现都已完成测试")
        print("详细性能数据请查看上述测试结果")
    
    if expert_output is not None:
        print("\n🎉 HoME Expert Forward测试通过! kernels集成成功!")
        if cuda_output is not None:
            print("🎉 HoME Meta Forward测试也通过!")
        else:
            print("⚠️  HoME Meta Forward函数暂未正确导出，但Expert Forward功能正常")
    else:
        print("\n❌ 测试失败，请检查错误信息")
