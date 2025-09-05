import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Dict, Optional
import math
import home_kernels  # 导入我们的CUDA kernel模块


class LoRAGateOptimized(nn.Module):
    """
    优化的LoRA低秩分解门控网络
    使用CUDA kernel加速计算
    """
    
    def __init__(self, input_dim: int, rank: int = 16, output_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.output_dim = output_dim if output_dim else input_dim
        
        # LoRA低秩分解矩阵
        self.A = nn.Parameter(torch.randn(self.input_dim, self.rank) * 0.02)
        self.B = nn.Parameter(torch.randn(self.rank, self.output_dim) * 0.02)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化LoRA权重"""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 使用CUDA kernel加速LoRA计算
        """
        # 确保输入是2D张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 使用CUDA kernel进行LoRA计算
        gate_output = home_kernels.lora_gate_forward(
            x, self.A, self.B, True
        )
        
        # 如果输入是1D，输出也应该是1D
        if single_input:
            gate_output = gate_output.squeeze(0)
        
        return gate_output


class ExpertNetworkOptimized(nn.Module):
    """
    优化的专家网络
    使用CUTLASS Group GEMM加速计算
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, 
                 use_batch_norm: bool = True, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        
        # 专家权重和偏置
        self.expert_weights = nn.Parameter(
            torch.randn(num_experts, input_dim, hidden_dim) * 0.02
        )
        self.expert_biases = nn.Parameter(
            torch.zeros(num_experts, hidden_dim) if use_bias else None
        )
        
        # BatchNorm参数
        if use_batch_norm:
            self.bn_weights = nn.Parameter(torch.ones(num_experts, hidden_dim))
            self.bn_biases = nn.Parameter(torch.zeros(num_experts, hidden_dim))
            self.running_mean = nn.Parameter(torch.zeros(num_experts, hidden_dim), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(num_experts, hidden_dim), requires_grad=False)
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 使用CUDA kernel加速专家计算
        """
        batch_size = x.size(0)
        num_selected_experts = expert_indices.size(0)
        
        # 使用CUDA kernel进行专家计算
        expert_outputs = home_kernels.home_expert_forward(
            x, self.expert_weights, self.expert_biases, 
            expert_indices, self.num_experts, self.use_bias
        )
        
        # 应用BatchNorm + SiLU融合算子
        if self.use_batch_norm:
            expert_outputs = home_kernels.fused_batch_norm_silu_forward(
                expert_outputs, self.bn_weights, self.bn_biases,
                self.running_mean, self.running_var, 1e-5
            )
        
        return expert_outputs


class GateNetworkOptimized(nn.Module):
    """
    优化的门控网络
    使用CUDA kernel加速门控计算
    """
    
    def __init__(self, gate_dim: int, num_experts: int, 
                 activation: str = "softmax", name: str = ""):
        super().__init__()
        self.gate_dim = gate_dim
        self.num_experts = num_experts
        self.activation = activation
        self.name = name
        
        # 门控权重
        self.gate_weights = nn.Parameter(
            torch.randn(gate_dim, num_experts) * 0.02
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化门控权重"""
        nn.init.kaiming_uniform_(self.gate_weights, a=math.sqrt(5))
    
    def forward(self, gate_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 使用CUDA kernel加速门控计算
        """
        use_softmax = (self.activation == "softmax")
        
        # 使用CUDA kernel进行门控计算
        gate_output = home_kernels.gate_weights_forward(
            gate_states, self.gate_weights, use_softmax
        )
        
        return gate_output


class HoMELayerOptimized(nn.Module):
    """
    优化的Hierarchical Mixture of Experts (HoME) Layer
    使用CUDA kernel加速所有关键计算
    """
    
    def __init__(self,
                 num_experts: int,
                 input_dims: List[int],
                 dim: int,
                 l_gate_dim: int,
                 g_gate_dim: int,
                 l_gate_dim_task: int,
                 task_groups: Dict[str, List[int]] = None,
                 use_lora_gate: bool = True,
                 lora_rank: int = 2,
                 use_cuda_kernels: bool = True):
        super().__init__()
        
        # 验证所有输入维度是否一致（当启用LoRA时）
        if use_lora_gate and len(set(input_dims)) > 1:
            raise ValueError("All input_dims must be the same when use_lora_gate=True")
        
        self.num_experts = num_experts
        self.dim = dim
        self.l_gate_dim = l_gate_dim
        self.g_gate_dim = g_gate_dim
        self.l_gate_dim_task = l_gate_dim_task
        self.input_dims = input_dims
        self.num_tasks = len(input_dims)
        self.use_lora_gate = use_lora_gate
        self.lora_rank = lora_rank
        self.use_cuda_kernels = use_cuda_kernels
        self.meta_output_dim = self.dim * 2
        self.meta_experts_num = num_experts * 2
        
        # 默认任务分组
        if task_groups is None:
            task_groups = {
                'recruitment_chat': [0, 1, 2],
                'success_refuse': [3, 4],
                'share': [5]
            }
        self.task_groups = task_groups
        
        # 构建任务到组的映射
        self._build_task_to_group_mapping()
        
        # 构建专家网络
        self._build_expert_networks()
        
        # 构建门控网络
        self._build_gate_networks()
        
        # 构建f门控网络
        if self.use_lora_gate:
            self._build_lora_gate_networks()
    
    def _build_task_to_group_mapping(self):
        """构建任务索引到组名的映射"""
        self.task_to_group = {}
        for group_name, task_indices in self.task_groups.items():
            for task_idx in task_indices:
                self.task_to_group[task_idx] = group_name
    
    def _build_expert_networks(self):
        """构建专家网络"""
        # Meta层专家网络
        self.meta_experts = nn.ModuleDict()
        for group_name, task_indices in self.task_groups.items():
            group_total_dim = sum(self.input_dims[i] for i in task_indices)
            self.meta_experts[group_name] = ExpertNetworkOptimized(
                input_dim=group_total_dim,
                hidden_dim=self.meta_output_dim,
                num_experts=self.meta_experts_num,
                use_batch_norm=True,
                use_bias=True
            )
        
        # Task层专家网络
        self.task_experts = nn.ModuleDict()
        for i in range(self.num_tasks):
            self.task_experts[f"task_{i}"] = ExpertNetworkOptimized(
                input_dim=self.meta_output_dim,
                hidden_dim=self.dim,
                num_experts=self.num_experts,
                use_batch_norm=True,
                use_bias=True
            )
        
        # Local专家网络
        self.local_experts = nn.ModuleDict()
        for i in range(len(self.task_groups) - 1):
            self.local_experts[f"group_{i}"] = ExpertNetworkOptimized(
                input_dim=self.meta_output_dim,
                hidden_dim=self.dim,
                num_experts=self.num_experts,
                use_batch_norm=True,
                use_bias=True
            )
    
    def _build_gate_networks(self):
        """构建门控网络"""
        # Meta层门控网络
        self.meta_l_gates = nn.ModuleDict()
        self.meta_g_gates = nn.ModuleDict()
        
        for group_name in self.task_groups.keys():
            self.meta_l_gates[group_name] = GateNetworkOptimized(
                gate_dim=self.l_gate_dim,
                num_experts=self.meta_experts_num,
                activation="softmax",
                name=f"meta_{group_name}_l_gate"
            )
            
            if group_name == 'share':
                total_experts = self.meta_experts_num * len(self.task_groups)
            else:
                total_experts = self.meta_experts_num * 2
            
            self.meta_g_gates[group_name] = GateNetworkOptimized(
                gate_dim=self.g_gate_dim,
                num_experts=total_experts,
                activation="softmax",
                name=f"meta_{group_name}_g_gate"
            )
        
        # Task层门控网络
        self.task_l_gates = nn.ModuleDict()
        self.task_g_gates = nn.ModuleDict()
        
        for i in range(self.num_tasks):
            activation = "sigmoid" if self.num_experts == 1 else "softmax"
            self.task_l_gates[f"task_{i}"] = GateNetworkOptimized(
                gate_dim=self.l_gate_dim_task,
                num_experts=self.num_experts,
                activation=activation,
                name=f"task_{i}_l_gate"
            )
            
            self.task_g_gates[f"task_{i}"] = GateNetworkOptimized(
                gate_dim=self.g_gate_dim,
                num_experts=self.num_experts * 3,
                activation="softmax",
                name=f"task_{i}_g_gate"
            )
    
    def _build_lora_gate_networks(self):
        """构建LoRA门控网络"""
        # 组级f门控
        self.group_f_gates = nn.ModuleDict()
        for group_name, task_indices in self.task_groups.items():
            if task_indices:
                group_total_dim = sum(self.input_dims[i] for i in task_indices)
                self.group_f_gates[f"{group_name}_f_gate"] = LoRAGateOptimized(
                    input_dim=group_total_dim,
                    rank=self.lora_rank,
                    output_dim=group_total_dim
                )
        
        # 专家级f门控
        self.expert_f_gates = nn.ModuleDict()
        for i in range(self.num_tasks):
            self.expert_f_gates[f"task_{i}_f_gate"] = LoRAGateOptimized(
                input_dim=self.dim,
                rank=self.lora_rank,
                output_dim=self.dim
            )
        
        # Local专家f门控
        for i in range(len(self.task_groups) - 1):
            self.group_f_gates[f"group_{i}_f_gate"] = LoRAGateOptimized(
                input_dim=self.meta_output_dim,
                rank=self.lora_rank,
                output_dim=self.meta_output_dim
            )
    
    def _apply_f_gate(self, input_tensor: torch.Tensor, gate_type: str = "group", 
                     group_name: str = None, expert_idx: int = None, 
                     task_idx: int = None) -> torch.Tensor:
        """应用f门控"""
        if not self.use_lora_gate:
            return input_tensor
        
        if gate_type == "group" and group_name is not None:
            group_key = f"{group_name}_f_gate"
            if group_key in self.group_f_gates:
                return self.group_f_gates[group_key](input_tensor)
        elif gate_type == "expert" and expert_idx is not None and task_idx is not None:
            expert_key = f"task_{task_idx}_f_gate"
            if expert_key in self.expert_f_gates:
                return self.expert_f_gates[expert_key](input_tensor)
        elif gate_type == "local" and expert_idx is not None:
            group_key = f"group_{expert_idx}_f_gate"
            if group_key in self.group_f_gates:
                return self.group_f_gates[group_key](input_tensor)
        
        return input_tensor
    
    def _process_meta_level(self, input_states: List[torch.Tensor], 
                           l_gate_states: torch.Tensor, 
                           g_gate_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """处理Meta级别"""
        meta_outputs = {}
        
        for group_name, task_indices in self.task_groups.items():
            # 组合该组内所有task的输入
            group_inputs = []
            for i in range(self.num_tasks):
                task_group = self.task_to_group.get(i, "share")
                if task_group == group_name:
                    group_inputs.append(input_states[i])
            
            if group_inputs:
                # 拼接输入
                combined_group_input = torch.cat(group_inputs, dim=-1)
                
                # 应用f门控
                if self.use_lora_gate:
                    gated_input = self._apply_f_gate(combined_group_input, "group", group_name)
                else:
                    gated_input = combined_group_input
                
                # 获取专家索引（这里简化处理，实际应该根据路由策略选择）
                expert_indices = torch.arange(self.meta_experts_num, device=gated_input.device, dtype=torch.int32)
                
                # 使用CUDA kernel进行专家计算
                meta_expert_outputs = self.meta_experts[group_name](
                    gated_input, expert_indices
                )  # [batch_size, num_experts, hidden_dim]
                
                # 转置为[batch_size, hidden_dim, num_experts]以匹配expert_weighted_sum_forward的期望格式
                meta_expert_outputs = meta_expert_outputs.transpose(1, 2)  # [batch_size, hidden_dim, num_experts]
                
                # 应用门控
                l_gate = self.meta_l_gates[group_name](l_gate_states)
                l_gate = l_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                
                # 使用CUDA kernel进行加权求和
                l_gate_output = home_kernels.expert_weighted_sum_forward(
                    meta_expert_outputs, l_gate
                )
                
                # 应用g门控（简化处理）
                g_gate = self.meta_g_gates[group_name](g_gate_states)
                g_gate = g_gate.unsqueeze(1)
                
                g_gate_output = home_kernels.expert_weighted_sum_forward(
                    meta_expert_outputs, g_gate
                )
                
                meta_outputs[group_name] = l_gate_output + g_gate_output
        
        return meta_outputs
    
    def _process_task_level(self, meta_outputs: Dict[str, torch.Tensor], 
                           l_gate_states_task: torch.Tensor, 
                           g_gate_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """处理Task级别"""
        task_outputs = {}
        
        # 收集所有专家输出
        all_expert_outputs = {}
        
        # 处理task特定专家
        for i in range(self.num_tasks):
            task_key = f"task_{i}"
            task_group = self.task_to_group.get(i, "share")
            
            if task_group in meta_outputs:
                gated_input = meta_outputs[task_group]
                
                # 应用f门控
                if self.use_lora_gate:
                    expert_gated_input = self._apply_f_gate(gated_input, "expert", i, i)
                else:
                    expert_gated_input = gated_input
                
                # 获取专家索引
                expert_indices = torch.arange(self.num_experts, device=gated_input.device, dtype=torch.int32)
                
                # 使用CUDA kernel进行专家计算
                task_expert_outputs = self.task_experts[task_key](
                    expert_gated_input, expert_indices
                )  # [batch_size, num_experts, hidden_dim]
                
                # 转置为[batch_size, hidden_dim, num_experts]以匹配expert_weighted_sum_forward的期望格式
                task_expert_outputs = task_expert_outputs.transpose(1, 2)  # [batch_size, hidden_dim, num_experts]
                
                all_expert_outputs[task_key] = task_expert_outputs
        
        # 处理local专家
        local_expert_outputs = {}
        for group_key, local_experts in self.local_experts.items():
            group_idx = int(group_key.split('_')[1])
            group_name = list(self.task_groups.keys())[group_idx]
            
            if group_name in meta_outputs:
                gated_input = meta_outputs[group_name]
                
                # 应用f门控
                if self.use_lora_gate:
                    expert_gated_input = self._apply_f_gate(gated_input, "local", group_idx)
                else:
                    expert_gated_input = gated_input
                
                # 获取专家索引
                expert_indices = torch.arange(self.num_experts, device=gated_input.device, dtype=torch.int32)
                
                # 使用CUDA kernel进行专家计算
                local_outputs = local_experts(
                    expert_gated_input, expert_indices
                )  # [batch_size, num_experts, hidden_dim]
                
                # 转置为[batch_size, hidden_dim, num_experts]
                local_outputs = local_outputs.transpose(1, 2)  # [batch_size, hidden_dim, num_experts]
                
                local_expert_outputs[group_name] = local_outputs
        
        # 应用门控并计算最终输出
        for i in range(self.num_tasks):
            task_key = f"task_{i}"
            if task_key in all_expert_outputs:
                task_group = self.task_to_group.get(i, "share")
                
                # 获取当前task的专家输出
                task_expert_tensor = all_expert_outputs[task_key]
                
                # 应用l门控
                l_gate = self.task_l_gates[task_key](l_gate_states_task)
                l_gate = l_gate.unsqueeze(1)
                
                l_gate_output = home_kernels.expert_weighted_sum_forward(
                    task_expert_tensor, l_gate
                )
                
                # 应用g门控
                g_gate_inputs = [task_expert_tensor]
                
                # 添加local专家输出
                if task_group in local_expert_outputs:
                    g_gate_inputs.append(local_expert_outputs[task_group])
                
                # 添加share任务输出
                if i != 5:  # 非share任务
                    share_task_key = "task_5"
                    if share_task_key in all_expert_outputs:
                        g_gate_inputs.append(all_expert_outputs[share_task_key])
                
                if len(g_gate_inputs) > 1:
                    combined_g_gate_input = torch.cat(g_gate_inputs, dim=2)
                    g_gate = self.task_g_gates[task_key](g_gate_states)
                    g_gate = g_gate.unsqueeze(1)
                    
                    g_gate_output = home_kernels.expert_weighted_sum_forward(
                        combined_g_gate_input, g_gate
                    )
                else:
                    g_gate = self.task_g_gates[task_key](g_gate_states)
                    g_gate = g_gate.unsqueeze(1)
                    
                    expert_tensor_expanded = task_expert_tensor.repeat(1, 1, 3)
                    g_gate_output = home_kernels.expert_weighted_sum_forward(
                        expert_tensor_expanded, g_gate
                    )
                
                task_outputs[task_key] = l_gate_output + g_gate_output
        
        return task_outputs
    
    def forward(self, input_states: List[torch.Tensor], 
                l_gate_states: torch.Tensor, 
                g_gate_states: torch.Tensor, 
                l_gate_states_task: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 第一层：Meta级别处理
        meta_outputs = self._process_meta_level(input_states, l_gate_states, g_gate_states)
        
        # 第二层：Task级别处理
        task_outputs = self._process_task_level(meta_outputs, l_gate_states_task, g_gate_states)
        
        return task_outputs
    
    def compile_model(self, fullgraph=True, mode="default"):
        """编译模型以提高性能"""
        print(f"Enabling compilation for optimized HoME layer: fullgraph={fullgraph}, mode={mode}")
        return torch.compile(self, fullgraph=fullgraph, mode=mode)

