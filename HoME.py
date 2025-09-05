import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Dict
from MLP import MLPLayer
from Dense import DenseLayer
import math


class LoRAGate(nn.Module):
    """
    LoRA低秩分解门控网络
    实现公式: Fea_Gate(v) = 2 × Sigmoid(v(BA))
    其中B和A是低秩矩阵，用于降维和升维
    """
    
    def __init__(self, input_dim: int, rank: int = 16, output_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.output_dim = output_dim if output_dim else input_dim
        
        # LoRA低秩分解矩阵
        # A: input_dim -> rank (降维)
        self.A = nn.Parameter(torch.randn(self.input_dim, self.rank) * 0.02)
        # B: rank -> output_dim (升维)
        self.B = nn.Parameter(torch.randn(self.rank, self.output_dim) * 0.02)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化LoRA权重"""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: Fea_Gate(v) = 2 × Sigmoid(v(BA))
        
        Args:
            x: 输入特征向量 [batch_size, input_dim] 或 [input_dim]
        
        Returns:
            gate_output: 门控输出 [batch_size, output_dim] 或 [output_dim]
        """
        # 确保输入是2D张量
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [input_dim] -> [1, input_dim]
            single_input = True
        else:
            single_input = False
        
        # 计算LoRA低秩分解: v(BA)
        # 先计算 BA
        BA = torch.mm(self.A, self.B)  # [input_dim, output_dim]
        
        # 再计算 v(BA)
        v_BA = torch.mm(x, BA)  # [batch_size, output_dim]
        
        # 应用Sigmoid激活并乘以2
        gate_output = 2.0 * torch.sigmoid(v_BA)
        
        # 如果输入是1D，输出也应该是1D
        if single_input:
            gate_output = gate_output.squeeze(0)  # [1, output_dim] -> [output_dim]
        
        return gate_output


class HoMELayer(nn.Module):
    """
    Hierarchical Mixture of Experts (HoME) Layer
    基于PLE层改进，使用MLP替换DenseLayer，添加BatchNorm和Swish激活
    
    门控机制设计:
    - f门控: 用于输入特征分解，使用LoRA低秩分解处理高维特征
    - l门控: 用于专家输出的局部控制 (Local Gate)
    - g门控: 用于专家输出的全局控制 (Global Gate)
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
                 lora_rank: int = 2):
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
        self.meta_output_dim = self.dim * 2
        self.meta_experts_num = num_experts * 2
        
        # 默认任务分组：重新设计为三组
        if task_groups is None:
            task_groups = {
                'recruitment_chat': [0, 1, 2],  # detail, addf, chat - 招聘和聊天任务组
                'success_refuse': [3, 4],        # success, refuse - 决策任务组
                'share': [5]                    # share - 分享任务组 (单独一组)
            }
        # 确保task_groups是有效的字典
        if not isinstance(task_groups, dict) or len(task_groups) == 0:
            task_groups = {
                'recruitment_chat': [0, 1, 2],  # detail, addf, chat
                'success_refuse': [3, 4],        # success, refuse
                'share': [5]                    # share
            }
        self.task_groups = task_groups
        
        # 构建任务到组的映射
        self._build_task_to_group_mapping()
        
        # 构建专家网络
        self._build_expert_networks()
        
        # 构建local expert网络
        self._build_local_expert_networks()
        
        # 构建门控网络
        self._build_gate_networks()
        
        # 构建Meta专家层
        self._build_meta_expert_layers()
        
        # 构建f门控网络 (LoRA低秩分解)
        if self.use_lora_gate:
            self._build_lora_gate_networks()

    
    def _build_lora_gate_networks(self):
        """构建f门控网络，使用LoRA低秩分解"""
        # 第一层：按任务组共享f门控（每个组有自己的f_gate）
        self.group_f_gates = nn.ModuleDict()
        for group_name, task_indices in self.task_groups.items():
            if task_indices:  # 只要有任务索引就创建
                # 计算该组内所有task的输入总维度
                group_total_dim = sum(self.input_dims[i] for i in task_indices)
                # 第一层：每个group共享一个f_gate，输入输出维度相同
                # 现在不再需要拆分，直接输出统一维度
                f_gate = LoRAGate(
                    input_dim=group_total_dim,
                    rank=self.lora_rank,
                    output_dim=group_total_dim  # 输出维度与输入维度相同
                )
                # 统一命名规则：{分组名}_f_gate
                self.group_f_gates[f"{group_name}_f_gate"] = f_gate
        
        # 第二层：每个expert都有独立的f门控
        self.expert_f_gates = nn.ModuleDict()
        
        # 为每个expert创建f_gate
        for i in range(self.num_tasks):
            # 获取task所属的组名
            f_gate = LoRAGate(
                input_dim=self.dim,  # 输入是上一层的输出维度
                rank=self.lora_rank,
                output_dim=self.dim
            )
            self.expert_f_gates[f"task_{i}_f_gate"] = f_gate

        # 第二层local expert的f_gate
        for i in range(len(self.task_groups) - 1):
            f_gate = LoRAGate(
                input_dim=self.meta_output_dim,  # local expert的输入是meta层的输出
                rank=self.lora_rank,
                output_dim=self.meta_output_dim
            )
            self.group_f_gates[f"group_{i}_f_gate"] = f_gate
    

    
    def _build_task_to_group_mapping(self):
        """构建任务索引到组名的映射"""
        self.task_to_group = {}
        for group_name, task_indices in self.task_groups.items():
            for task_idx in task_indices:
                self.task_to_group[task_idx] = group_name
    
    def _build_expert_networks(self):
        """构建专家网络 - 第二层：面向具体task的expert层"""
        # 第二层：每个expert都有独立的f_gate，每个task有num_experts个专家
        self.task_experts = nn.ModuleDict()
        for i in range(self.num_tasks):
            task_key = f"task_{i}"
            task_experts = nn.ModuleList()
            
            # 其他任务有num_experts个专家
            for j in range(self.num_experts):
                # 创建expert + BatchNorm + Swish的组合
                expert = MLPLayer(self.meta_output_dim, [self.dim], activate="relu", name=f"task_{i}_expert_{j}")
                batch_norm = nn.BatchNorm1d(self.dim)
                
                # 将expert和batch_norm组合成一个Sequential模块
                expert_with_bn = nn.Sequential(
                    expert,
                    batch_norm,
                    nn.SiLU()  # Swish激活函数
                )
                task_experts.append(expert_with_bn)
            
            self.task_experts[task_key] = task_experts

    def _build_local_expert_networks(self):
        """构建local expert网络"""
        self.local_experts = nn.ModuleDict()
        for i in range(len(self.task_groups) - 1):
            local_experts = nn.ModuleList()
            for j in range(self.num_experts):
                # 创建local_expert + BatchNorm + Swish的组合
                local_expert = MLPLayer(self.meta_output_dim, [self.dim], activate="relu", name=f"group_{i}_local_expert_{j}")
                batch_norm = nn.BatchNorm1d(self.dim)
                
                # 将local_expert和batch_norm组合成一个Sequential模块
                local_expert_with_bn = nn.Sequential(
                    local_expert,
                    batch_norm,
                    nn.SiLU()  # Swish激活函数
                )
                local_experts.append(local_expert_with_bn)
            self.local_experts[f"group_{i}_local_experts"] = local_experts
    
    def _build_gate_networks(self):
        """构建门控网络"""
        # Meta层门控网络
        self._build_meta_gate_networks()
        
        # Task层门控网络
        self._build_task_gate_networks()
    
    def _build_meta_gate_networks(self):
        """构建Meta层门控网络"""
        # Meta Specific的l_gate：每个meta specific输出都有唯一的l_gate（包括share组）
        self.meta_specific_l_gates = nn.ModuleDict()
        for group_name in self.task_groups.keys():
            gate = DenseLayer(self.l_gate_dim, self.meta_experts_num, "softmax", name=f"meta_{group_name}_l_gate")
            self.meta_specific_l_gates[group_name] = gate
        
        # Meta Specific的g_gate：每个meta specific输出都有唯一的g_gate（包括share组）
        self.meta_specific_g_gates = nn.ModuleDict()
        for group_name in self.task_groups.keys():
            if group_name == 'share':
                # share组需要处理所有分组的expert
                total_experts = self.meta_experts_num * len(self.task_groups)  # 所有分组的expert总数
                gate = DenseLayer(self.g_gate_dim, total_experts, "softmax", name=f"meta_{group_name}_g_gate")
            else:
                # 非share组需要处理自己分组和share分组的expert
                total_experts = self.meta_experts_num * 2  # 当前分组 + share分组
                gate = DenseLayer(self.g_gate_dim, total_experts, "softmax", name=f"meta_{group_name}_g_gate")
            self.meta_specific_g_gates[group_name] = gate
        
    def _build_task_gate_networks(self):
        """构建Task层门控网络"""
        # Task层的l_gate：每个task对应所有特定expert
        self.task_l_gates = nn.ModuleDict()
        for i in range(self.num_tasks):
            if self.num_experts == 1:
                # 如果只有一个expert，使用sigmoid
                gate = DenseLayer(self.l_gate_dim_task, 1, "sigmoid", name=f"task_{i}_l_gate")
            else:
                # 如果有多个expert，使用softmax
                gate = DenseLayer(self.l_gate_dim_task, self.num_experts, "softmax", name=f"task_{i}_l_gate")
            self.task_l_gates[f"task_{i}"] = gate
        
        # Task层的g_gate：
        self.task_g_gates = nn.ModuleDict()
        for i in range(self.num_tasks):
                gate = DenseLayer(self.g_gate_dim, self.num_experts * 3, "softmax", name=f"task_{i}_g_gate")
                self.task_g_gates[f"task_{i}"] = gate
    
    def _build_meta_expert_layers(self):
        """构建Meta专家层"""
        # Meta特定专家层
        self.meta_specific_experts = nn.ModuleDict()
        # 预定义所有可能的组名
        #all_group_names = list(self.task_groups.keys())
        for group_name, task_indices in self.task_groups.items():
            meta_experts = nn.ModuleList()
            group_total_dim = sum(self.input_dims[i] for i in task_indices)
            for j in range(self.meta_experts_num):
                # Meta层的输入维度是任务级别的输出维度
                if group_total_dim // 2 > self.meta_output_dim:
                    expert = MLPLayer(group_total_dim, [group_total_dim // 2, self.meta_output_dim], activate="relu", name=f"meta_{group_name}_expert_{j}")
                else:
                    expert = MLPLayer(group_total_dim, [group_total_dim, self.meta_output_dim], activate="relu", name=f"meta_{group_name}_expert_{j}")
                
                # 为meta expert添加BatchNorm和Swish激活
                batch_norm = nn.BatchNorm1d(self.meta_output_dim)
                expert_with_bn = nn.Sequential(
                    expert,
                    batch_norm,
                    nn.SiLU()  # Swish激活函数
                )
                meta_experts.append(expert_with_bn)
            self.meta_specific_experts[group_name] = meta_experts
        
    
    def _apply_f_gate(self, input_tensor: torch.Tensor, gate_type: str = "group", group_name: str = None, expert_idx: int = None, task_idx: int = None) -> torch.Tensor:
        """
        应用f门控 (LoRA低秩分解)
        
        f门控的作用:
        1. 第一层：按任务组共享f门控，每个组有自己的f_gate，输出统一维度
        2. 第二层：每个expert都有独立的f门控，处理上一层的输出
        
        Args:
            input_tensor: 输入特征张量
            gate_type: 门控类型 ("group", "expert", "local")
            group_name: 组名（当gate_type="group"时使用，如"recruitment_chat"）
            expert_idx: 专家索引（当gate_type="expert"或"local"时使用）
            task_idx: 任务索引（当gate_type="expert"时使用，用于构造expert key）
            
        Returns:
            经过f门控处理的特征张量
        """
        if not self.use_lora_gate:
            return input_tensor
        
        if gate_type == "group" and group_name is not None:
            # 第一层：组内共享f门控，输出统一维度
            # 统一命名规则：{分组名}_f_gate
            group_key = f"{group_name}_f_gate"
            if group_key in self.group_f_gates:
                return self.group_f_gates[group_key](input_tensor)
        elif gate_type == "expert" and expert_idx is not None and task_idx is not None:
            # 第二层：专家特定f门控
            # 统一命名规则：task_{task_idx}_f_gate
            expert_key = f"task_{task_idx}_f_gate"
            if expert_key in self.expert_f_gates:
                return self.expert_f_gates[expert_key](input_tensor)
        elif gate_type == "local" and expert_idx is not None:
            # 第二层local expert的f门控
            # 统一命名规则：group_{expert_idx}_f_gate
            group_key = f"group_{expert_idx}_f_gate"
            if group_key in self.group_f_gates:
                return self.group_f_gates[group_key](input_tensor)
        
        # 如果没有找到对应的f门控，返回原始输入
        return input_tensor
    
    def _process_task_level(self, meta_outputs: Dict[str, torch.Tensor], l_gate_states_task: torch.Tensor, g_gate_states: torch.Tensor):
        """处理任务级别的专家网络 - 第二层：面向具体task的expert层"""
        task_outputs = {}
        
        # 第一步：收集所有expert的输出（包括task特定expert和local expert）
        all_expert_outputs = {}
        local_expert_outputs = {}
        
        # 处理task特定expert
        for i in range(self.num_tasks):
            task_key = f"task_{i}"
            if task_key in self.task_experts:
                # 获取task所属的组名
                task_group = self.task_to_group.get(i, "share")
                
                # 使用Meta层的输出作为输入（确定一定会有对应的Meta层输出）
                gated_input = meta_outputs[task_group]
                
                # 收集task特定expert的输出
                task_expert_outputs = []
                for j, expert in enumerate(self.task_experts[task_key]):
                    # 先通过f_gate，再通过expert（expert已包含BatchNorm和Swish）
                    expert_gated_input = self._apply_f_gate(gated_input, "expert", j, i)
                    expert_output = expert(expert_gated_input)
                    task_expert_outputs.append(expert_output)
                
                # 将task expert输出stack成tensor
                task_expert_tensor = torch.stack(task_expert_outputs, dim=2)  # [batch_size, dim, num_experts]
                all_expert_outputs[task_key] = task_expert_tensor
        
        # 处理local expert
        if hasattr(self, 'local_experts'):
            for group_key, local_experts in self.local_experts.items():
                group_idx = int(group_key.split('_')[1])  # 从"group_0_local_experts"提取索引
                group_name = list(self.task_groups.keys())[group_idx]  # 获取对应的组名
                
                if group_name in meta_outputs:
                    gated_input = meta_outputs[group_name]
                    
                    # 收集local expert的输出
                    local_expert_outputs_list = []
                    for local_expert in local_experts:
                        # 先通过f_gate，再通过local expert
                        expert_gated_input = self._apply_f_gate(gated_input, "local", expert_idx=group_idx)
                        expert_output = local_expert(expert_gated_input)
                        local_expert_outputs_list.append(expert_output)
                    
                    if local_expert_outputs_list:
                        # 将local expert输出stack成tensor
                        local_expert_tensor = torch.stack(local_expert_outputs_list, dim=2)  # [batch_size, dim, num_experts]
                        local_expert_outputs[group_name] = local_expert_tensor
        
        # 第二步：规划如何通过l_gate和g_gate
        for i in range(self.num_tasks):
            task_key = f"task_{i}"
            if task_key in all_expert_outputs:
                task_group = self.task_to_group.get(i, "share")
                
                # 获取当前task的expert输出
                task_expert_tensor = all_expert_outputs[task_key]  # [batch_size, dim, num_experts]
                
                # 应用l_gate到task特定expert输出（Local Gate）
                l_gate = self.task_l_gates[f"task_{i}"](l_gate_states_task)
                l_gate = l_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                l_gate_output = torch.sum(task_expert_tensor * l_gate, dim=2)  # [batch_size, dim]
                
                # 应用g_gate，整合所有相关信息
                g_gate_inputs = [task_expert_tensor]  # 当前task的expert输出
                
                # 添加local expert输出（如果存在）
                if task_group in local_expert_outputs:
                    g_gate_inputs.append(local_expert_outputs[task_group])
                
                # 添加global expert输出（share任务的输出）
                if i != 5:  # 非share任务需要share任务的输出
                    share_task_key = "task_5"
                    if share_task_key in all_expert_outputs:
                        g_gate_inputs.append(all_expert_outputs[share_task_key])
                
                # 计算g_gate输出
                if len(g_gate_inputs) > 1:
                    # 沿着expert维度拼接，保持特征维度为dim
                    combined_g_gate_input = torch.cat(g_gate_inputs, dim=2)  # [batch_size, dim, num_experts*3]
                    
                    # 应用g_gate
                    g_gate = self.task_g_gates[f"task_{i}"](g_gate_states)
                    g_gate = g_gate.unsqueeze(1)  # [batch_size, 1, num_experts * 3]
                    
                    # 计算g_gate输出：每个expert输出dim维，g_gate为每个expert分配权重
                    g_gate_output = torch.sum(combined_g_gate_input * g_gate, dim=2)  # [batch_size, dim]
                else:
                    # 如果没有额外的输入，使用原始的g_gate计算
                    g_gate = self.task_g_gates[f"task_{i}"](g_gate_states)
                    g_gate = g_gate.unsqueeze(1)  # [batch_size, 1, num_experts * 3]
                    
                    # 需要调整expert_tensor的维度以匹配g_gate
                    expert_tensor_expanded = task_expert_tensor.repeat(1, 1, 3)  # [batch_size, dim, num_experts*3]
                    g_gate_output = torch.sum(expert_tensor_expanded * g_gate, dim=2)  # [batch_size, dim]
                
                # 最终输出：l_gate + g_gate
                final_task_output = l_gate_output + g_gate_output  # [batch_size, dim]
                task_outputs[task_key] = final_task_output
        
        return task_outputs
    
    def _process_global_gate(self, task_outputs: Dict, gate_states: torch.Tensor):
        """处理g门控：share任务的所有expert只会和其他task的global gate相连"""
        # 注意：g_gate现在已经在_process_task_level中处理了
        # 这个方法保留用于可能的扩展功能
        return {}
    
    def _process_meta_level(self, input_states: List[torch.Tensor], l_gate_states: torch.Tensor, g_gate_states: torch.Tensor):
        """处理Meta级别的专家网络 - 第一层：task group级别的专家网络"""
        meta_outputs = {}

        # 预定义所有组名
        all_group_names = list(self.task_groups.keys())
        
        # 处理Meta特定层：每个task group有自己的专家网络
        for group_name in all_group_names:
            if group_name in self.meta_specific_experts:
                # 组合该组内所有task的输入
                group_inputs = []
                for i in range(self.num_tasks):
                    task_group = self.task_to_group.get(i, "share")
                    if task_group == group_name:
                        group_inputs.append(input_states[i])
                
                # 确保有输入数据
                if group_inputs:
                    # 将同一分组的task输入拼接起来，而不是求平均
                    # 这样每个分组都有完整的输入信息
                    combined_group_input = torch.cat(group_inputs, dim=-1)  # [batch_size, total_dim]

                    # Meta特定专家：每个group有num_experts个专家
                    meta_experts = self.meta_specific_experts[group_name]
                    meta_expert_outputs = []
                    
                    for j, expert in enumerate(meta_experts):
                        # Meta层：先通过f_gate，再通过meta expert
                        if self.use_lora_gate:
                            # 先通过group级别的f_gate
                            expert_gated_input = self._apply_f_gate(combined_group_input, "group", group_name)
                            # 直接使用拼接后的输入，不需要拆分
                            # expert_gated_input已经是完整的拼接输入，可以直接用于meta expert
                        else:
                            expert_gated_input = combined_group_input
                        # 再通过meta expert（expert已包含BatchNorm和Swish）
                        expert_output = expert(expert_gated_input)
                        meta_expert_outputs.append(expert_output)

                    # 应用Meta门控：l_gate和g_gate
                    if group_name in self.meta_specific_l_gates:
                        l_gate = self.meta_specific_l_gates[group_name](l_gate_states)
                        l_gate = l_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                        
                        if group_name in self.meta_specific_g_gates:
                            g_gate = self.meta_specific_g_gates[group_name](g_gate_states)
                            g_gate = g_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                        else:
                            g_gate = torch.zeros_like(l_gate)
                        
                        # 组合Meta特定专家输出：分别应用l_gate和g_gate
                        if meta_expert_outputs:
                            meta_expert_tensor = torch.stack(meta_expert_outputs, dim=2)  # [batch_size, dim, num_experts]
                            
                            # 应用l_gate到当前group的expert
                            l_gate_output = torch.sum(meta_expert_tensor * l_gate, dim=2)
                            
                            # 应用g_gate：需要接收来自其他分组的expert输出
                            if group_name == 'share':
                                # share组：接收所有分组的expert输出
                                # 由于share组需要等待其他组处理完成，我们先只使用l_gate
                                # g_gate的处理将在所有组处理完成后进行
                                weighted_meta_output = l_gate_output
                            else:
                                # 非share组：接收自己分组和share分组的expert输出
                                # 先处理share分组（如果存在）
                                share_expert_outputs = []
                                if 'share' in self.task_groups and 'share' in self.meta_specific_experts:
                                    # 获取share组的输入
                                    share_task_indices = self.task_groups['share']
                                    share_inputs = [input_states[i] for i in share_task_indices]
                                    if share_inputs:
                                        share_combined_input = torch.cat(share_inputs, dim=-1)  # [batch_size, share_total_dim]
                                        
                                        share_experts = self.meta_specific_experts['share']
                                        for expert in share_experts:
                                            # 使用share组自己的输入
                                            share_expert_output = expert(share_combined_input)
                                            share_expert_outputs.append(share_expert_output)
                                
                                if share_expert_outputs:
                                    # 将当前group的expert输出和share expert输出拼接
                                    share_expert_tensor = torch.stack(share_expert_outputs, dim=2)  # [batch_size, dim, num_share_experts]
                                    combined_g_input = torch.cat([meta_expert_tensor, share_expert_tensor], dim=2)  # [batch_size, dim, total_experts]
                                    
                                    # 应用g_gate到所有expert（当前group + share）
                                    if g_gate.size(-1) == combined_g_input.size(-1):
                                        g_gate_output = torch.sum(combined_g_input * g_gate, dim=2)
                                        weighted_meta_output = l_gate_output + g_gate_output
                                    else:
                                        weighted_meta_output = l_gate_output
                                else:
                                    weighted_meta_output = l_gate_output
                            
                            meta_outputs[group_name] = weighted_meta_output

        # 处理Meta共享层：所有group共享的专家网络
        if hasattr(self, 'meta_shared_experts') and self.meta_shared_experts:
            # 组合所有group的输出作为输入
            all_group_outputs = []
            for group_name in all_group_names:
                if group_name in meta_outputs:
                    all_group_outputs.append(meta_outputs[group_name])
            
            if all_group_outputs:
                # 计算全局共享输入
                global_input = torch.mean(torch.stack(all_group_outputs, dim=0), dim=0)  # [batch_size, dim]
                
                meta_shared_expert_outputs = []
                for j, expert in enumerate(self.meta_shared_experts):
                    # Meta共享层：先通过f_gate，再通过meta expert
                    if self.use_lora_gate:
                        # 先通过share group的f_gate
                        expert_gated_input = self._apply_f_gate(global_input, "group", "share")
                    else:
                        expert_gated_input = global_input
                    # 再通过meta expert（expert已包含BatchNorm和Swish）
                    expert_output = expert(expert_gated_input)
                    meta_shared_expert_outputs.append(expert_output)

                # 应用Meta共享门控：l_gate和g_gate
                if hasattr(self, 'meta_shared_l_gate') and hasattr(self, 'meta_shared_g_gate'):
                    meta_shared_l_gate = self.meta_shared_l_gate(l_gate_states)
                    meta_shared_l_gate = meta_shared_l_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                    
                    meta_shared_g_gate = self.meta_shared_g_gate(g_gate_states)
                    meta_shared_g_gate = meta_shared_g_gate.unsqueeze(1)  # [batch_size, 1, num_experts]
                    
                    # 组合Meta共享专家输出：l_gate + g_gate
                    if meta_shared_expert_outputs:
                        meta_shared_expert_tensor = torch.stack(meta_shared_expert_outputs, dim=2)
                        weighted_meta_shared_output = torch.sum(meta_shared_expert_tensor * (l_gate + g_gate), dim=2)
                        meta_outputs['shared'] = weighted_meta_shared_output

        return meta_outputs
    
    def forward(self, input_states: List[torch.Tensor], l_gate_states: torch.Tensor, g_gate_states: torch.Tensor, l_gate_states_task: torch.Tensor):
        """
        前向传播

        Args:
            input_states: 输入状态列表 [task_0_input, task_1_input, ...]
            l_gate_states: l门控状态 [batch_size, l_gate_dim]
            g_gate_states: g门控状态 [batch_size, g_gate_dim]

        Returns:
            final_outputs: 最终输出字典，包含每个任务的输出
        """
        # 第一层：Meta级别处理（task group级别的专家网络）
        meta_outputs = self._process_meta_level(input_states, l_gate_states, g_gate_states)
        
        # 第二层：Task级别处理（具体task级别的专家网络）
        # 使用Meta层的输出作为Task层的输入
        task_outputs = self._process_task_level(meta_outputs, l_gate_states_task, g_gate_states)
        
        return task_outputs
    

    
    def compile_model(self, fullgraph=True, mode="default"):
        """编译模型以提高性能"""
        print(f"Enabling compilation for HoME layer: fullgraph={fullgraph}, mode={mode}")
        # 启用编译
        return torch.compile(self, fullgraph=fullgraph, mode=mode)

