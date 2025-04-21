#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MOEMultiClassification(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        target_size: int,
        label_size: int,
        top_k: int,
        dropout=None,
        norm_topk_prob=True,
    ):
        super(MOEMultiClassification, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = ExpertMLP(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
        )
        self.label_size = label_size
        self.shared_expert_gate = torch.nn.Linear(hidden_size, 1, bias=False)
        self.feature = nn.Linear(
            in_features=hidden_size * 3, out_features=hidden_size)
        self.output = nn.Linear(in_features=hidden_size,
                                out_features=target_size)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ):
        # 语义模型编码得到向量表征
        model_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = model_output.last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # MOE层，每个专家获取表征
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        coord = (
            torch.unsqueeze(torch.range(0, batch_size - 1)
                            * sequence_length, dim=1)
            .to(input_ids.device)
            .int()
        )

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_ids = torch.topk(
            routing_weights, self.top_k, dim=1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            selected_id = torch.squeeze(selected_ids[:, :, expert_idx]) + coord
            current_real = torch.index_select(
                input=hidden_states.reshape(-1, hidden_dim),
                dim=0,
                index=selected_id.reshape(-1),
            )
            current_real = current_real.reshape(batch_size, -1, hidden_dim)
            current_real = self.dropout(current_real)
            rel = expert_layer(x=current_real)
            rel = self.dropout(rel)
            rel = torch.einsum(
                "bs, bsh -> bh", torch.squeeze(routing_weights[:, :, expert_idx]), rel)
            rel = torch.unsqueeze(rel, dim=1)
            expert_outputs.append(rel)

        expert_outputs = torch.concat(expert_outputs, dim=1)

        # 共享表征
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = self.dropout(shared_expert_output)
        shared_expert_output = torch.einsum(
            "bsh, bsh -> bh", F.sigmoid(self.shared_expert_gate(hidden_states)), shared_expert_output)
        shared_expert_output = torch.unsqueeze(shared_expert_output, dim=1).expand(
            batch_size, self.num_experts, hidden_dim)
        cls_output = torch.unsqueeze(hidden_states[:, 0], dim=1).expand(
            batch_size, self.num_experts, hidden_dim)
        cls_output = self.dropout(cls_output)
        final_hidden_states = self.feature(torch.concat(
            [expert_outputs, shared_expert_output, cls_output], dim=-1))

        # 求取概率分布
        logits = self.output(final_hidden_states)

        return logits
