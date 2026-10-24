import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家网络"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2

        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.network(x)


class MoEGate(nn.Module):
    """MoE门控网络"""

    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络：根据输入决定使用哪些专家
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        gate_logits = self.gate_net(x)  # [batch, seq_len, num_experts]

        # 选择top-k个专家
        top_k_gates, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # softmax归一化权重
        weights = F.softmax(top_k_gates, dim=-1)  # [batch, seq_len, top_k]

        return weights, top_k_indices


class MoESelfAttention(nn.Module):
    """基于MoE的自注意力模块（替换原来的SAtt）- 向量化优化版"""

    def __init__(self, dim, nhead=4, dropout=0.2, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # MoE层：多个专家网络
        self.experts = nn.ModuleList([
            Expert(dim, dim * 2) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = MoEGate(dim, num_experts, top_k)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, dim]
        返回: [batch, seq_len, dim]
        优化：使用向量化操作替代循环
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. MoE路由
        weights, indices = self.gate(x)  # weights: [batch, seq_len, top_k], indices: [batch, seq_len, top_k]

        # 2. 展平为 [batch*seq_len, dim] 以便批量处理
        x_flat = x.view(-1, self.dim)  # [batch*seq_len, dim]
        total_tokens = x_flat.size(0)

        # 3. 初始化输出
        moe_output_flat = torch.zeros_like(x_flat)  # [batch*seq_len, dim]

        # 4. 对每个专家，找出所有需要它的token，批量计算
        for expert_idx in range(self.num_experts):
            # 找到所有使用该专家的token位置
            mask_expert = (indices == expert_idx)  # [batch, seq_len, top_k]
            token_mask = mask_expert.any(dim=-1)  # [batch, seq_len]
            token_mask_flat = token_mask.view(-1)  # [batch*seq_len]

            if token_mask_flat.sum() > 0:
                # 提取需要使用该专家的token
                selected_tokens = x_flat[token_mask_flat]  # [num_selected, dim]

                # 批量通过专家网络
                expert_output = self.experts[expert_idx](selected_tokens)  # [num_selected, dim]

                # 获取对应的权重
                # 需要找到每个token对该专家的权重
                expert_weights = torch.where(
                    mask_expert,
                    weights,
                    torch.zeros_like(weights)
                ).sum(dim=-1)  # [batch, seq_len]
                expert_weights_flat = expert_weights.view(-1)[token_mask_flat]  # [num_selected]

                # 加权累加
                moe_output_flat[token_mask_flat] += expert_weights_flat.unsqueeze(1) * expert_output

        # 5. 恢复形状
        moe_output = moe_output_flat.view(batch_size, seq_len, self.dim)

        # 6. 残差连接和归一化
        output = self.norm(x + self.dropout(moe_output))

        return output


class CrossAttention(nn.Module):
    """交叉注意力模块（保留原来的CAtt实现）"""

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.nhead = n_heads
        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)

        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, y, y_mask=None):
        """
        x: query [batch, seq_len, dim]
        y: key & value [batch, seq_len, dim]
        """
        n_batches = x.size(0)

        v = self.linear_v(y).view(
            n_batches, -1, self.nhead, self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(y).view(
            n_batches, -1, self.nhead, self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(x).view(
            n_batches, -1, self.nhead, self.hidden_size_head
        ).transpose(1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        if y_mask is not None:
            scores = scores.masked_fill(y_mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        atted = torch.matmul(att_map, v)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches, -1, self.hid_dim
        )
        atted = self.linear_merge(atted)

        output = self.norm(x + self.dropout(atted))

        return output


class Datt(nn.Module):
    """MoE自注意力 + 交叉注意力的混合架构（与deep_attention.Datt接口一致）"""

    def __init__(self, dim, nhead=4, dropout=0.2, num_experts=8, top_k=2):
        super().__init__()
        # MoE自注意力（替换原来的SAtt）
        self.self_att = MoESelfAttention(dim, nhead, dropout, num_experts, top_k)
        # 交叉注意力（保留原来的CAtt）
        self.cross_att = CrossAttention(dim, nhead, dropout)

        self.drug_norm = nn.LayerNorm(dim)
        self.protein_norm = nn.LayerNorm(dim)

    def forward(self, drug_vector, protein_vector):
        """
        drug_vector: [batch, 1, dim]
        protein_vector: [batch, 1, dim]
        返回: (drug_covector, protein_covector)
        """
        # 1. MoE自注意力增强（各自内部特征提取）
        drug_vector = self.self_att(drug_vector, None)
        protein_vector = self.self_att(protein_vector, None)

        # 2. 交叉注意力（药物-蛋白质交互）
        drug_covector = self.cross_att(drug_vector, protein_vector, None)
        protein_covector = self.cross_att(protein_vector, drug_vector, None)

        return drug_covector, protein_covector

