import torch
import numpy as np


def compute_hll_fingerprint_fast(edge_index, num_nodes, p=6):
    """
    快速计算 HyperLogLog 拓扑指纹
    Args:
        edge_index: [2, E]
        num_nodes: 节点数
        p: 精度参数，2^p = 维度 (p=6 -> 64维)
    Returns:
        [num_nodes, 2^p]
    """
    num_buckets = 2 ** p
    src, dst = edge_index

    # 1. 确定性哈希 (使用大素数乘法模拟)
    large_prime = 2654435761
    mod = 2 ** 32
    h_vals = (dst.cpu().numpy() * large_prime) % mod
    h_vals = h_vals.astype(np.uint32)

    # 2. 提取桶索引和秩 (Leading Zeros)
    bucket_indices = h_vals & (num_buckets - 1)
    remaining = h_vals >> p

    rhos = np.zeros_like(h_vals, dtype=np.float32)
    non_zero_mask = remaining != 0
    rhos[non_zero_mask] = (32 - p) - np.floor(np.log2(remaining[non_zero_mask])).astype(np.float32) + 1
    rhos[~non_zero_mask] = 32 - p + 1

    # 3. 聚合 (Scatter Max)
    src_np = src.cpu().numpy()
    all_buckets = np.zeros((num_nodes, num_buckets), dtype=np.float32)

    for b in range(num_buckets):
        mask = (bucket_indices == b)
        if not np.any(mask): continue
        nodes_in_bucket = src_np[mask]
        rhos_in_bucket = rhos[mask]

        idx_tensor = torch.tensor(nodes_in_bucket, dtype=torch.long)
        rho_tensor = torch.tensor(rhos_in_bucket, dtype=torch.float32)

        temp_bucket = torch.zeros(num_nodes, dtype=torch.float32)
        temp_bucket.scatter_reduce_(0, idx_tensor, rho_tensor, reduce="amax", include_self=False)
        all_buckets[:, b] = temp_bucket.numpy()

    return torch.from_numpy(all_buckets)


def add_hll_features_to_heterodata(data, p=6):
    """
    为异质图数据添加 HLL 拓扑特征
    """
    print(f"⚡ Computing HLL Structural Fingerprints (dim={2 ** p})...")

    # Drug 节点的 HLL (基于它连接的 Protein)
    drug_edge = data['drug', 'interaction', 'protein'].edge_index
    data['drug'].hll_feat = compute_hll_fingerprint_fast(drug_edge, data['drug'].num_nodes, p=p)

    # Protein 节点的 HLL (基于连接它的 Drug)
    # 修复：直接反转 drug_edge，避免访问可能不存在的 rev_interaction 属性
    prot_edge = torch.stack([drug_edge[1], drug_edge[0]])
    data['protein'].hll_feat = compute_hll_fingerprint_fast(prot_edge, data['protein'].num_nodes, p=p)

    print(f"✅ HLL Features Added: Drug {data['drug'].hll_feat.shape}, Protein {data['protein'].hll_feat.shape}")
    return data

