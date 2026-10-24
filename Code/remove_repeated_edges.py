"""
跨数据集重复边去除预处理 - 完整实现版
用于跨数据库测试验证,防止数据泄露
包含: 1. 过滤DTI数据  2. 构建异质图
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_hetero_graph(database, dti_data=None, dti_path=None):
    """
    构建异质图

    参数:
        database: 数据集名称
        dti_data: DTI DataFrame (可选,如果提供则直接使用)
        dti_path: DTI pkl文件路径 (可选,如果dti_data未提供则从文件加载)

    返回:
        hetero_data: HeteroData 对象
    """
    print(f"🔨 构建 {database} 的异质图...")

    # 加载 DTI 数据
    if dti_data is None:
        if dti_path is None:
            raise ValueError("必须提供 dti_data 或 dti_path")
        with open(dti_path, 'rb') as f:
            dti_data = pickle.load(f)

    # 加载药物和蛋白质特征
    drug_feat_path = f'../Data/drug_features/{database}/drug_features_fused.pkl'
    protein_feat_path = f'../Data/protein_features/{database}/protein_features_fused.pkl'

    if not os.path.exists(drug_feat_path):
        logging.error(f'❌ 药物特征不存在: {drug_feat_path}')
        return None

    if not os.path.exists(protein_feat_path):
        logging.error(f'❌ 蛋白质特征不存在: {protein_feat_path}')
        return None

    with open(drug_feat_path, 'rb') as f:
        drug_features = pickle.load(f)

    with open(protein_feat_path, 'rb') as f:
        protein_features = pickle.load(f)

    print(f"   药物特征: {len(drug_features)} 个, 维度: {drug_features['Fusion_emb'].iloc[0].shape}")
    print(f"   蛋白质特征: {len(protein_features)} 个, 维度: {protein_features['Fusion_emb'].iloc[0].shape}")
    print(f"   DTI 边数: {len(dti_data)}")

    # 创建映射
    drug_mapping = {str(pubchem): idx for idx, pubchem in enumerate(drug_features['PubChem'].astype(str))}
    protein_mapping = {str(uniprot): idx for idx, uniprot in enumerate(protein_features['Uniprot'])}

    # 过滤 DTI 数据,只保留有特征的节点
    dti_filtered = dti_data[
        dti_data['Drug'].astype(str).isin(drug_mapping.keys()) &
        dti_data['Protein'].astype(str).isin(protein_mapping.keys())
    ].reset_index(drop=True)

    print(f"   过滤后 DTI 边数: {len(dti_filtered)}")

    # 构建异质图
    hetero_data = HeteroData()

    # 添加药物节点
    drug_embeddings = np.vstack(drug_features['Fusion_emb'].values)
    hetero_data['drug'].x = torch.FloatTensor(drug_embeddings)
    hetero_data['drug'].num_nodes = len(drug_features)

    # 添加蛋白质节点
    protein_embeddings = np.vstack(protein_features['Fusion_emb'].values)
    hetero_data['protein'].x = torch.FloatTensor(protein_embeddings)
    hetero_data['protein'].num_nodes = len(protein_features)

    # 添加边 (只添加正向边,与原始数据集格式完全一致)
    edge_index_drug_protein = []
    for _, row in dti_filtered.iterrows():
        drug_idx = drug_mapping[str(row['Drug'])]
        protein_idx = protein_mapping[str(row['Protein'])]
        edge_index_drug_protein.append([drug_idx, protein_idx])

    if len(edge_index_drug_protein) > 0:
        edge_index_drug_protein = torch.LongTensor(edge_index_drug_protein).t().contiguous()
        hetero_data['drug', 'interaction', 'protein'].edge_index = edge_index_drug_protein
    else:
        # 如果没有边,创建空边索引
        hetero_data['drug', 'interaction', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)

    print(f"   ✅ 异质图构建完成:")
    print(f"      药物节点: {hetero_data['drug'].num_nodes}")
    print(f"      蛋白质节点: {hetero_data['protein'].num_nodes}")
    print(f"      交互边: {hetero_data['drug', 'interaction', 'protein'].edge_index.shape[1]}")

    return hetero_data

def remove_repeated_edges_and_build_graph(database_train, database_test):
    """
    从训练数据集中去除与测试数据集重叠的边,并构建新的异质图

    参数:
        database_train: 训练数据集名称 (如 'DrugBank')
        database_test: 测试数据集名称 (如 'BIOSNAP')
    """
    print(f"\n{'='*70}")
    print(f"🔄 处理: {database_train} w/o {database_test}")
    print(f"{'='*70}")

    # 1. 加载两个数据集的 DTI 数据
    dti_train_path = f'../Data/DTI/{database_train}/dit_filter.pkl'
    dti_test_path = f'../Data/DTI/{database_test}/dit_filter.pkl'

    if not os.path.exists(dti_train_path):
        logging.error(f'❌ 训练数据不存在: {dti_train_path}')
        return False

    if not os.path.exists(dti_test_path):
        logging.error(f'❌ 测试数据不存在: {dti_test_path}')
        return False

    # 加载数据
    with open(dti_train_path, 'rb') as f:
        dti_train = pickle.load(f)

    with open(dti_test_path, 'rb') as f:
        dti_test = pickle.load(f)

    print(f"📊 训练集原始大小: {len(dti_train)} 条边")
    print(f"📊 测试集大小: {len(dti_test)} 条边")

    # 2. 创建测试集的边集合 (Drug-Protein 对)
    test_edges = set()
    for _, row in dti_test.iterrows():
        edge = (str(row['Drug']), str(row['Protein']))
        test_edges.add(edge)

    print(f"🔍 测试集唯一边数: {len(test_edges)}")

    # 3. 从训练集中过滤掉与测试集重叠的边
    mask = ~dti_train.apply(
        lambda row: (str(row['Drug']), str(row['Protein'])) in test_edges,
        axis=1
    )

    dti_train_filtered = dti_train[mask].reset_index(drop=True)

    removed_count = len(dti_train) - len(dti_train_filtered)
    print(f"⚠️  移除重叠边: {removed_count} 条")
    print(f"✅ 过滤后训练集: {len(dti_train_filtered)} 条边")

    # 4. 保存过滤后的 DTI 数据
    output_dir_dti = f'../Data/DTI/{database_train}'
    os.makedirs(output_dir_dti, exist_ok=True)

    dti_output_path = f'{output_dir_dti}/dit_filter_without_{database_test.lower()}.pkl'
    with open(dti_output_path, 'wb') as f:
        pickle.dump(dti_train_filtered, f)

    print(f"💾 DTI 已保存到: {dti_output_path}")

    # 5. 构建新的异质图
    # 5. 构建新的异质图
    try:
        hetero_data = build_hetero_graph(database_train, dti_data=dti_train_filtered)

        if hetero_data is not None:
            # 保存到新文件夹,结构与 BINDINGDB 一致
            graph_output_dir = f'../Data/{database_train.upper()}_WO_{database_test.upper()}'
            os.makedirs(graph_output_dir, exist_ok=True)

            # 1. 保存异质图 (与 BINDINGDB 格式一致)
            graph_filename = f'hetero_data_{database_train.lower()}_wo_{database_test.lower()}.pt'
            graph_output_path = f'{graph_output_dir}/{graph_filename}'
            torch.save(hetero_data, graph_output_path)

            # 2. 保存 DTI 数据 (与 BINDINGDB 格式一致)
            dti_filename = f'dti_{database_train.lower()}_wo_{database_test.lower()}.pkl'
            dti_output_path = f'{graph_output_dir}/{dti_filename}'
            with open(dti_output_path, 'wb') as f:
                pickle.dump(dti_train_filtered, f)

            # 3. 保存药物映射 (JSON格式,与 BINDINGDB 一致)
            drug_feat_path = f'../Data/drug_features/{database_train}/drug_features_fused.pkl'
            protein_feat_path = f'../Data/protein_features/{database_train}/protein_features_fused.pkl'

            with open(drug_feat_path, 'rb') as f:
                drug_features = pickle.load(f)
            with open(protein_feat_path, 'rb') as f:
                protein_features = pickle.load(f)

            drug_mapping = {str(pubchem): idx for idx, pubchem in enumerate(drug_features['PubChem'].astype(str))}
            protein_mapping = {str(uniprot): idx for idx, uniprot in enumerate(protein_features['Uniprot'])}

            # 保存为 JSON 格式
            import json
            drug_mapping_path = f'{graph_output_dir}/drug_mapping.json'
            with open(drug_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(drug_mapping, f, indent=2, ensure_ascii=False)

            protein_mapping_path = f'{graph_output_dir}/protein_mapping.json'
            with open(protein_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(protein_mapping, f, indent=2, ensure_ascii=False)

            # 4. 保存信息文件 (与 BINDINGDB 的 info_dti_*.out 一致)
            info_path = f'{graph_output_dir}/info_dti_{database_train.lower()}_wo_{database_test.lower()}.out'
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset: {database_train} without {database_test}\n")
                f.write(f"Drugs: {len(drug_mapping)}\n")
                f.write(f"Proteins: {len(protein_mapping)}\n")
                f.write(f"Edges: {len(dti_train_filtered)}\n")
                f.write(f"Removed edges: {removed_count}\n")

            print(f"💾 异质图已保存到: {graph_output_path}")
            print(f"💾 DTI 数据已保存到: {dti_output_path}")
            print(f"💾 药物映射已保存到: {drug_mapping_path}")
            print(f"💾 蛋白映射已保存到: {protein_mapping_path}")
            print(f"💾 信息文件已保存到: {info_path}")
    except Exception as e:
        logging.error(f'❌ 构建异质图失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False
    # ... existing code ...

    # 6. 统计信息
    train_drugs = set(dti_train_filtered['Drug'].astype(str))
    train_proteins = set(dti_train_filtered['Protein'].astype(str))
    test_drugs = set(dti_test['Drug'].astype(str))
    test_proteins = set(dti_test['Protein'].astype(str))

    overlap_drugs = train_drugs & test_drugs
    overlap_proteins = train_proteins & test_proteins

    print(f"\n📈 统计信息:")
    print(f"   训练集药物数: {len(train_drugs)}")
    print(f"   训练集蛋白数: {len(train_proteins)}")
    print(f"   与测试集重叠药物: {len(overlap_drugs)} ({len(overlap_drugs)/len(train_drugs)*100:.1f}%)")
    print(f"   与测试集重叠蛋白: {len(overlap_proteins)} ({len(overlap_proteins)/len(train_proteins)*100:.1f}%)")
    print(f"   重叠边比例: {removed_count}/{len(dti_train)} ({removed_count/len(dti_train)*100:.2f}%)")

    return True


def main():
    """主函数 - 执行所有数据集组合的重复边去除和异质图构建"""

    datasets = ['DrugBank', 'BIOSNAP', 'BindingDB']

    print(f"开始跨数据集重复边去除和异质图构建")
    print(f"数据集: {datasets}")
    print(f"总共需要执行 {len(datasets) * (len(datasets) - 1)} 次处理\n")

    success_count = 0
    fail_count = 0

    for db_train in datasets:
        for db_test in datasets:
            if db_test != db_train:
                try:
                    result = remove_repeated_edges_and_build_graph(db_train, db_test)
                    if result:
                        logging.info(f'✅ 成功: {db_train} w/o {db_test}')
                        success_count += 1
                    else:
                        logging.error(f'❌ 失败: {db_train} w/o {db_test}')
                        fail_count += 1

                except Exception as e:
                    logging.error(f'💥 异常: {db_train} w/o {db_test} - {str(e)}')
                    import traceback
                    traceback.print_exc()
                    fail_count += 1

    print(f"\n{'='*70}")
    print(f"🎉 处理完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
