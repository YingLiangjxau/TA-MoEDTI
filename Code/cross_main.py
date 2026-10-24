"""
跨数据集泛化实验
在一个数据集上训练,在另一个数据集上测试
支持命令行参数选择和批量运行
每次运行立即保存结果到 results.txt
"""

import os
import sys
import numpy as np
import logging
import argparse
import time
import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from model import Model, EarlyStopper, shuffle_label_data
from utils import plot_auc
from hll_topology import add_hll_features_to_heterodata


def run_single_experiment(train_database, test_database, hidden_channels=64):
    """
    运行单次跨数据集实验

    返回:
        best_test_auc: 最佳测试AUC
        best_test_aupr: 最佳测试AUPR
        elapsed_time: 耗时(分钟)
        best_epoch: 最佳epoch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== 1. 加载训练数据 ==========
    train_path = f'../Data/{train_database.upper()}/hetero_data_{train_database.lower()}.pt'
    if not os.path.exists(train_path):
        train_path = f'../Data/{train_database.upper()}/hetero_data_{train_database.upper()}.pt'

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据不存在: {train_path}")

    train_data = torch.load(train_path, weights_only=False)
    train_data = train_data.to(device)

    # 添加 HLL 特征
    train_data = add_hll_features_to_heterodata(train_data, p=6)
    train_data = T.ToUndirected()(train_data)

    # 分割训练集和验证集
    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.0,
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )

    train_split, val_split, _ = split(train_data)

    # ========== 2. 加载测试数据 ==========
    test_path = f'../Data/{test_database.upper()}/hetero_data_{test_database.lower()}.pt'
    if not os.path.exists(test_path):
        test_path = f'../Data/{test_database.upper()}/hetero_data_{test_database.upper()}.pt'

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据不存在: {test_path}")

    test_data = torch.load(test_path, weights_only=False)
    test_data = test_data.to(device)
    test_data = add_hll_features_to_heterodata(test_data, p=6)
    test_data = T.ToUndirected()(test_data)

    # ========== 3. 构建模型 ==========
    model = Model(hidden_channels=hidden_channels, data=train_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopper(tolerance=10, min_delta=0.05)

    # Lazy init
    hll_init = {
        'drug': train_split['drug'].hll_feat,
        'protein': train_split['protein'].hll_feat
    }
    with torch.no_grad():
        model.encoder(train_split.x_dict, train_split.edge_index_dict)
        model.decoder(
            {'drug': torch.randn(1, hidden_channels).to(device),
             'protein': torch.randn(1, hidden_channels).to(device)},
            hll_init,
            torch.tensor([[0], [0]]).to(device))

    # ========== 4. 训练函数 ==========
    def train(train_split_data):
        model.train()
        optimizer.zero_grad()
        train_split_data = shuffle_label_data(train_split_data)

        hll_dict = {
            'drug': train_split_data['drug'].hll_feat,
            'protein': train_split_data['protein'].hll_feat
        }

        _, pred = model(train_split_data.x_dict, train_split_data.edge_index_dict,
                        train_split_data['drug', 'protein'].edge_label_index, hll_dict)
        target = train_split_data['drug', 'protein'].edge_label
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        return float(loss)

    # ========== 5. 测试函数 ==========
    @torch.no_grad()
    def test_on_dataset(data):
        model.eval()

        num_pos_edges = data['drug', 'interaction', 'protein'].edge_index.shape[1]
        pos_edge_index = data['drug', 'interaction', 'protein'].edge_index

        num_nodes_drug = data['drug'].num_nodes
        num_nodes_prot = data['protein'].num_nodes

        neg_edge_index = torch.randint(0, num_nodes_drug, (1, num_pos_edges)).to(device)
        neg_edge_index = torch.cat([
            neg_edge_index,
            torch.randint(0, num_nodes_prot, (1, num_pos_edges)).to(device)
        ], dim=0)

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([
            torch.ones(num_pos_edges).to(device),
            torch.zeros(num_pos_edges).to(device)
        ])

        hll_dict = {
            'drug': data['drug'].hll_feat,
            'protein': data['protein'].hll_feat
        }

        emb, pred = model(data.x_dict, data.edge_index_dict, edge_label_index, hll_dict)
        target = edge_label.float()
        loss = criterion(pred, target)

        out = pred.view(-1).sigmoid()
        auc = roc_auc_score(target.cpu().numpy(), out.detach().cpu().numpy())
        aupr = average_precision_score(target.cpu().numpy(), out.detach().cpu().numpy())

        return auc, aupr, loss.cpu().numpy(), emb

    # ========== 6. 训练循环 ==========
    init_time = time.time()
    best_test_auc = 0
    best_test_aupr = 0
    best_epoch = 0

    for epoch in range(1000):
        loss = train(train_split)
        val_auc, val_aupr, val_loss, _ = test_on_dataset(val_split)
        test_auc, test_aupr, test_loss, emb = test_on_dataset(test_data)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_aupr = test_aupr
            best_epoch = epoch

        if early_stopper.early_stop(loss, val_loss) and epoch > 700:
            break

    end_time = time.time()
    elapsed_time = (end_time - init_time) / 60

    return best_test_auc, best_test_aupr, elapsed_time, best_epoch


def cross_dataset_test(train_database, test_database, hidden_channels=64, repeat_times=20):
    """
    跨数据集测试: 在 train_database 上训练,在 test_database 上测试
    自动重复多次实验并保存统计结果

    参数:
        train_database: 训练数据集名称 (如 'DRUGBANK_WO_BIOSNAP')
        test_database: 测试数据集名称 (如 'BIOSNAP')
        hidden_channels: 隐藏层维度
        repeat_times: 重复次数 (None表示使用默认值20)

    返回:
        mean_auc: 平均AUC
        std_auc: AUC标准差
    """
    # 设置默认重复次数
    if repeat_times is None:
        repeat_times = 20

    print(f"\n{'=' * 70}")
    print(f"🔄 跨数据集实验: {train_database} → {test_database}")
    print(f"重复次数: {repeat_times}")
    print(f"{'=' * 70}")

    # 创建结果目录
    OUTPUT_PATH = f'../Results/CROSS_{train_database.upper()}_TO_{test_database.upper()}_{hidden_channels}'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 初始化 results.txt (写入表头)
    results_file = f'{OUTPUT_PATH}/results.txt'
    with open(results_file, 'a') as f:
        f.write(f"Cross-dataset Experiment: {train_database} → {test_database}\n")
        f.write(f"Hidden Channels: {hidden_channels}\n")
        f.write(f"Repeat Times: {repeat_times}\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Run':<6} {'Best AUC':<12} {'Best AUPR':<12} {'Time(min)':<12} {'Epoch':<8}\n")
        f.write(f"{'-'*70}\n")

    # 存储所有运行的结果
    all_aucs = []
    all_auprs = []
    all_times = []
    all_epochs = []

    # 运行多次实验
    for i in range(repeat_times):
        print(f"\n▶️  第 {i+1}/{repeat_times} 次运行...")

        try:
            auc, aupr, elapsed_time, best_epoch = run_single_experiment(
                train_database, test_database, hidden_channels
            )

            all_aucs.append(auc)
            all_auprs.append(aupr)
            all_times.append(elapsed_time)
            all_epochs.append(best_epoch)

            # 立即保存到 results.txt (追加模式)
            with open(results_file, 'a') as f:
                f.write(f"{auc:.4f}\t{aupr:.4f}\t{elapsed_time:.4f}\t{best_epoch}\n")

            print(f"✅ 完成: AUC={auc:.4f}, AUPR={aupr:.4f}, Time={elapsed_time:.2f}min, Epoch={best_epoch}")

        except Exception as e:
            print(f"❌ 第 {i+1} 次运行失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 记录失败
            with open(results_file, 'a') as f:
                f.write(f"FAILED\tFAILED\tERROR\t0\n")

    print(f"\n📈 成功运行次数: {len(all_aucs)}/{repeat_times}")

    # 计算统计信息
    if len(all_aucs) > 0:
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
        mean_aupr = np.mean(all_auprs)
        std_aupr = np.std(all_auprs)
        mean_time = np.mean(all_times)

        # 追加统计总结到 results.txt
        with open(results_file, 'a') as f:
            f.write(f"{'-'*70}\n")
            f.write(f"Mean AUROC:  {mean_auc:.4f} ± {std_auc:.4f}\n")
            f.write(f"Mean AUPRC: {mean_aupr:.4f} ± {std_aupr:.4f}\n")
            f.write(f"Mean Time: {mean_time:.2f} min\n")

        print(f"\n📊 统计结果:")
        print(f"   Mean AUROC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"   Mean AUPRC: {mean_aupr:.4f} ± {std_aupr:.4f}")
        print(f"   Mean Time: {mean_time:.2f} min")
        print(f"💾 结果保存到: {results_file}")

        return mean_auc, std_auc
    else:
        print("\n❌ 所有实验都失败了")
        return None, None


def main(train_db=None, test_db=None, hidden_channels=64, repeat_times=20):
    """
    主函数 - 支持单个实验或批量实验

    参数:
        train_db: 训练数据集 (None表示运行所有)
        test_db: 测试数据集 (None表示运行所有)
        hidden_channels: 隐藏层维度
        repeat_times: 重复次数 (None表示默认20)
    """

    # 定义所有跨数据集实验组合
    all_experiments = [
        ('DRUGBANK_WO_BIOSNAP', 'BIOSNAP'),
        ('DRUGBANK_WO_BINDINGDB', 'BINDINGDB'),
        ('BIOSNAP_WO_DRUGBANK', 'DRUGBANK'),
        ('BIOSNAP_WO_BINDINGDB', 'BINDINGDB'),
        ('BINDINGDB_WO_DRUGBANK', 'DRUGBANK'),
        ('BINDINGDB_WO_BIOSNAP', 'BIOSNAP'),
    ]

    # 如果指定了数据集,运行单个实验
    if train_db and test_db:
        try:
            mean_auc, std_auc = cross_dataset_test(train_db, test_db, hidden_channels, repeat_times)
            if mean_auc is not None:
                print(f"\n🎉 实验完成: {train_db} → {test_db} (Mean AUC: {mean_auc:.4f} ± {std_auc:.4f})")
            return mean_auc
        except Exception as e:
            print(f"\n❌ 实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # 否则运行所有实验
    print(f"开始跨数据集泛化实验")
    print(f"实验组合: {len(all_experiments)} 个")
    print(f"重复次数: {repeat_times if repeat_times else 20} 次")
    print(f"总实验数: {len(all_experiments) * (repeat_times if repeat_times else 20)}\n")

    results = {}

    for train_database, test_database in all_experiments:
        try:
            mean_auc, std_auc = cross_dataset_test(train_database, test_database, hidden_channels, repeat_times)
            if mean_auc is not None:
                results[f"{train_database}→{test_database}"] = (mean_auc, std_auc)
                print(f"✅ 完成: {train_database} → {test_database} (AUC: {mean_auc:.4f} ± {std_auc:.4f})\n")
        except Exception as e:
            print(f"❌ 失败: {train_database} → {test_database}")
            print(f"错误: {str(e)}\n")
            import traceback
            traceback.print_exc()

    # 打印总结
    if results:
        print(f"\n{'=' * 70}")
        print(f"📊 跨数据集实验总结:")
        print(f"{'=' * 70}")
        for exp, (mean_auc, std_auc) in results.items():
            print(f"{exp:40s} AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='跨数据集泛化实验')
    parser.add_argument('-t', '--train', type=str, help='训练数据集名称 (如 DRUGBANK_WO_BIOSNAP)')
    parser.add_argument('-e', '--test', type=str, help='测试数据集名称 (如 BIOSNAP)')
    parser.add_argument('-c', '--channels', type=int, default=64, help='隐藏层维度 (默认: 64)')
    parser.add_argument('-r', '--repeat', type=int, default=None, help='重复次数 (默认: 20)')

    args = parser.parse_args()

    main(train_db=args.train, test_db=args.test, hidden_channels=args.channels, repeat_times=args.repeat)
