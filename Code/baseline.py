import os
import numpy as np
import torch
import logging
import argparse
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm


def load_dataset_features(database):
    """Load node features and edge information from HeteroData"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, '..', 'Data', database.upper(), f'hetero_data_{database.lower()}.pt')

    print(f'Loading data from: {data_path}')
    data = torch.load(data_path)

    # Extract drug and protein features
    drug_features = data['drug'].x.cpu().numpy()
    protein_features = data['protein'].x.cpu().numpy()

    # Get positive edges
    edge_index = data['drug', 'protein'].edge_index.cpu().numpy()

    # Create positive pairs with features
    positive_pairs = []
    for i in tqdm(range(edge_index.shape[1]), desc='Creating positive pairs'):
        drug_idx = edge_index[0, i]
        prot_idx = edge_index[1, i]

        # Concatenate drug and protein features
        pair_feature = np.concatenate([drug_features[drug_idx], protein_features[prot_idx]])
        positive_pairs.append(pair_feature)

    positive_pairs = np.array(positive_pairs)
    positive_labels = np.ones(len(positive_pairs))

    # Create negative pairs by random sampling
    print('Creating negative samples...')
    negative_pairs = []
    n_negatives = len(positive_pairs)

    # Track positive edges for efficient negative sampling
    positive_set = set()
    for i in range(edge_index.shape[1]):
        positive_set.add((edge_index[0, i], edge_index[1, i]))

    # Random negative sampling
    np.random.seed(42)
    n_drugs = data['drug'].num_nodes
    n_proteins = data['protein'].num_nodes

    attempts = 0
    max_attempts = n_negatives * 10

    while len(negative_pairs) < n_negatives and attempts < max_attempts:
        drug_idx = np.random.randint(0, n_drugs)
        prot_idx = np.random.randint(0, n_proteins)

        if (drug_idx, prot_idx) not in positive_set:
            pair_feature = np.concatenate([drug_features[drug_idx], protein_features[prot_idx]])
            negative_pairs.append(pair_feature)
            attempts += 1

    negative_pairs = np.array(negative_pairs[:n_negatives])
    negative_labels = np.zeros(len(negative_pairs))

    # Combine positive and negative samples
    X = np.vstack([positive_pairs, negative_pairs])
    y = np.concatenate([positive_labels, negative_labels])

    print(f'Dataset: {database.upper()}')
    print(f'Total samples: {len(X)}')
    print(f'Positive samples: {np.sum(y == 1)}')
    print(f'Negative samples: {np.sum(y == 0)}')
    print(f'Feature dimension: {X.shape[1]}')

    return X, y


def run_rf_experiment(database, n_estimators=200, max_depth=20, n_runs=20):
    """Run Random Forest experiment with multiple runs"""
    print('=' * 70)
    print(f'Running RF experiment on {database.upper()}')
    print('=' * 70)

    # Load data once
    X, y = load_dataset_features(database)

    auc_scores = []
    aupr_scores = []

    for run in range(n_runs):
        print(f'\n--- Run {run + 1}/{n_runs} ---')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=run, stratify=y
        )

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=run,
            n_jobs=-1,
            class_weight='balanced'
        )

        rf.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = rf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)

        auc_scores.append(auc)
        aupr_scores.append(aupr)

        print(f'Run {run + 1} - AUC: {auc:.4f}, AUPR: {aupr:.4f}')

    # Calculate statistics
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    aupr_mean = np.mean(aupr_scores)
    aupr_std = np.std(aupr_scores)

    print('\n' + '=' * 70)
    print(f'FINAL RESULTS - {database.upper()}')
    print('=' * 70)
    print(f'AUROC: {auc_mean:.4f} ± {auc_std:.4f}')
    print(f'AUPRC: {aupr_mean:.4f} ± {aupr_std:.4f}')
    print('=' * 70)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Results', f'RF_{database.upper()}')
    os.makedirs(output_dir, exist_ok=True)

    # Save results.txt in same format as main experiments
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        for auc, aupr in zip(auc_scores, aupr_scores):
            f.write(f'{auc:.4f}\t{aupr:.4f}\n')

    # Save statistics
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write(f'Database: {database.upper()}\n')
        f.write(f'Model: Random Forest\n')
        f.write(f'Parameters: n_estimators={n_estimators}, max_depth={max_depth}\n')
        f.write(f'Number of runs: {n_runs}\n\n')
        f.write(f'AUROC: {auc_mean:.4f} ± {auc_std:.4f}\n')
        f.write(f'AUPRC: {aupr_mean:.4f} ± {aupr_std:.4f}\n')

    # Save detailed scores
    np.save(os.path.join(output_dir, 'auc_scores.npy'), np.array(auc_scores))
    np.save(os.path.join(output_dir, 'aupr_scores.npy'), np.array(aupr_scores))

    print(f'\n✓ Results saved to: {output_dir}')

    return auc_scores, aupr_scores


def main():
    parser = argparse.ArgumentParser(description='Random Forest Baseline for DTI Prediction')
    parser.add_argument('-d', '--database', type=str, choices=['BINDINGDB', 'BIOSNAP', 'DRUGBANK', 'ALL'],
                        default='ALL', help='Dataset to run (default: ALL)')
    parser.add_argument('--n_estimators', type=int, default=200, help='Number of trees (default: 200)')
    parser.add_argument('--max_depth', type=int, default=20, help='Max depth of trees (default: 20)')
    parser.add_argument('--n_runs', type=int, default=20, help='Number of independent runs (default: 20)')

    args = parser.parse_args()

    databases = ['BINDINGDB', 'BIOSNAP', 'DRUGBANK'] if args.database == 'ALL' else [args.database]

    all_results = {}

    for db in databases:
        start_time = time.time()
        auc_scores, aupr_scores = run_rf_experiment(
            db.lower(),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_runs=args.n_runs
        )
        all_results[db] = {'auc': auc_scores, 'aupr': aupr_scores}
        elapsed = time.time() - start_time
        print(f'\n⏱️  {db} completed in {elapsed / 60:.2f} minutes')

    # Print comparison table
    print('\n' + '=' * 80)
    print('SUMMARY COMPARISON')
    print('=' * 80)
    for db, results in all_results.items():
        auc_mean = np.mean(results['auc'])
        auc_std = np.std(results['auc'])
        aupr_mean = np.mean(results['aupr'])
        aupr_std = np.std(results['aupr'])
        print(f'{db:12s} | AUROC: {auc_mean:.4f} ± {auc_std:.4f} | AUPRC: {aupr_mean:.4f} ± {aupr_std:.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
