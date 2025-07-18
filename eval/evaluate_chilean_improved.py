# Chilean Underground Mine Dataset Evaluation - Improved Version
# Evaluation using Chilean dataset protocol with enhanced metrics

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.chilean.chilean_raw import ChileanPointCloudLoader


def evaluate_chilean(model, device, params: TrainingParams, log: bool = False, show_progress: bool = False):
    """运行Chilean数据集的评估"""

    eval_database_files = ['chilean_evaluation_database.pickle']
    eval_query_files = ['chilean_evaluation_query_1.pickle', 'chilean_evaluation_query_2.pickle']

    stats = {}

    # 加载数据库集
    database_file = eval_database_files[0]
    p = os.path.join(params.dataset_folder, database_file)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)

    # 分别评估两个查询集
    for query_file in eval_query_files:
        # 从文件名提取查询集名称
        query_name = query_file.split('_')[-1].split('.')[0]  # query_1 或 query_2

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_chilean_dataset(model, device, params, database_sets, query_sets,
                                        log=log, show_progress=show_progress)
        stats[f'chilean_{query_name}'] = temp

    return stats


def evaluate_chilean_dataset(model, device, params: TrainingParams, database_sets, query_sets,
                             log: bool = False, show_progress: bool = False):
    """运行单个Chilean数据集的评估"""

    model.eval()

    # 计算数据库嵌入
    print("Computing database embeddings...")
    database_embeddings = get_latent_vectors_chilean(model, database_sets[0], device, params, show_progress)

    # 计算查询嵌入
    print("Computing query embeddings...")
    query_embeddings = get_latent_vectors_chilean(model, query_sets[0], device, params, show_progress)

    # 计算增强的召回率指标
    enhanced_stats = get_enhanced_recall_chilean(database_embeddings, query_embeddings,
                                                 query_sets[0], database_sets[0], log=log)

    return enhanced_stats


def get_latent_vectors_chilean(model, dataset_dict, device, params: TrainingParams, show_progress: bool = False):
    """获取Chilean数据集的潜在向量"""

    if params.debug:
        embeddings = np.random.rand(len(dataset_dict), 256)
        return embeddings

    pc_loader = ChileanPointCloudLoader()
    model.eval()
    embeddings = None

    for i, elem_ndx in enumerate(tqdm.tqdm(dataset_dict, disable=not show_progress, desc='Computing embeddings')):
        # Chilean数据集中直接存储完整路径
        pc_file_path = dataset_dict[elem_ndx]["query"]

        # 如果路径不是绝对路径，则与数据集文件夹拼接
        if not os.path.isabs(pc_file_path):
            pc_file_path = os.path.join(params.dataset_folder, pc_file_path)

        pc = pc_loader(pc_file_path)

        if len(pc) == 0:
            print(f"Warning: Empty point cloud for {pc_file_path}")
            # 创建一个虚拟的嵌入向量
            embedding = np.zeros((1, 256), dtype=np.float32)
        else:
            pc = torch.tensor(pc, dtype=torch.float32)
            embedding = compute_embedding_chilean(model, pc, device, params)

        if embeddings is None:
            embeddings = np.zeros((len(dataset_dict), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding_chilean(model, pc, device, params: TrainingParams):
    """计算Chilean点云的嵌入向量"""
    coords, _ = params.model_params.quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # 计算全局描述符
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def get_enhanced_recall_chilean(database_vectors, query_vectors, query_set, database_set, log=False):
    """
    计算Chilean数据集的增强召回率指标
    包括Recall@K, Precision@K, F1@K
    """

    # 建立KDTree用于最近邻搜索
    database_nbrs = KDTree(database_vectors)

    # 计算K值范围：从1到数据库大小的1%
    database_size = len(database_vectors)
    max_k = max(int(database_size * 0.01), 1)
    k_values = list(range(1, max_k + 1))

    print(f"Evaluating with K from 1 to {max_k} (1% of database size: {database_size})")

    # 初始化指标数组
    recall_at_k = np.zeros(max_k)  # Binary Recall@K
    precision_at_k = np.zeros(max_k)  # Precision@K
    f1_at_k = np.zeros(max_k)  # F1@K (将在最后计算)

    # 传统指标（保持兼容性）
    traditional_recall = np.zeros(25)
    one_percent_retrieved = 0
    traditional_threshold = max(int(round(database_size / 100.0)), 1)

    num_evaluated = 0
    all_query_precisions = [[] for _ in range(max_k)]  # 存储每个查询的precision用于计算平均值

    print("Computing enhanced metrics...")
    for i in tqdm.tqdm(range(len(query_vectors)), desc="Processing queries"):
        # 检查查询是否存在
        if i not in query_set:
            continue

        query_details = query_set[i]

        # Chilean数据集中正样本索引存储在键'0'中
        if 0 not in query_details:
            continue
        true_neighbors = query_details[0]
        if len(true_neighbors) == 0:
            continue

        num_evaluated += 1
        true_neighbors_set = set(true_neighbors)

        # 找到最近邻 - 需要足够多的邻居用于评估
        k_needed = min(max_k, database_size)
        distances, indices = database_nbrs.query(np.array([query_vectors[i]]), k=k_needed)
        retrieved_indices = indices[0]

        # 对每个K值计算指标
        for k_idx, k in enumerate(k_values):
            if k <= len(retrieved_indices):
                # 前K个检索结果
                top_k_indices = retrieved_indices[:k]
                top_k_set = set(top_k_indices)

                # 计算true positives
                true_positives = len(top_k_set & true_neighbors_set)

                # Binary Recall@K: 前K个结果中是否包含至少一个正样本
                if true_positives > 0:
                    recall_at_k[k_idx] += 1

                # Precision@K: 前K个结果中正样本的比例
                precision_k = true_positives / k
                all_query_precisions[k_idx].append(precision_k)
            else:
                # 如果K大于可用的检索结果数量，precision为0
                all_query_precisions[k_idx].append(0.0)

        # 传统指标计算（保持兼容性）
        for j in range(min(25, len(retrieved_indices))):
            if retrieved_indices[j] in true_neighbors_set:
                traditional_recall[j] += 1
                break

        # 计算传统的1% recall
        top_1_percent = retrieved_indices[:traditional_threshold]
        if len(set(top_1_percent) & true_neighbors_set) > 0:
            one_percent_retrieved += 1

        # 记录日志（如果需要）
        if log:
            log_search_results(query_details, retrieved_indices, true_neighbors_set,
                               database_set, distances, k_values[:5])  # 只记录前5个K值

    # 计算最终指标
    if num_evaluated > 0:
        # Binary Recall@K (平均)
        recall_at_k = recall_at_k / num_evaluated

        # Precision@K (平均)
        for k_idx in range(max_k):
            if all_query_precisions[k_idx]:
                precision_at_k[k_idx] = np.mean(all_query_precisions[k_idx])
            else:
                precision_at_k[k_idx] = 0.0

        # F1@K计算
        for k_idx in range(max_k):
            recall_k = recall_at_k[k_idx]
            precision_k = precision_at_k[k_idx]

            if recall_k + precision_k > 0:
                f1_at_k[k_idx] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_at_k[k_idx] = 0.0

        # 传统指标
        traditional_recall = traditional_recall / num_evaluated
        traditional_one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
    else:
        traditional_recall = np.zeros(25)
        traditional_one_percent_recall = 0.0

    # 打印详细结果
    print_detailed_results(k_values, recall_at_k, precision_at_k, f1_at_k, num_evaluated)

    # 返回增强的统计信息
    enhanced_stats = {
        # 传统指标（保持兼容性）
        'ave_recall': traditional_recall,
        'ave_one_percent_recall': traditional_one_percent_recall,

        # 新增的详细指标
        'k_values': k_values,
        'recall_at_k': recall_at_k,
        'precision_at_k': precision_at_k,
        'f1_at_k': f1_at_k,
        'max_k': max_k,
        'num_evaluated': num_evaluated,
        'database_size': database_size,

        # 关键指标摘要
        'recall_at_1': recall_at_k[0] if len(recall_at_k) > 0 else 0.0,
        'precision_at_1': precision_at_k[0] if len(precision_at_k) > 0 else 0.0,
        'f1_at_1': f1_at_k[0] if len(f1_at_k) > 0 else 0.0,
        'mean_recall': np.mean(recall_at_k) if len(recall_at_k) > 0 else 0.0,
        'mean_precision': np.mean(precision_at_k) if len(precision_at_k) > 0 else 0.0,
        'mean_f1': np.mean(f1_at_k) if len(f1_at_k) > 0 else 0.0,
    }

    return enhanced_stats


def log_search_results(query_details, retrieved_indices, true_neighbors_set, database_set, distances, k_values_to_log):
    """记录搜索结果到日志文件"""
    s = f"{query_details['query']}, {query_details.get('x', 0)}, {query_details.get('y', 0)}, {query_details.get('z', 0)}"

    for k in k_values_to_log:
        if k <= len(retrieved_indices):
            for idx in range(k):
                if idx < len(retrieved_indices):
                    is_match = retrieved_indices[idx] in true_neighbors_set
                    e_ndx = retrieved_indices[idx]
                    if e_ndx in database_set:
                        e = database_set[e_ndx]
                        e_emb_dist = distances[0][idx] if idx < len(distances[0]) else 0.0
                        # 计算3D欧几里得距离
                        if all(key in query_details for key in ['x', 'y', 'z']) and \
                                all(key in e for key in ['x', 'y', 'z']):
                            world_dist = np.sqrt((query_details['x'] - e['x']) ** 2 +
                                                 (query_details['y'] - e['y']) ** 2 +
                                                 (query_details['z'] - e['z']) ** 2)
                        else:
                            world_dist = 0.0
                        s += f", {e['query']}, {e_emb_dist:0.2f}, {world_dist:0.2f}, {1 if is_match else 0}"
                    else:
                        s += f", Unknown, 0.0, 0.0, 0"

    s += '\n'

    out_file_name = "chilean_search_results_enhanced.txt"
    with open(out_file_name, "a") as f:
        f.write(s)


def print_detailed_results(k_values, recall_at_k, precision_at_k, f1_at_k, num_evaluated):
    """打印详细的评估结果"""
    print(f"\n{'=' * 80}")
    print(f"DETAILED EVALUATION RESULTS ({num_evaluated} queries evaluated)")
    print(f"{'=' * 80}")
    print(f"{'K':<4} {'Recall@K':<10} {'Precision@K':<12} {'F1@K':<10}")
    print(f"{'-' * 40}")

    for i, k in enumerate(k_values):
        print(f"{k:<4} {recall_at_k[i]:<10.4f} {precision_at_k[i]:<12.4f} {f1_at_k[i]:<10.4f}")

    print(f"{'-' * 40}")
    print(f"{'Mean':<4} {np.mean(recall_at_k):<10.4f} {np.mean(precision_at_k):<12.4f} {np.mean(f1_at_k):<10.4f}")
    print(f"{'=' * 80}")


def print_eval_stats_chilean(stats):
    """打印Chilean数据集的评估统计（增强版）"""
    for dataset_name in stats:
        print(f'\n{"=" * 60}')
        print(f'Dataset: {dataset_name}')
        print(f'{"=" * 60}')

        dataset_stats = stats[dataset_name]

        # 传统指标
        print(f'Traditional Metrics:')
        print(f'  Avg. top 1% recall: {dataset_stats["ave_one_percent_recall"]:.2f}%')

        # 关键新指标
        print(f'\nKey Enhanced Metrics:')
        print(f'  Recall@1: {dataset_stats.get("recall_at_1", 0):.4f}')
        print(f'  Precision@1: {dataset_stats.get("precision_at_1", 0):.4f}')
        print(f'  F1@1: {dataset_stats.get("f1_at_1", 0):.4f}')

        print(f'\nOverall Performance:')
        print(f'  Mean Recall@K: {dataset_stats.get("mean_recall", 0):.4f}')
        print(f'  Mean Precision@K: {dataset_stats.get("mean_precision", 0):.4f}')
        print(f'  Mean F1@K: {dataset_stats.get("mean_f1", 0):.4f}')

        print(f'\nEvaluation Details:')
        print(f'  Database size: {dataset_stats.get("database_size", 0)}')
        print(f'  Max K evaluated: {dataset_stats.get("max_k", 0)}')
        print(f'  Queries evaluated: {dataset_stats.get("num_evaluated", 0)}')


def chilean_write_eval_stats(file_name, prefix, stats):
    """将Chilean评估统计写入文件（增强版）"""
    with open(file_name, "a") as f:
        for dataset_name in stats:
            dataset_stats = stats[dataset_name]

            # 写入传统指标（保持兼容性）
            ave_1p_recall = dataset_stats.get('ave_one_percent_recall', 0)
            ave_recall_1 = dataset_stats.get('ave_recall', [0])[0] if len(
                dataset_stats.get('ave_recall', [])) > 0 else 0

            traditional_line = f"{prefix}, {dataset_name}, {ave_1p_recall:.2f}, {ave_recall_1:.2f}\n"
            f.write(traditional_line)

            # 写入增强指标
            enhanced_line = f"{prefix}_enhanced, {dataset_name}, "
            enhanced_line += f"R@1:{dataset_stats.get('recall_at_1', 0):.4f}, "
            enhanced_line += f"P@1:{dataset_stats.get('precision_at_1', 0):.4f}, "
            enhanced_line += f"F1@1:{dataset_stats.get('f1_at_1', 0):.4f}, "
            enhanced_line += f"Mean_R:{dataset_stats.get('mean_recall', 0):.4f}, "
            enhanced_line += f"Mean_P:{dataset_stats.get('mean_precision', 0):.4f}, "
            enhanced_line += f"Mean_F1:{dataset_stats.get('mean_f1', 0):.4f}, "
            enhanced_line += f"Max_K:{dataset_stats.get('max_k', 0)}, "
            enhanced_line += f"DB_Size:{dataset_stats.get('database_size', 0)}\n"
            f.write(enhanced_line)


def save_detailed_metrics(stats, output_file="chilean_detailed_metrics.txt"):
    """保存详细的K值指标到文件"""
    with open(output_file, "w") as f:
        f.write("Chilean Underground Mine Dataset - Detailed Evaluation Metrics\n")
        f.write("=" * 80 + "\n\n")

        for dataset_name in stats:
            dataset_stats = stats[dataset_name]
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 50 + "\n")

            k_values = dataset_stats.get('k_values', [])
            recall_at_k = dataset_stats.get('recall_at_k', [])
            precision_at_k = dataset_stats.get('precision_at_k', [])
            f1_at_k = dataset_stats.get('f1_at_k', [])

            f.write("K\tRecall@K\tPrecision@K\tF1@K\n")
            for i, k in enumerate(k_values):
                f.write(f"{k}\t{recall_at_k[i]:.6f}\t{precision_at_k[i]:.6f}\t{f1_at_k[i]:.6f}\n")

            f.write(f"\nSummary Statistics:\n")
            f.write(f"Mean Recall@K: {dataset_stats.get('mean_recall', 0):.6f}\n")
            f.write(f"Mean Precision@K: {dataset_stats.get('mean_precision', 0):.6f}\n")
            f.write(f"Mean F1@K: {dataset_stats.get('mean_f1', 0):.6f}\n")
            f.write(f"Queries Evaluated: {dataset_stats.get('num_evaluated', 0)}\n")
            f.write(f"Database Size: {dataset_stats.get('database_size', 0)}\n")
            f.write(f"Max K: {dataset_stats.get('max_k', 0)}\n\n")


if __name__ == "__main__":
    # 直接设置参数，不使用命令行解析
    class Args:
        def __init__(self):
            self.config = '../config/config_chilean.txt'
            # self.model_config = '../models/minkloc3dv1.txt'
            self.model_config = '../models/minkloc3dv2.txt'
            self.weights = '../weights/model_chilean_MinkLoc_20250718_1408_final.pth'  # 需要根据实际权重文件路径修改
            self.debug = False
            self.log = False


    args = Args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    print("Starting enhanced Chilean dataset evaluation...")
    stats = evaluate_chilean(model, device, params, args.log, show_progress=True)

    # 打印结果
    print_eval_stats_chilean(stats)

    # 保存结果到文本文件
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

    # 保存传统格式和增强格式的结果
    chilean_write_eval_stats("chilean_experiment_results.txt", prefix, stats)

    # 保存详细的K值指标
    save_detailed_metrics(stats, "chilean_detailed_metrics.txt")

    print(f"\nResults saved to:")
    print(f"  - chilean_experiment_results.txt (summary)")
    print(f"  - chilean_detailed_metrics.txt (detailed K-value metrics)")
    if args.log:
        print(f"  - chilean_search_results_enhanced.txt (search logs)")