# Chilean Underground Mine Dataset Evaluation - Fixed Version
# Fixed 1% recall calculation bug

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
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []

    model.eval()

    # 计算数据库嵌入
    print("Computing database embeddings...")
    database_embeddings = get_latent_vectors_chilean(model, database_sets[0], device, params, show_progress)

    # 计算查询嵌入
    print("Computing query embeddings...")
    query_embeddings = get_latent_vectors_chilean(model, query_sets[0], device, params, show_progress)

    # 计算召回率
    pair_recall, pair_opr = get_recall_chilean(database_embeddings, query_embeddings,
                                               query_sets[0], database_sets[0], log=log)
    recall = np.array(pair_recall)
    one_percent_recall = pair_opr

    ave_recall = recall
    ave_one_percent_recall = one_percent_recall
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


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


def get_recall_chilean(database_vectors, query_vectors, query_set, database_set, log=False):
    """计算Chilean数据集的召回率 - 修复1% recall计算bug"""

    # 当嵌入归一化时，使用欧几里得距离与使用余弦距离给出相同的最近邻搜索结果
    database_nbrs = KDTree(database_vectors)

    # 计算需要的最大K值
    threshold = max(int(round(len(database_vectors) / 100.0)), 1)
    max_k_needed = max(25, threshold)  # 取25和1%阈值的最大值

    print(f"Database size: {len(database_vectors)}")
    print(f"1% threshold: {threshold}")
    print(f"Max K needed: {max_k_needed}")

    recall = [0] * 25  # 保持原有25个位置的recall
    one_percent_retrieved = 0
    num_evaluated = 0

    for i in range(len(query_vectors)):
        # i是查询元素索引
        if i not in query_set:
            continue

        query_details = query_set[i]  # {'query': path, 'x': , 'y': , 'z': }

        # Chilean数据集中正样本索引存储在键'0'中
        if 0 not in query_details:
            continue
        true_neighbors = query_details[0]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # 检索足够多的邻居 - 修复bug的关键
        k_to_retrieve = min(max_k_needed, len(database_vectors))
        distances, indices = database_nbrs.query(
            np.array([query_vectors[i]]),
            k=k_to_retrieve
        )

        # 计算Recall@1到Recall@25（保持原有逻辑）
        for j in range(min(25, len(indices[0]))):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        # 正确计算1% recall
        if len(indices[0]) >= threshold:
            top_1_percent = set(indices[0][:threshold])
            if len(top_1_percent.intersection(set(true_neighbors))) > 0:
                one_percent_retrieved += 1
        else:
            # 如果检索结果少于1%阈值，检查全部结果
            if len(set(indices[0]).intersection(set(true_neighbors))) > 0:
                one_percent_retrieved += 1

        if log:
            # 记录Chilean数据集的搜索结果
            s = f"{query_details['query']}, {query_details.get('x', 0)}, {query_details.get('y', 0)}, {query_details.get('z', 0)}"
            for k in range(min(len(indices[0]), 5)):
                is_match = indices[0][k] in true_neighbors
                e_ndx = indices[0][k]
                if e_ndx in database_set:
                    e = database_set[e_ndx]  # 数据库元素
                    e_emb_dist = distances[0][k]
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

            out_file_name = "chilean_search_results.txt"
            with open(out_file_name, "a") as f:
                f.write(s)

    if num_evaluated > 0:
        one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
        recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    else:
        one_percent_recall = 0
        recall = np.zeros(25)

    return recall, one_percent_recall


def print_eval_stats_chilean(stats):
    """打印Chilean数据集的评估统计"""
    for dataset_name in stats:
        print('Dataset: {}'.format(dataset_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[dataset_name]['ave_one_percent_recall']))
        print(stats[dataset_name]['ave_recall'])


def chilean_write_eval_stats(file_name, prefix, stats):
    """将Chilean评估统计写入文件"""
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []

    # 打印最终模型的结果
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
    # 直接设置参数，不使用命令行解析
    class Args:
        def __init__(self):
            self.config = '../config/config_chilean.txt'
            # self.model_config = '../models/minkloc3dv1.txt'
            self.model_config = '../models/minkloc3dv2.txt'  # 可选择更强的模型
            self.weights = '../weights/model_chilean_MinkLoc_20250718_1514_final.pth'  # 需要根据实际权重文件路径修改
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

    stats = evaluate_chilean(model, device, params, args.log, show_progress=True)
    print_eval_stats_chilean(stats)

    # 保存结果到文本文件
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    chilean_write_eval_stats("chilean_experiment_results.txt", prefix, stats)
