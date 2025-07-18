# Chilean dataset utilities
# Adapted dataset utilities for Chilean Underground Mine Dataset

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.chilean.chilean_train import ChileanTrainingDataset, ChileanTrainTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import PointCloudLoader
from datasets.chilean.chilean_raw import ChileanPointCloudLoader


def get_pointcloud_loader_chilean(dataset_type) -> PointCloudLoader:
    """获取Chilean数据集的点云加载器"""
    return ChileanPointCloudLoader()


def make_datasets_chilean(params: TrainingParams, validation: bool = True):
    """创建Chilean数据集的训练和验证数据集"""
    datasets = {}
    set_transform = TrainSetTransform(params.set_aug_mode)

    # Chilean数据集有自己的transform
    train_transform = ChileanTrainTransform(params.aug_mode)

    # 创建训练数据集
    datasets['train'] = ChileanTrainingDataset(
        params.dataset_folder,
        params.train_file,
        transform=train_transform,
        set_transform=set_transform
    )

    # 创建验证数据集（如果存在）
    if validation and params.val_file is not None:
        datasets['val'] = ChileanTrainingDataset(
            params.dataset_folder,
            params.val_file,
            transform=None,  # 验证时不使用数据增强
            set_transform=None
        )

    return datasets


def make_collate_fn_chilean(dataset: TrainingDataset, quantizer, batch_split_size=None):
    """为Chilean数据集创建collate函数"""

    def collate_fn(data_list):
        # 构建批次对象
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            # 对所有数据集元素应用相同的变换
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.set_transform(clouds)
            clouds = clouds.split(lens)

        # 计算正样本和负样本掩码
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in
                          labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # 转换为极坐标（当使用极坐标时）并量化
        coords = [quantizer(e)[0] for e in clouds]

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            # 为每个点分配虚拟特征=1
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}
        else:
            # 将批次分割成多个较小的块
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders_chilean(params: TrainingParams, validation=True):
    """
    创建Chilean数据集的训练和验证数据加载器
    """
    datasets = make_datasets_chilean(params, validation=validation)

    dataloaders = {}
    train_sampler = BatchSampler(
        datasets['train'],
        batch_size=params.batch_size,
        batch_size_limit=params.batch_size_limit,
        batch_expansion_rate=params.batch_expansion_rate
    )

    # Collate函数将项目整理成批次并在整个批次上应用'set transform'
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn_chilean(datasets['train'], quantizer, params.batch_split_size)

    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=params.num_workers,
        pin_memory=True
    )

    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn_chilean(datasets['val'], quantizer, params.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)

        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True
        )

    return dataloaders


def filter_query_elements_chilean(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                                  dist_threshold: float) -> List[EvaluationTuple]:
    """
    Chilean数据集专用的查询元素过滤函数
    过滤掉在dist_threshold阈值内没有对应地图元素的查询元素
    """
    # 提取地图位置（使用3D坐标：x, y, z）
    map_pos = np.zeros((len(map_set), 3), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        # Chilean数据集position包含x,y,z坐标
        if len(e.position) >= 3:
            map_pos[ndx] = e.position[:3]
        else:
            # 如果只有2D坐标，z设为0
            map_pos[ndx, :2] = e.position
            map_pos[ndx, 2] = 0

    # 构建3D kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0

    for ndx, e in enumerate(query_set):
        # 处理查询位置
        query_pos = np.zeros(3)
        if len(e.position) >= 3:
            query_pos = e.position[:3]
        else:
            query_pos[:2] = e.position
            query_pos[2] = 0

        position = query_pos.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    """检查元素是否在排序数组中"""
    if len(array) == 0:
        return False
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e