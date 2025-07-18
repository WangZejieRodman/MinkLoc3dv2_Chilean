# Warsaw University of Technology
# Train MinkLoc model on Chilean Underground Mine Dataset

import os
import numpy as np
import torch
import tqdm
import pathlib

from misc.utils import TrainingParams, get_datetime
from models.losses.loss import make_losses
from models.model_factory import model_factory
from datasets.chilean.chilean_dataset_utils import make_dataloaders_chilean
from eval.evaluate_chilean import evaluate_chilean, print_eval_stats_chilean, chilean_write_eval_stats


def print_global_stats(phase, stats):
    s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    print(s)


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}

    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        embeddings = y['global']

        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # 防止SparseTensors过度消耗GPU内存

    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    """
    使用多阶段反向传播算法的训练步骤
    适用于Chilean数据集的大批次训练
    """
    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # 阶段1 - 计算每个批次元素的描述符（关闭梯度）
    embeddings_l = []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])

    torch.cuda.empty_cache()

    # 阶段2 - 计算损失相对于嵌入的梯度
    embeddings = torch.cat(embeddings_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad

    # 删除中间值
    embeddings_l, embeddings, y, loss = None, None, None, None

    # 阶段3 - 重新计算带梯度的描述符，并使用缓存的损失梯度计算网络参数的梯度
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['global']
                minibatch_size = len(embeddings)
                # 使用链式法则计算网络参数相对于损失的梯度
                embeddings.backward(gradient=embeddings_grad[i: i + minibatch_size])
                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()

    return stats


def do_train_chilean(params: TrainingParams):
    """Chilean数据集的训练主函数"""

    # 创建模型
    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = 'model_chilean_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # 将模型移动到适当的设备
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    print('Model device: {}'.format(device))

    # 设置数据加载器（使用Chilean特定的数据加载器）
    dataloaders = make_dataloaders_chilean(params)

    loss_fn = make_losses(params)

    # 训练元素
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs + 1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # 大批次分割成多个较小块的多阶段训练方法
        train_step_fn = multistaged_training_step

    print("Chilean地下巷道数据集训练开始...")
    print("=" * 60)

    # 训练统计
    stats = {'train': [], 'eval': []}

    if 'val' in dataloaders:
        phases = ['train', 'val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        print(f"\nEpoch {epoch}/{params.epochs}")
        metrics = {'train': {}, 'val': {}}

        for phase in phases:
            running_stats = []
            count_batches = 0

            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if params.debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    batch_stats['global'] = temp_stats

                except StopIteration:
                    # 当其中一个数据加载器耗尽时终止epoch
                    break

                running_stats.append(batch_stats)

            # 计算该阶段的平均统计
            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats]
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

            # 记录指标
            metrics[phase]['loss1'] = epoch_stats['global']['loss']
            if 'num_non_zero_triplets' in epoch_stats['global']:
                metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

            if 'positive_ranking' in epoch_stats['global']:
                metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

            if 'recall' in epoch_stats['global']:
                metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

            if 'ap' in epoch_stats['global']:
                metrics[phase]['AP'] = epoch_stats['global']['ap']

        # ******* 完成EPOCH *******

        if scheduler is not None:
            scheduler.step()

        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), model_pathname + "_" + str(epoch) + ".pth")

        if params.batch_expansion_th is not None:
            # 基于非零三元组数量的动态批次大小扩展
            le_train_stats = stats['train'][-1]
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()

    print('')

    # 保存最终模型权重
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    # 评估最终模型
    print("开始Chilean数据集最终评估...")
    stats = evaluate_chilean(model, device, params, log=False)
    print_eval_stats_chilean(stats)

    print('训练完成！')

    # 将关键实验指标追加到实验摘要文件
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

    chilean_write_eval_stats("chilean_experiment_results.txt", prefix, stats)


def create_weights_folder():
    """创建保存训练模型权重的文件夹"""
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path