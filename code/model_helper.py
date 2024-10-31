import logging
import numpy as np
import torch
import torch.nn as nn
import utils
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from data_helper import BiDataloader, EvalDataset
from os.path import join
import dgl


def clip_parameter(model: nn.Module, p_name):
    for n, p in model.named_parameters():
        if n == p_name:
            p.data.clamp_(min=0, max=10)


def train_step(model, data, g, edge_weight, optimizer, scheduler):
    """
    A single train step. Apply back-propation and return the loss
    :param model:
    :param data: batch_data
    :param g: train集合构成的整张大图, 用于聚合节点和边
    :param edge_weight: train集合的边权重
    :param optimizer:
    :param scheduler:
    :return:
    """

    cfg = utils.get_global_config()
    device = torch.device(cfg.device)
    model.train()
    optimizer.zero_grad()

    (src, rel, _), label, rm_edges = data
    src, rel, label = src.to(device), rel.to(device), label.to(device)
    # if cfg.graph == 'kg' and cfg.rm_rate > 0:
    #     rm_edges = rm_edges.to(device)
    #     g.remove_edges(rm_edges)
    score, agg_ent_emb, agg_rel_emb = model(src, rel, g, edge_weight)
    loss = model.loss(score, label)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    log = {
        'loss': loss.item()
    }

    return log


def evaluate(model, d_flag, g, edge_weight, record=False, do_round=False) -> dict:
    """
    Evaluate the dataset.
    :param model:
    :param d_flag: dataset标识符
    :param g: 需要聚合embedding的图
    :param edge_weight: 图中边的权重
    :param record: 是否需要记录所有数据的rank值
    :param do_round: 是否需要做取2位小数点处理
    :return:
    """
    # assert data_flag in ['train', 'dev', 'test']
    model.eval()
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    device = torch.device(cfg.device)

    eval_h_loader = DataLoader(
        dataset=EvalDataset(d_flag, 'tail_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    )
    eval_t_loader = DataLoader(
        EvalDataset(d_flag, 'head_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    )
    eval_loader = BiDataloader(eval_h_loader, eval_t_loader)

    head_metric, tail_metric = {}, {}
    metrics = {
        'head_batch': head_metric,
        'tail_batch': tail_metric,
        'ranking': []
    }
    hits_range = [1, 3, 10, 100, 1000, round(0.5*n_ent)]
    with torch.no_grad():
        ent_emb, rel_emb = model.aggragate_emb(g, edge_weight)
        for i, data in enumerate(eval_loader):
            # if i == 6: break
            # filter_bias: (bs, n_ent)
            (src, rel, dst), filter_bias, mode = data
            src, rel, dst, filter_bias = src.to(device), rel.to(device), dst.to(device), filter_bias.to(device)
            # (bs, n_ent)
            score = model.predictor(ent_emb[src], rel_emb[rel], ent_emb)
            score = score + filter_bias

            pos_inds = dst
            batch_size = filter_bias.shape[0]
            pos_score = score[torch.arange(batch_size), pos_inds].unsqueeze(dim=1)
            # 正例和负例做比较, 看有多少负例比正例大, 即可知道正例的排名
            # 考虑分值相等的情况, 取排名上界和下界的平均值
            compare_up = torch.gt(score, pos_score)  # (B,N)
            compare_low = torch.ge(score, pos_score)
            ranking_up = compare_up.to(dtype=torch.float).sum(dim=1) + 1  # (bs, )
            ranking_low = compare_low.to(dtype=torch.float).sum(dim=1)  # 默认是带上其自身的，无需+1
            ranking = (ranking_up + ranking_low) / 2
            # ranking = ranking_up
            # if i == 0:
            #     logging.info('Ranking: {}'.format(ranking.tolist()[96:128]))
            if record:
                rank = torch.stack([src, rel, dst, ranking], dim=1)  # (bs, 4)
                metrics['ranking'].append(rank)

            results = metrics[mode]
            results['MR'] = results.get('MR', 0.) + ranking.sum().item()
            results['MRR'] = results.get('MRR', 0.) + (1 / ranking).sum().item()
            for k in hits_range:
                results['HITS@{}'.format(k)] = results.get('HITS@{}'.format(k), 0.) + \
                                               (ranking <= k).to(torch.float).sum().item()
            results['n_data'] = results.get('n_data', 0) + batch_size

        assert metrics['head_batch']['n_data'] == metrics['tail_batch']['n_data']
        # 得出最终各指标值
        for k, results in metrics.items():
            if k in ['ranking']:
                continue
            results['MR'] /= results['n_data']
            results['MRR'] /= results['n_data']
            for j in hits_range:
                results['HITS@{}'.format(j)] /= results['n_data']
        if record:
            metrics['ranking'] = torch.cat(metrics['ranking'], dim=0).tolist()
            metrics['ranking'] = sorted(metrics['ranking'], key=lambda x: x[3], reverse=True)
        if do_round:
            for k, results in metrics.items():
                if k in ['ranking']:
                    continue
                results['MR'] = round(results['MR'], 2)
                results['MRR'] = round(results['MRR'], 2)
                for j in hits_range:
                    results['HITS@{}'.format(j)] = round(results['HITS@{}'.format(j)], 2)
    return metrics

