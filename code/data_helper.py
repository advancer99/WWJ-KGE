import hydra
import logging
import os
import time
import numpy as np
import torch
from torch import tensor
import math
from os.path import join
import utils
import json
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import os
from collections import defaultdict
import operator
from itertools import chain
import copy
import dgl
import dgl.function as fn
import math
import pickle


def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print('save done!')

def construct_dict(dir_path):
    """
    加载字典
    :param dir_path:
    :return:
    """
    ent2id, rel2id = dict(), dict()

    # 按照在train, valid, test中出现的顺序排序
    train_path, valid_path, test_path = join(dir_path, 'train.txt'), join(dir_path, 'valid.txt'), join(dir_path, 'test.txt')
    for path in [train_path, valid_path, test_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.split('\t')
                t = t[:-1]  # 去掉\n
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)

    # 按id (即出现次序) 排序
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    
    return ent2id, rel2id


def read_data_segnn(data_flag):
    assert data_flag in [
        'train', 'valid', 'test',
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    cfg = utils.get_global_config()
    dir_p = join(cfg.data_path, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)

    # 加载数据
    if data_flag in ['train', 'valid', 'test']:
        path = join(dir_p, '{}.txt'.format(data_flag))
        file = open(path, 'r', encoding='utf-8')
    elif data_flag == ['train', 'valid']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file = chain(file1, file2)
    elif data_flag == ['train', 'valid', 'test']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        path3 = join(dir_p, 'test.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file3 = open(path3, 'r', encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    # logging.info('---加载{}数据---'.format(data_flag))
    src_list = []
    dst_list = []
    rel_list = []
    # data_index = dict()
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)

    for i, line in enumerate(file):
        h, r, t = line.strip().split('\t')
        h, r, t = ent2id[h], rel2id[r], ent2id[t]
        src_list.append(h)
        dst_list.append(t)
        rel_list.append(r)

        # 添加pos head/tail信息
        pos_tails[(h, r)].add(t)
        pos_heads[(r, t)].add(h)
        pos_rels[(h, t)].add(r)
        pos_rels[(t, h)].add(r+len(rel2id))  # inverse edges

    output_dict = {
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        'pos_tails': pos_tails,
        'pos_heads': pos_heads,
        'pos_rels': pos_rels
    }

    return output_dict


def read_data(data_flag):
    assert data_flag in [
        'train', 'valid', 'test',
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    cfg = utils.get_global_config()
    dir_p = join(cfg.data_path, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)
    #save_pkl(ent2id, '/home/zlx/entity_id.pkl')

    # 加载数据
    if data_flag in ['train', 'valid', 'test']:
        path = join(dir_p, data_flag+'.txt')
        file = open(path, 'r', encoding='utf-8')
    elif data_flag == ['train', 'valid']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file = chain(file1, file2)
    elif data_flag == ['train', 'valid', 'test']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        path3 = join(dir_p, 'test.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file3 = open(path3, 'r', encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    # cfg = utils.get_global_config()
    dataset = cfg.dataset

    # logging.info('---加载{}数据---'.format(data_flag))
    src_list, dst_list, rel_list, triple_list = [], [], [], []
    # data_index = dict()
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)
    tail_part_count = {}
    head_part_count = {}
    pos_rel_count = {}
    neigh_count = {} # 入度出度都包括
    in_rels = defaultdict(set)  # key: ent, value: set of in relations
    out_rels = defaultdict(set)

    for i, line in enumerate(file):
        # if i % 10000 == 0:
        #     logging.info(i)
        # if i == 30000:
        #     break
        src, rel, dst = line.strip().split('\t')
        src, rel, dst = ent2id[src], rel2id[rel], ent2id[dst]
        # print(src, rel, dst, sep=' | ')
        # if i == 5:
        #     exit(1)
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(rel)
        triple_list.append((src, rel, dst))

        # 添加pos head/tail信息
        pos_tails[(src, rel)].add(dst)
        # 这个count可以直接用 len(pos_tails[(src, rel)]) 代替?
        head_part_count[(src, rel)] = head_part_count.setdefault((src, rel), 0) + 1
        pos_heads[(rel, dst)].add(src)
        tail_part_count[(rel, dst)] = tail_part_count.setdefault((rel, dst), 0) + 1
        pos_rels[(src, dst)].add(rel)
        # pos_rels[(dst, src)].add(rel+len(rel2id))
        pos_rel_count[(src, dst)] = pos_rel_count.setdefault((src, dst), 0) + 1
        pos_rel_count[(dst, src)] = pos_rel_count.setdefault((dst, src), 0) + 1
        neigh_count[src] = neigh_count.setdefault(src, 0) + 1
        neigh_count[dst] = neigh_count.setdefault(dst, 0) + 1
        in_rels[dst].add(rel)
        out_rels[src].add(rel)

    output_dict = {
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        'triple_list': triple_list,
        'pos_tails': pos_tails,
        'pos_heads': pos_heads,
        'pos_rels': pos_rels,
        'in_rels': in_rels,
        'out_rels': out_rels,
        'head_count': head_part_count,
        'tail_count': tail_part_count,
        'rel_count': pos_rel_count,
        'neigh_count': neigh_count,
    }

    return output_dict


# 构建Knowledge Graph相关代码

def find_large_deg_data(pos_tails, pos_heads, max_size):
    remove_h_r, remove_r_t = set(), set()
    for (h, r), tails in pos_tails.items():
        if len(tails) >= max_size:
            remove_h_r.add((h, r))
    for (r, t), heads in pos_heads.items():
        if len(heads) >= max_size:
            remove_r_t.add((r, t))
    # print('remove_h_r: {}'.format(remove_h_r))
    # print('remove_r_t: {}'.format(remove_r_t))
    return remove_h_r, remove_r_t


def construct_kg(data_flag='train'):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

    d = read_data(data_flag)
    src_list, dst_list, rel_list = [], [], []

    # remove_h_r, remove_r_t = find_large_deg_data(d['pos_tails'], d['pos_heads'],
    #                                              max_size=cfg.kg_size)

    eid = 0
    hr2eid, rt2eid = defaultdict(list), defaultdict(list)
    # weight = []
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        # 如果query-set的目标太多, 就不交互了
        # if (h, r) in remove_h_r or (r, t) in remove_r_t:
        #     continue
        src_list.extend([h, t])
        dst_list.extend([t, h])
        rel_list.extend([r, r+n_rel])
        hr2eid[(h, r)].extend([eid, eid+1])  # (h, r)数据对应的所有在kg中的边, 包括逆边
        rt2eid[(r, t)].extend([eid, eid+1])  # (t, r+n_rel)对应的在kg中的边, 包括逆边
        # hr2eid[(h, r)].append(eid)  # (h, r)数据对应的所有在kg中的边
        # rt2eid[(r, t)].append(eid+1)  # (t, r+n_rel)对应的在kg中的边
        eid += 2
        # weight.extend([len(d['pos_tails'][(h, r)]), len(d['pos_heads'][(r, t)])])

    src, dst, rel = tensor(src_list), tensor(dst_list), tensor(rel_list)

    # g = dgl.graph((src, dst), num_nodes=n_ent)
    # g.edata['rel_id'] = rel
    #
    # # 确认所有的节点都在agg_graph中出现
    # assert g.number_of_nodes() == n_ent, '# node of graph: {}, # ent: {}'.format(g.number_of_nodes(), n_ent)

    return src, dst, rel, hr2eid, rt2eid


def construct_kg_directed(data_flag='train'):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']

    d = read_data(data_flag)
    src_list, dst_list, rel_list = [], [], []

    remove_h_r, remove_r_t = find_large_deg_data(d['pos_tails'], d['pos_heads'],
                                                 max_size=cfg.kg_size)

    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        # 如果query-set的目标太多, 就不交互了
        if (h, r) in remove_h_r or (r, t) in remove_r_t:
            continue
        src_list.extend([h])
        dst_list.extend([t])
        rel_list.extend([r])

    src, dst, rel = tensor(src_list), tensor(dst_list), tensor(rel_list)

    g = dgl.graph((src, dst), num_nodes=n_ent)
    g.edata['rel_id'] = rel

    # 确认所有的节点都在agg_graph中出现
    assert g.number_of_nodes() == n_ent, '# node of graph: {}, # ent: {}'.format(g.number_of_nodes(), n_ent)

    return src, dst, rel


# 构建Knowledge Similarity Graph相关代码

def filter_pair2w(pair2w, remove_proportion=0.5, flag='total'):
    sorted_pair_list = sorted(pair2w.items(), key=lambda x: x[1])
    if flag == 'total':
        total_num = len(sorted_pair_list)
        remove_num = int(total_num * remove_proportion)
        new_pair2w = [k for k, v in sorted_pair_list[remove_num:]]
        edge_weight = [v for k, v in sorted_pair_list[remove_num:]]
    elif flag == 'node':
        tmp_pair = sorted_pair_list.copy()
        for k, v in tmp_pair:
            sorted_pair_list.append(((k[1], k[0]), v))
        sorted_pair_list = sorted(sorted_pair_list, key=lambda x: x[1])
        sorted_pair_list = sorted(sorted_pair_list, key=lambda x: x[0][1])
        new_pair2w = []
        edge_weight = []
        last_node = -1
        last_edge = []
        last_score = []
        for k, v in sorted_pair_list:
            if k[1] != last_node:
                last_node = k[1]
                # last_mean = np.mean(last_score)
                # last_var = np.std(last_score,ddof=1)
                # lower_bound = last_mean# - 0.5*last_var
                # for j in range(len(last_score)):
                #     if last_score[j] > lower_bound:
                #         new_pair2w.append(last_edge[j])
                remove_num = int(len(last_edge) * remove_proportion)
                new_pair2w.extend(last_edge[remove_num:])
                edge_weight.extend(last_score[remove_num:])
                last_edge = [k]
                last_score = [v]
            else:
                last_edge.append(k)
                last_score.append(v)
        remove_num = int(len(last_edge) * remove_proportion)
        new_pair2w.extend(last_edge[remove_num:])
        edge_weight.extend(last_score[remove_num:])
        # last_mean = np.mean(last_score)
        # last_var = np.std(last_score, ddof=1)
        # lower_bound = last_mean# - 0.5*last_var
        # for j in range(len(last_score)):
        #     if last_score[j] > lower_bound:
        #         new_pair2w.append(last_edge[j])
    return new_pair2w, edge_weight


def count_sim(max_size, data_flag='train', norm=True):
    """
    计算节点间的二阶相似度，本质是共同邻居的数量，统计两节点共现在同一节点的尾实体、头实体位置的次数。
    :param data_flag:
    :return:
    """
    cfg = utils.get_global_config()
    n_rel = utils.DATASET_STATISTICS[cfg.dataset]['n_rel']
    d = read_data(data_flag)
    pos_tails, pos_heads = d['pos_tails'], d['pos_heads']
    pair2w = dict()
    # O(n^2)
    for (h, r), tails in pos_tails.items():
        tails, n = list(tails), len(tails)
        if n > max_size:
            continue
        for i in range(n):
            for j in range(i+1, n):
                t1, t2 = min(tails[i], tails[j]), max(tails[i], tails[j])
                pair2w[(t1, t2)] = pair2w.setdefault((t1, t2), 0) + 1

    for (r, t), heads in pos_heads.items():
        heads, n = list(heads), len(heads)
        if n > max_size:
            continue
        for i in range(n):
            for j in range(i+1, n):
                h1, h2 = min(heads[i], heads[j]), max(heads[i], heads[j])
                pair2w[(h1, h2)] = pair2w.setdefault((h1, h2), 0) + 1

    # 类似Jaccard系数
    if norm:
        neigh_count = d['neigh_count']
        for k, v in pair2w.items():
            total_neigh = neigh_count[k[0]] + neigh_count[k[1]] - v
            assert total_neigh > 0
            pair2w[k] = v / total_neigh

    return pair2w


def count_rel_sim(max_size, data_flag='train', norm=True):
    """
    计算节点间的二阶相似度，本质是共同邻居的数量，统计两节点共现在同一节点的尾实体、头实体位置的次数。
    :param data_flag:
    :return:
    """
    cfg = utils.get_global_config()
    d = read_data(data_flag)
    in_rels, out_rels = d['in_rels'], d['out_rels']

    # 统计集合大小, 用于设置max_size.
    # in_rels_count, out_rels_count = {}, {}  # key: set size, value: count
    # for ent, rel_set in in_rels.items():
    #     in_rels_count[len(rel_set)] = in_rels_count.setdefault(len(rel_set), 0) + 1
    # in_rels_count = dict(sorted(in_rels_count.items(), key=lambda x: x[0]))
    # print('in_rels count: {}'.format(in_rels_count))
    #
    # for ent, rel_set in out_rels.items():
    #     out_rels_count[len(rel_set)] = out_rels_count.setdefault(len(rel_set), 0) + 1
    # out_rels_count = dict(sorted(out_rels_count.items(), key=lambda x: x[0]))
    # print('out_rels count: {}'.format(out_rels_count))

    pair2w = dict()
    rel_neighs = dict()  # 统计每个relation总的邻居共现次数
    for ent, rel_set in in_rels.items():
        if len(rel_set) > max_size:
            continue
        # 用于之后计算r的所有邻居数
        for r in rel_set:
            rel_neighs[r] = rel_neighs.setdefault(r, 0) + len(rel_set)-1
        # 统计两两rel的共现次数
        rels, n = list(rel_set), len(rel_set)
        for i in range(n):
            for j in range(i+1, n):
                r1, r2 = min(rels[i], rels[j]), max(rels[i], rels[j])
                pair2w[(r1, r2)] = pair2w.setdefault((r1, r2), 0) + 1

    for ent, rel_set in out_rels.items():
        if len(rel_set) > max_size:
            continue
        for r in rel_set:
            rel_neighs[r] = rel_neighs.setdefault(r, 0) + len(rel_set) - 1
        rels, n = list(rel_set), len(rel_set)
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = min(rels[i], rels[j]), max(rels[i], rels[j])
                pair2w[(r1, r2)] = pair2w.setdefault((r1, r2), 0) + 1

    if norm:
        for k, v in pair2w.items():
            total_neigh = rel_neighs[k[0]] + rel_neighs[k[1]] - v
            assert total_neigh > 0, print(k, v, sep=' | ')
            pair2w[k] = v / total_neigh

    return pair2w


def construct_ksg(mode='entity', direct='bi'):
    """
    构造agg_graph, 为根据二阶相似性构造的同质图.
    :return:
    """
    assert mode in ['entity', 'relation']
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

    if mode == 'entity':
        pair2w = count_sim(max_size=cfg.max_sim_set, norm=True)
        pair2w_sorted, edge_weight = filter_pair2w(pair2w, remove_proportion=cfg.filter_rate, flag='node')
    else:
        pair2w = count_rel_sim(max_size=cfg.rel_max_sim_set, norm=True)
        pair2w_sorted, edge_weight = filter_pair2w(pair2w, remove_proportion=cfg.rel_filter_rate)

    # 构造图的边
    src, dst = [], []
    for k in pair2w_sorted:
        # 判断不会有计算两次的pair
        if direct == 'bi':
            assert k[0] <= k[1], k
            src.extend([k[0], k[1]])
            dst.extend([k[1], k[0]])
        elif direct == 'uni':
            src.extend([k[0]])
            dst.extend([k[1]])
    src, dst = torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)

    # 构图
    # g = dgl.graph((src, dst), num_nodes=n_ent if mode == 'entity' else n_rel)
    # 添加自环
    # g = dgl.add_self_loop(g)

    return src, dst, edge_weight


class KGTrainDataset(Dataset):
    """
    按照(h, r, pos_tails) & (pos_heads, r, t)的格式来训练, 一次训练多个entity
    """
    def __init__(self, data_flag, hr2eid, rt2eid):
        assert data_flag in ['train', 'valid', 'test']
        logging.info('---加载Train数据---')
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(data_flag)
        self.query = []
        self.label = []
        self.rm_edges = []
        self.set_scaling_weight = []

        for k, v in self.d['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            self.rm_edges.append(hr2eid[k])

        # count_2_dict = {}
        for k, v in self.d['pos_heads'].items():
            # 添加逆边数据, label即为pos_tails，但要注意边的id
            self.query.append((k[1], k[0] + self.n_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(rt2eid[k])

    def __len__(self):
        # 保证head_batch和tail_batch数据量相等
        # return 2*max(self.n_head_data, self.n_tail_data)
        return len(self.label)

    def __getitem__(self, item):

        h, r, t = self.query[item]
        label = self.get_onehot_label(self.label[item])

        # 随机去掉一定比例的边
        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.cfg.rm_rate)
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num]
        rm_edges = rm_edges[rm_inds]

        return (h, r, t), label, rm_edges

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.n_ent)
        onehot_label[label] = 1
        if self.cfg.label_smooth != 0.0:
            onehot_label = (1.0 - self.cfg.label_smooth) * onehot_label + (1.0 / self.n_ent)

        return onehot_label

    def get_pos_inds(self, label):
        pos_inds = torch.zeros(self.n_ent).to(torch.bool)
        pos_inds[label] = True
        return pos_inds

    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)  # (bs, )
        label = torch.stack(label, dim=0)  # (bs, n_ent)
        rm_edges = torch.cat(rm_edges, dim=0)  # (n_rm_edges, )

        return (src, rel, dst), label, rm_edges


class KSGTrainDataset(Dataset):
    """
    按照(h, r, pos_tails) & (pos_heads, r, t)的格式来训练, 一次训练多个entity
    """
    def __init__(self, data_flag):
        assert data_flag in ['train', 'valid', 'test']
        logging.info('---加载Train数据---')
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(data_flag)
        self.query = []
        self.label = []
        self.set_scaling_weight = []

        for k, v in self.d['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))

        # count_2_dict = {}
        for k, v in self.d['pos_heads'].items():
            # 添加逆边数据, label即为pos_tails，但要注意边的id
            self.query.append((k[1], k[0] + self.n_rel, -1))
            self.label.append(list(v))

    def __len__(self):
        # 保证head_batch和tail_batch数据量相等
        # return 2*max(self.n_head_data, self.n_tail_data)
        return len(self.label)

    def __getitem__(self, item):

        h, r, t = self.query[item]
        label = self.get_onehot_label(self.label[item])

        return (h, r, t), label

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.n_ent)
        onehot_label[label] = 1
        if self.cfg.label_smooth != 0.0:
            onehot_label = (1.0 - self.cfg.label_smooth) * onehot_label + (1.0 / self.n_ent)

        return onehot_label

    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)  # (bs, )
        label = torch.stack(label, dim=0)  # (bs, n_ent)
        rm_edges = None  # (n_rm_edges, )

        return (src, rel, dst), label, rm_edges


class EvalDataset(Dataset):
    """
    eval时按正常情况的单个triple来分析, 和train时按照set来分析不一样
    """
    def __init__(self, data_flag, mode):
        assert data_flag in ['train', 'valid', 'test']
        assert mode in ['head_batch', 'tail_batch']
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.mode = mode
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.d = read_data(data_flag)
        self.trip = [_ for _ in zip(self.d['src_list'], self.d['rel_list'], self.d['dst_list'])]
        self.d_all = read_data(['train', 'valid', 'test'])
        self.pos_t = self.d_all['pos_tails']
        self.pos_h = self.d_all['pos_heads']

    def __len__(self):
        return len(self.trip)

    def __getitem__(self, item):
        h, r, t = self.trip[item]

        if self.mode == 'tail_batch':
            # filter_bias
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_t[(h, r)])] = -float('inf')
            filter_bias[t] = 0.
        elif self.mode == 'head_batch':
            # filter_bias
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_h[(r, t)])] = -float('inf')
            filter_bias[h] = 0.
            h, r, t = t, r+self.n_rel, h  # 逆边整体倒过来
        else:
            raise NotImplementedError

        return (h, r, t), filter_bias.tolist(), self.mode

    @staticmethod
    def collate_fn(data: List[Tuple[tuple, list, str]]):
        h = [d[0][0] for d in data]
        r = [d[0][1] for d in data]
        t = [d[0][2] for d in data]
        filter_bias = [d[1] for d in data]
        mode = data[0][-1]

        h = torch.tensor(h, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.int64)
        t = torch.tensor(t, dtype=torch.int64)
        filter_bias = torch.tensor(filter_bias, dtype=torch.float)

        return (h, r, t), filter_bias, mode


class BiDataloader(object):
    """
    将头, 尾节点的dataloader整合在一起, 轮流返回.
    """
    def __init__(self, h_loader: iter, t_loader: iter):
        self.h_loader_len = len(h_loader)
        self.t_loader_len = len(t_loader)
        self.h_loader_step = 0
        self.t_loader_step = 0
        self.total_len = self.h_loader_len + self.t_loader_len
        self.h_loader = self.inf_loop(h_loader)
        self.t_loader = self.inf_loop(t_loader)
        self._step = 0

    def __next__(self):
        if self._step == self.total_len:
            # 确认两个迭代器中的数据都被访问到
            assert self.h_loader_step == self.h_loader_len
            assert self.t_loader_step == self.t_loader_len
            self._step = 0
            self.h_loader_step = 0
            self.t_loader_step = 0
            raise StopIteration
        if self._step % 2 == 0:
            # 正常情况下访问head_loader; 若迭代完毕, 则访问tail_loader
            if self.h_loader_step < self.h_loader_len:
                data = next(self.h_loader)
                self.h_loader_step += 1
            else:
                data = next(self.t_loader)
                self.t_loader_step += 1
        else:
            if self.t_loader_step < self.t_loader_len:
                data = next(self.t_loader)
                self.t_loader_step += 1
            else:
                data = next(self.h_loader)
                self.h_loader_step += 1
        self._step += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_len

    @staticmethod
    def inf_loop(dataloader):
        '''
        将dataloader转为无限循环
        '''
        while True:
            for data in dataloader:
                yield data
