import hydra
from os.path import join
import utils
from data_helper import read_data, construct_kg, construct_dict
from collections import defaultdict
import json
import logging
from model_helper import evaluate
import pickle
import torch
from model import KGEModel
import dgl
import data_helper
from data_helper import read_data
from run import get_kg
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from omegaconf import DictConfig


def updata_cfg(saved_cfg, cur_cfg):
    assert saved_cfg.dataset == cur_cfg.dataset, 'dataset value of two configs should be the same'
    saved_cfg.device = cur_cfg.device
    saved_cfg.project_dir = cur_cfg.project_dir
    saved_cfg.data_path = cur_cfg.data_path
    saved_cfg.working_dir = cur_cfg.working_dir


class ModelAnalyzer:
    """
    Analyse the model performance
    """
    def __init__(self, path):
        # 读取当前配置,
        cur_cfg = utils.get_global_config()
        self.device = torch.device(cur_cfg.device)
        # saved_cfg = pickle.load(open(join(path, 'config.pickle'), 'rb'))
        # updata_cfg(saved_cfg, cur_cfg)
        # utils.set_global_config(saved_cfg)
        # self.cfg = saved_cfg
        self.cfg = cur_cfg

        self.n_ent = utils.DATASET_STATISTICS[self.cfg.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.cfg.dataset]['n_rel']
        src, dst, rel, _, _ = data_helper.construct_kg('train')
        self.kg = get_kg(src, dst, rel, self.device)

        # 加载模型
        self.model = KGEModel(self.cfg.ent_dim).to(self.device)
        self.model.eval()
        checkpoint = torch.load(join(path, 'checkpoint.torch'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def compute_model_rank(self, save_path, data_flag='test'):
        # 计算SE-GNN模型结果值
        data2rank = dict()
        metrics = evaluate(self.model, data_flag, self.kg, record=True, do_round=True)
        for (h, r, t, rank) in metrics['ranking']:
            h, r, t = int(h), int(r), int(t)
            if r < self.n_rel:
                # tb
                data2rank[(h, r, t, 'tail-batch')] = rank
            else:
                # hb, 这里的h是反着的, 代表正常三元组的t
                data2rank[(t, r-self.n_rel, h, 'head-batch')] = rank
        logging.info('rank computed')
        pickle.dump(data2rank, open(save_path, 'wb'))


class Metrics:
    def __init__(self):
        self.cfg = utils.get_global_config()
        device = self.cfg.device
        src, dst, rel = data_helper.construct_kg_directed('train')
        self.kg = get_kg(src, dst, rel, device)

    def compute_semantic_metrics(self, save_path, max_prox_size=25):
        """
        计算数据 (M_rt, M_hr, M_hrt) 的三类语义指标值
        :param max_prox_size: 计算proximity时query的最大size
        :return:
        """

        # 计算三类指标值
        # entity pair to similarity
        epair2simlar = Metrics.count_simlar_between_ents(max_prox_size)
        train_d = read_data('train')
        test_d = read_data('test')
        # data to semantic evidence dict, key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        data2se = defaultdict(dict)
        for (h, r), t_set in test_d['pos_tails'].items():
            for t in t_set:
                # M_rt, 统计r和尾节点t在train中共现次数
                mrt = len(train_d['pos_heads'].get((r, t), []))
                # M_ht, 统计h和t在train中的路径数
                # 无向图, 所以h, t顺序并不重要, 但这和论文中的结论不一样
                # 分析并非严格顺序的连边是否有影响?
                npath = Metrics.count_path_between_ent(self.kg, h, t)
                mht = npath
                # M_hrt, 统计hr的train中的其他answer, 与t的相似程度
                mhrt = 0
                answer_set = train_d['pos_tails'].get((h, r), set())
                for e in answer_set:
                    e1, e2 = min(t, e), max(t, e)
                    mhrt += epair2simlar.get((e1, e2), 0)
                data2se[(h, r, t, 'tail-batch')]['M_rt'] = mrt
                data2se[(h, r, t, 'tail-batch')]['M_ht'] = mht
                data2se[(h, r, t, 'tail-batch')]['M_hrt'] = mhrt
        for (r, t), h_set in test_d['pos_heads'].items():
            for h in h_set:
                # M_rt
                mrt = len(train_d['pos_tails'].get((h, r), []))
                # M_ht
                npath = Metrics.count_path_between_ent(self.kg, t, h)
                mht = npath
                # M_hrt
                mhrt = 0
                answer_set = train_d['pos_heads'].get((r, t), set())
                for e in answer_set:
                    e1, e2 = min(h, e), max(h, e)
                    mhrt += epair2simlar.get((e1, e2), 0)
                data2se[(h, r, t, 'head-batch')]['M_rt'] = mrt
                data2se[(h, r, t, 'head-batch')]['M_ht'] = mht
                data2se[(h, r, t, 'head-batch')]['M_hrt'] = mhrt
        pickle.dump(data2se, open(save_path, 'wb'))
        logging.info('SE metrics computing end...')

    def plot_se2rank(self, model_name: list, rank_load_paths: list, fig_save_path, se_metric_path):
        """
        计算三个se分别与mean rank的关系
        :param model_name: 模型名字
        :param rank_load_paths: 所有模型的结果
        :param fig_save_path: figure的保存位置
        :param se_metric_path: semantic metrics保存位置
        :return:
        """
        # 加载数据
        data2ranks = []
        for p in rank_load_paths:
            # key: (h, r, t, mode), value: rank
            data2ranks.append(pickle.load(open(p, 'rb')))

        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        data2se = pickle.load(open(se_metric_path, 'rb'))
        for q2r in data2ranks:
            assert len(q2r) == len(data2se)
        total_ndata = len(data2se)

        # 把四元组取出来, rank_list: 每个模型声明一个list, 记录每个se metric对应的rank
        mrt_list, mht_list, mhrt_list, rank_list = [], [], [], [[] for _ in data2ranks]
        for q, metrics in data2se.items():
            mrt_list.append(metrics['M_rt'])
            mht_list.append(metrics['M_ht'])
            mhrt_list.append(metrics['M_hrt'])
            for i, q2r in enumerate(data2ranks):
                # 根据query取对应rank
                assert q in q2r
                rank_list[i].append(q2r[q])
        for i, rank in enumerate(rank_list):
            rank_list[i] = np.array(rank)
            # print('{} rank shape: {}'.format(i, rank_list[i].shape))

        # 指标分段
        if self.cfg.dataset == 'FB15k_237':
            mrt_bounds = [0, 3, 40]
            mht_bounds = [0, 1, 3]
            mhrt_bounds = [0, 4, 50]
        elif self.cfg.dataset == 'WN18RR':
            mrt_bounds = [0, 1]
            mht_bounds = [0, 1]
            mhrt_bounds = [0, 1]
        else:
            raise NotImplementedError
        mrt_range2ndata, mrt_range2id = Metrics.ndata_of_range(mrt_list, mrt_bounds)
        mht_range2ndata, mht_range2id = Metrics.ndata_of_range(mht_list, mht_bounds)
        mhrt_range2ndata, mhrt_range2id = Metrics.ndata_of_range(mhrt_list, mhrt_bounds)
        print('M_rt range stat: {}'.format(mrt_range2ndata))
        print('M_ht range stat: {}'.format(mht_range2ndata))
        print('M_hrt range stat: {}'.format(mhrt_range2ndata))

        # 计算每个指标的区间名和数据比例 (底部和顶部的横轴)
        mrt_ranges, mrt_ndata = [], []
        for r_str, r_ndata in mrt_range2ndata.items():
            mrt_ranges.append(r_str)
            mrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mht_ranges, mht_ndata = [], []
        for r_str, r_ndata in mht_range2ndata.items():
            mht_ranges.append(r_str)
            mht_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mhrt_ranges, mhrt_ndata = [], []
        for r_str, r_ndata in mhrt_range2ndata.items():
            mhrt_ranges.append(r_str)
            mhrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        # 计算每个模型在指标区间内的Mean Rank, n_model*3 list
        mrt_ranks, mht_ranks, mhrt_ranks = [], [], []
        for rank in rank_list:
            mrt_r, mht_r, mhrt_r = [], [], []
            for _, r_ids in mrt_range2id.items():
                mrt_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mht_range2id.items():
                mht_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mhrt_range2id.items():
                mhrt_r.append(np.mean(rank[r_ids]))
            mrt_ranks.append(mrt_r)
            mht_ranks.append(mht_r)
            mhrt_ranks.append(mhrt_r)
        # print('mrt ranks: {}'.format(mrt_ranks))
        # print('mht ranks: {}'.format(mht_ranks))
        # print('mhrt ranks: {}'.format(mhrt_ranks))

        # 画图

        if len(model_name) == 6:
            # 6柱
            plt.figure(figsize=(15, 5))
            n_model = len(model_name)
            n_round = None
            # 因为数据集的分段不一样, 所以横坐标数量也不一样
            if self.cfg.dataset == 'FB15k_237':
                b_width = 0.6
                inds = np.array([4, 8, 12])
                zeros = np.array([0, 0, 0])
                # b_width = 0.6
                # inds = np.array([5, 10, 15])
                # zeros = np.array([0, 0, 0])
            elif self.cfg.dataset == 'WN18RR':
                b_width = 0.5
                inds = np.array([4, 8])
                zeros = np.array([0, 0])
            else:
                raise NotImplementedError
            offset_inds = inds + (n_model - 1) / 2 * b_width
        elif len(model_name) == 2:
            # 2柱
            plt.figure(figsize=(10, 5))
            n_model = len(model_name)
            n_round = 1
            # 因为数据集的分段不一样, 所以横坐标数量也不一样
            if self.cfg.dataset == 'FB15k_237':
                b_width = 0.4
                inds = np.array([1,2,3])
                zeros = np.array([0, 0, 0])
            elif self.cfg.dataset == 'WN18RR':
                b_width = 0.4
                inds = np.array([1, 2])
                zeros = np.array([0, 0])
            else:
                raise NotImplementedError
            offset_inds = inds + (n_model - 1) / 2 * b_width
        else:
            raise NotImplementedError

        ax1 = plt.subplot(131)
        plt.xlabel('Relation level SE Ranges', size='large')
        plt.ylabel('Mean Rank', size='large')
        for i, (rank, name) in enumerate(zip(mrt_ranks, model_name)):
            plt.bar(inds+i*b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mrt_ranges, rotation=0)
        plt.legend(fontsize=10)
        ax1.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mrt_ndata, rotation=0)

        ax2 = plt.subplot(132, sharey=ax1)
        plt.xlabel('Entity level SE Ranges', size='large')
        for i, (rank, name) in enumerate(zip(mht_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mht_ranges, rotation=0)
        plt.legend(fontsize=10)
        ax2.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mht_ndata, rotation=0)

        ax3 = plt.subplot(133, sharey=ax1)
        plt.xlabel('Triple level SE Ranges', size='large')
        for i, (rank, name) in enumerate(zip(mhrt_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mhrt_ranges, rotation=0)
        plt.legend(fontsize=10)
        ax3.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mhrt_ndata, rotation=0)

        plt.subplots_adjust(wspace=0.2)  # 调整子图间距
        plt.savefig(fig_save_path, format='svg')
        logging.info('Fig saved')

    @staticmethod
    def count_simlar_between_ents(max_size, data_flag='train', norm=False):
        """
        计算节点间的二阶相似度，本质是共同邻居的数量，统计两节点共现在同一节点的尾实体、头实体位置的次数。
        :param data_flag:
        :return:
        """
        d = read_data(data_flag)
        pos_tails, pos_heads = d['pos_tails'], d['pos_heads']
        pair2w = dict()
        # O(n^2)
        for (h, r), tails in pos_tails.items():
            tails, n = list(tails), len(tails)
            if n > max_size:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    t1, t2 = min(tails[i], tails[j]), max(tails[i], tails[j])
                    pair2w[(t1, t2)] = pair2w.setdefault((t1, t2), 0) + 1

        for (r, t), heads in pos_heads.items():
            heads, n = list(heads), len(heads)
            if n > max_size:
                continue
            for i in range(n):
                for j in range(i + 1, n):
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

    @staticmethod
    def count_path_between_ent(kg, e1, e2):
        """
        分析实体e1 -> e2的路径数 (长度<=2)
        :param kg:
        :param e1:
        :param e2:
        :return:
        """

        _, hop1_neighbors = kg.out_edges(e1, form='uv')
        _, hop2_neighbors = kg.out_edges(hop1_neighbors, form='uv')

        hop1_npath, hop2_npath = 0, 0
        for n in hop1_neighbors.tolist():
            if n == e2:
                hop1_npath += 1
        for n in hop2_neighbors.tolist():
            if n == e2:
                hop2_npath += 1

        npath = hop1_npath + hop2_npath

        return npath

    @staticmethod
    def ndata_of_range(data: list, bounds: list):
        """
        按给定分界符统计处于某个区间的数据个数
        :param data:
        :param bounds:
        :return:
        """
        cfg = utils.get_global_config()
        # eg. bounds: [0, 10, 20, 30], n个界标, 分为n个range, 左闭右开
        i2str = dict()
        for i in range(len(bounds)):
            if i < len(bounds)-1:
                i2str[i] = '[{}, {})'.format(bounds[i], bounds[i+1])
            else:
                if cfg.dataset == 'FB15k_237':
                    i2str[i] = '[{}, +∞)'.format(bounds[i])
                elif cfg.dataset == 'WN18RR':
                    i2str[i] = '[{}, Max]'.format(bounds[i])
                else:
                    raise NotImplementedError
        range2ndata = dict([(i2str[i], 0) for i in range(len(bounds))])
        # 记录的是数据的id, 而不是数据的值, 后边要根据id去寻找对应的rank
        range2id = dict([(i2str[i], []) for i in range(len(bounds))])
        for i, d in enumerate(data):
            assert d >= bounds[0]
            # 从后向前遍历数据属于哪个数据区间
            for j in range(len(bounds) - 1, -1, -1):
                if d >= bounds[j]:
                    range2ndata[i2str[j]] += 1
                    range2id[i2str[j]].append(i)
                    break
        return range2ndata, range2id


def modify_data2rank_key(path, output_path):
    data2rank = pickle.load(open(path, 'rb'))
    new_data2rank = dict()
    for d, rank in data2rank.items():
        if d[3] == 'hb':
            new_data2rank[(d[0], d[1], d[2], 'head-batch')] = rank
        elif d[3] == 'tb':
            new_data2rank[(d[0], d[1], d[2], 'tail-batch')] = rank
        else:
            raise NotImplementedError
    pickle.dump(new_data2rank, open(output_path, 'wb'))


def FB15k_237():
    cfg = utils.get_global_config()
    met = Metrics()
    se_metric_path = join(cfg.data_path, cfg.dataset, 'test_se_metrics_2')

    # met.compute_semantic_metrics(save_path=se_metric_path)

    # 整体趋势图
    # dir_path = join(cfg.project_dir, 'fb_result')
    # rank_paths = [join(dir_path, '2_fb_rank'), join(dir_path, '7_fb_rank'),
    #               join(dir_path, '4_fb_rank'), join(dir_path, '3_fb_rank'),
    #               join(dir_path, '6_fb_rank'), join(dir_path, '5_fb_rank'),
    #               join(dir_path, 'SE_GNN_3645_FB15k_237_rank')]
    # met.plot_se2rank(
    #     model_name=['ConvE', 'CompGCN', 'RotatE', 'TransE', 'DistMult', 'ComplEx'],
    #     rank_load_paths=rank_paths, fig_save_path=join(dir_path, 'fb_baselines_se2rank.svg'),
    #     se_metric_path=se_metric_path
    # )

    # ConvE / SE-GNN对比图
    dir_path = join(cfg.project_dir, 'fb_result')
    rank_paths = [join(dir_path, '2_fb_rank'), join(dir_path, 'SE_GNN_3645_FB15k_237_rank')]
    met.plot_se2rank(
        model_name=['ConvE', 'SE-GNN'],
        rank_load_paths=rank_paths, fig_save_path=join(dir_path, 'fb_conve_segnn_se2rank.svg'),
        se_metric_path=se_metric_path
    )


def WN18RR():
    cfg = utils.get_global_config()
    met = Metrics()
    se_metric_path = join(cfg.data_path, cfg.dataset, 'test_se_metrics_2')

    # met.compute_semantic_metrics(save_path=se_metric_path)
    # exit(1)

    # 整体趋势图
    dir_path = join(cfg.project_dir, 'wn_result')
    rank_paths = [join(dir_path, '5_wn_rank'), join(dir_path, '6_wn_rank'),
                  join(dir_path, '2_wn_rank'), join(dir_path, '7_wn_rank'),
                  join(dir_path, '4_wn_rank'), join(dir_path, '3_wn_rank')]
    met.plot_multi_fig(model_name=['ComplEx', 'DistMult', 'ConvE', 'CompGCN', 'RotatE', 'TransE'],
                       rank_load_paths=rank_paths, fig_save_path=join(dir_path, '243657_wn_fig.svg'),
                       se_metric_path=se_metric_path)


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(cfg: DictConfig):
    utils.set_global_config(cfg)

    # compute SE-GNN rank
    # fb_3645_path = '/home/liren/project/semantic_evidence/data/outputs/FB15k_237/2021-12-05/22-05-58'
    # ma = ModelAnalyzer(fb_3645_path)
    # rank_save_path = '/home/liren/project/semantic_evidence/fb_result/SE_GNN_3645_FB15k_237_rank'
    # ma.compute_model_rank(rank_save_path)
    # exit(1)

    FB15k_237()



if __name__ == '__main__':
    main()
    pass
