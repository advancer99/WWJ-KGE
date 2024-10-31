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
from model import LinkPredModel
import dgl
import data_helper
from data_helper import read_data
from run import get_kg
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join


@hydra.main(config_path=join('..', 'config'), config_name="config")
def set_current_config(cfg):
    utils.set_global_config(cfg)


class CaseStudy:
    """
    case study相关代码
    """
    def __init__(self):
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

    def select_relation_se_cases(self, dataset='FB15k_237', data_flag='test'):
        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        dir_path = join(self.cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid)+1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        metrics_save_path = join(self.cfg.raw_dataset_dir, self.cfg.dataset, '{}_semantic_metrics'.format(data_flag))
        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        query2metrics = pickle.load(open(metrics_save_path, 'rb'))
        m_rt_threshold, m_ht_threshold, m_hrt_threshold = 40, 7, 50
        i = 0
        for q, metric in query2metrics.items():
            if metric['M_rt'] > m_rt_threshold:
                h, r, t, mode = q[0], q[1], q[2], q[3]
                h_mid, t_mid = id2mid[h], id2mid[t]
                h_name, t_name = mid2name.get(h_mid, 'None'), mid2name.get(t_mid, 'None')
                r_name = id2rel[r]
                if (i > 2000) and (i <= 3000):
                    print('mode: {}, M_rt: {}'.format(mode, metric['M_rt']))
                    print('{}\t{}\t{}'.format(h, r, t))
                    print('{}\t{}\t{}\n'.format(h_name, r_name, t_name))
                if i > 3000:
                    break
                i += 1

    def select_entity_se_cases(self, dataset='FB15k_237', data_flag='test'):
        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        dir_path = join(self.cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid)+1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        metrics_save_path = join(self.cfg.raw_dataset_dir, self.cfg.dataset, '{}_semantic_metrics'.format(data_flag))
        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        query2metrics = pickle.load(open(metrics_save_path, 'rb'))
        m_rt_threshold, m_ht_threshold, m_hrt_threshold = 40, 7, 50
        i = 0
        for q, metric in query2metrics.items():
            if metric['M_ht'] > m_ht_threshold:
                h, r, t, mode = q[0], q[1], q[2], q[3]
                h_mid, t_mid = id2mid[h], id2mid[t]
                h_name, t_name = mid2name.get(h_mid, 'None'), mid2name.get(t_mid, 'None')
                r_name = id2rel[r]
                print('mode: {}'.format(mode))
                print('{}\t{}\t{}'.format(h, r, t))
                print('{}\t{}\t{}\n'.format(h_name, r_name, t_name))
                i += 1
                if i >= 1000:
                    break

    def select_triple_se_cases(self, dataset='FB15k_237', data_flag='test'):
        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        dir_path = join(self.cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid)+1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        metrics_save_path = join(self.cfg.raw_dataset_dir, self.cfg.dataset, '{}_semantic_metrics'.format(data_flag))
        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        query2metrics = pickle.load(open(metrics_save_path, 'rb'))
        m_rt_threshold, m_ht_threshold, m_hrt_threshold = 40, 7, 50
        i = 0
        for q, metric in query2metrics.items():
            if metric['M_hrt'] > m_hrt_threshold:
                h, r, t, mode = q[0], q[1], q[2], q[3]
                h_mid, t_mid = id2mid[h], id2mid[t]
                h_name, t_name = mid2name.get(h_mid, 'None'), mid2name.get(t_mid, 'None')
                r_name = id2rel[r]
                print('mode: {}'.format(mode))
                print('{}\t{}\t{}'.format(h, r, t))
                print('{}\t{}\t{}\n'.format(h_name, r_name, t_name))
                i += 1
                if i >= 1000:
                    break

    @staticmethod
    def relation_se_case_study(h, r, t, mode, dataset='FB15k_237'):
        """
        分析实体e1和e2间的entity level SE, 即训练集中的路径数 (长度<=2), 这里只输出有向图的结果 (否则逆关系不好表达)
        :param kg:
        :param e1:
        :param e2:
        :return:
        """

        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        cfg = utils.get_global_config()
        cfg.dataset = dataset
        n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        dir_path = join(cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid) + 1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        h_mid, t_mid = id2mid[h], id2mid[t]
        h_name, t_name = mid2name[h_mid], mid2name[t_mid]
        r_name = id2rel[r]
        print('mode: {}'.format(mode))
        print('{}\t{}\t{}'.format(h, r, t))
        print('{}\t{}\t{}'.format(h_name, r_name, t_name))
        print('-------------------------')
        train_d = read_data('train')
        if mode == 'tb':
            e_set = train_d['pos_heads'][(r, t)]  # 要看训练集中有多少满足 (xx, r, t) 的数据
            for e in e_set:
                e_mid = id2mid[e]
                e_name = mid2name.get(e_mid, 'None')
                print('id: {}, name: {}'.format(e, e_name))
        elif mode == 'hb':
            pass
        else:
            raise NotImplementedError

    @staticmethod
    def entity_se_case_study(kg, e1, e2, dataset='FB15k_237'):
        """
        分析实体e1和e2间的entity level SE, 即训练集中的路径数 (长度<=2), 这里只输出有向图的结果 (否则逆关系不好表达)
        :param kg:
        :param e1:
        :param e2:
        :return:
        """

        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        cfg = utils.get_global_config()
        cfg.dataset = dataset
        n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        dir_path = join(cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid) + 1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        u1, v1, eid_1 = kg.out_edges(e1, form='all')
        u2, v2, eid_2 = kg.in_edges(e2, form='all')

        v1, u2 = v1.tolist(), u2.tolist()
        n = 0
        for i, e in enumerate(v1):
            if e == e2:
                n += 1
                r = kg.edata['rel_id'][eid_1[i]].item()
                if r >= n_rel:
                    continue
                e1_mid, e2_mid = id2mid[e1], id2mid[e2]
                e1_name, e2_name = mid2name.get(e1_mid, 'None'), mid2name.get(e2_mid, 'None')
                r_name = id2rel[r]
                print('{}\t{}\t{}'.format(e1, r, e2))
                print('{}\t{}\t{}'.format(e1_name, r_name, e2_name))

        for i, e in enumerate(v1):
            for j, temp_e in enumerate(u2):
                if e == temp_e:
                    n += 1
                    r1, r2 = kg.edata['rel_id'][eid_1[i]].item(), kg.edata['rel_id'][eid_2[j]].item()
                    if r1 >= n_rel or r2 >= n_rel:
                        continue
                    e_mid, e1_mid, e2_mid = id2mid[e], id2mid[e1], id2mid[e2]
                    e_name, e1_name, e2_name = mid2name.get(e_mid, 'None'), mid2name.get(e1_mid, 'None'), mid2name.get(e2_mid, 'None')
                    r1_name, r2_name = id2rel[r1], id2rel[r2]
                    print('{} \t {} \t {} \t {} \t {}'.format(e1, r1, e, r2, e2))
                    print('{} \t {} \t {} \t {} \t {}'.format(e1_name, r1_name, e_name, r2_name, e2_name))

        print(n)

    @staticmethod
    def triple_se_case_study(h, r, t, mode, dataset='FB15k_237', max_prox_size=25):
        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        cfg = utils.get_global_config()
        cfg.dataset = dataset
        n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        dir_path = join(cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid) + 1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        h_mid, t_mid = id2mid[h], id2mid[t]
        h_name, t_name = mid2name[h_mid], mid2name[t_mid]
        r_name = id2rel[r]
        train_d = read_data('train')
        np2prox = data_helper.count_prox(max_prox_size)
        print('mode: {}'.format(mode))
        print('{}\t{}\t{}'.format(h, r, t))
        print('{}\t{}\t{}'.format(h_name, r_name, t_name))
        print('-------------------------')
        e2sim = dict()  # key: e, value: similarity
        if mode == 'tb':
            assert (h, r) in train_d['pos_tails']
            answer_set = train_d['pos_tails'][(h, r)]
            for e in answer_set:
                sim = np2prox.get((t, e), 0) + np2prox.get((e, t), 0)
                e2sim[e] = sim
        elif mode == 'hb':
            assert (r, t) in train_d['pos_heads']
            answer_set = train_d['pos_heads'][(r, t)]
            for e in answer_set:
                sim = np2prox.get((h, e), 0) + np2prox.get((e, h), 0)
                e2sim[e] = sim
        else:
            raise NotImplementedError

        e2sim = dict(sorted(e2sim.items(), key=lambda x: x[1], reverse=True))

        for e, sim in e2sim.items():
            e_mid = id2mid[e]
            e_name = mid2name.get(e_mid, 'None')
            print('{} | sim: {} | {}'.format(e_name, sim, e))

    @staticmethod
    def compute_case_rank_using_multi_models(query: tuple, model_name: list, rank_load_paths: list, dataset='FB15k_237', data_flag='test'):
        """
        计算query (h, r, t, mode) 在各个模型中的rank值
        :param query:
        :param model_name:
        :param rank_load_paths:
        :param data_flag:
        :return:
        """
        query2ranks, ranks = [], []
        for p in rank_load_paths:
            query2ranks.append(pickle.load(open(p, 'rb')))
        ndata = len(query2ranks[0])
        for q2r in query2ranks:
            assert len(q2r) == ndata, 'ndata: {}, len_q2r: {}'.foramt(ndata, len(q2r))
            assert query in q2r, query
            ranks.append(q2r[query])

        logging.info('Start')
        # id2mid, mid2ent_name, id2rel_name
        cfg = utils.get_global_config()
        cfg.dataset = dataset
        n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        dir_path = join(cfg.raw_dataset_dir, dataset)
        ent2id, rel2id = construct_dict(dir_path)
        id2mid, id2rel = dict(), dict()
        for ent, i in ent2id.items():
            id2mid[i] = ent
        for rel, i in rel2id.items():
            id2rel[i] = rel
        mid2name = dict()
        mid2name_file = open(join(dir_path, 'fb15k-mid2name.txt'), 'r')
        for line in mid2name_file:
            line = line.strip()
            if '\t' in line:
                mid, name = line.split('\t')[0], line.split('\t')[1]
            else:
                mid = line.split(' ')[0]
                name = line[len(mid) + 1:]
            mid2name[mid] = name
        logging.info('dict reading complete')

        h, r, t, mode = query[0], query[1], query[2], query[3]
        h_mid, t_mid = id2mid[h], id2mid[t]
        h_name, t_name = mid2name[h_mid], mid2name[t_mid]
        r_name = id2rel[r]
        train_d = read_data('train')
        print('mode: {}'.format(mode))
        print('{}\t{}\t{}'.format(h, r, t))
        print('{}\t{}\t{}'.format(h_name, r_name, t_name))
        print('-------------------------')
        for model, rank in zip(model_name, ranks):
            print('{}\trank: {}'.format(model, rank))


def case_study():
    # relation: (6651,13,816,'tb'), (1239,43,141,'tb'), (7515,43,384,'tb'),
    # (3493,14,2272,'tb'), (6680, 148, 9623, 'tb'), (3600, 19, 457, 'tb')

    # entity: (2134,17,862,'tb'), (1124,15,12314,'tb'), ×(1031,42,1359,'tb'),
    # ×(1546,42,6958,'tb'), (2147,66,592,'tb')

    # triple: (4143,101,1467,'tb'), (5376,17,6198,'tb'), (4705,17,6287,'tb'),
    # (2255,17,2006,'tb'), (7370,17,1730,'tb'), (2719,17,2123,'tb'),
    # (9143,31,1211,'tb'), (3177,31,658,'tb'), (3044,31,136,'tb')
    # set_current_config()
    # dir_path = '/home2/liren/project/semantic_variety/model_result'
    # rank_paths = [join(dir_path, '1_fb_rank'), join(dir_path, '3_fb_rank'), join(dir_path, '4_fb_rank'),
    #               join(dir_path, '6_fb_rank'), join(dir_path, '5_fb_rank'), join(dir_path, '2_fb_rank')]
    # model_name = ['SE-GNN', 'TransE', 'RotatE', 'DistMult', 'ComplEx', 'ConvE']
    # compute_case_rank_using_multi_models((7515, 43, 384,'tb'), model_name, rank_paths)
    # print('')
    # compute_case_rank_using_multi_models((3493, 14, 2272,'tb'), model_name, rank_paths)
    # print('')
    # compute_case_rank_using_multi_models((1239, 43, 141,'tb'), model_name, rank_paths)
    # print('')
    # compute_case_rank_using_multi_models((4143, 101, 1467, 'tb'), model_name, rank_paths)
    # print('')
    # compute_case_rank_using_multi_models((5376, 17, 6198, 'tb'), model_name, rank_paths)
    # print('')
    # compute_case_rank_using_multi_models((3177, 31, 658, 'tb'), model_name, rank_paths)
    # print('')
    # exit(1)

    set_current_config()
    relation_se_case_study(6680, 148, 9623, 'tb')
    # id: 7455, name: Todo Sobre Mi Madre (film)
    # id: 5462, name: Midnight in Barcelona
    relation_se_case_study(3493, 14, 2272, 'tb')
    relation_se_case_study(7515, 43, 384, 'tb')
    exit(1)
    # set_current_config()
    # triple_se_case_study(4143,101,1467,'tb')
    # print()
    # triple_se_case_study(5376,17,6198,'tb')
    # print()
    # triple_se_case_study(4705,17,6287,'tb')
    # print()
    # triple_se_case_study(2255,17,2006,'tb')
    # print()
    # triple_se_case_study(7370,17,1730,'tb')
    # print()
    # triple_se_case_study(2719,17,2123,'tb')
    # print()
    # triple_se_case_study(9143,31,1211,'tb')
    # print()
    # triple_se_case_study(3177,31,658,'tb')
    # print()
    # triple_se_case_study(3044,31,136,'tb')
    # print()

    # path = '/home2/liren/project/semantic_variety/data/outputs/GNN/FB15k_237/2021-08-23/12-29-38'
    # ma = ModelAnalyzer(path=path)
    # # ma.select_relation_se_cases()
    # entity_se_case_study(ma.kg, 2134, 862)
    # print()
    # entity_se_case_study(ma.kg, 1124, 12314)
    # print()
    # entity_se_case_study(ma.kg, 2147, 592)
    # exit(1)


if __name__ == '__main__':

    met = Metrics('WN18RR')
    # met.compute_semantic_metrics()
    dir_path = '/home2/liren/project/semantic_variety/result_wn'
    # rank_path = join(dir_path, '4_wn_rank')
    # fig_path = join(dir_path, 'test')
    # met.plot_rank_vs_metrics(rank_path, fig_path, 'RotatE')
    rank_paths = [join(dir_path, '5_wn_rank'), join(dir_path, '6_wn_rank'), join(dir_path, '2_wn_rank'), join(dir_path, '7_wn_rank'), join(dir_path, '4_wn_rank'), join(dir_path, '3_wn_rank')]
    met.plot_se2rank(model_name=['ComplEx', 'DistMult', 'ConvE', 'CompGCN', 'RotatE', 'TransE'], rank_load_paths=rank_paths, fig_save_path=join(dir_path, '243657_wn_fig.svg'))
    exit(1)
    # print('model: {}\n'.format(ma.cfg.model.name))
    # ma.compute_semantic_metrics()
    # ma.compute_model_performance(save_path='/home2/liren/project/semantic_variety/model_result/1_FB237_Rank')
    path = '/home2/liren/project/semantic_variety/data/outputs/GNN/FB15k_237/2021-08-23/12-29-38'
    ma = ModelAnalyzer(path=path)
    ma.compute_semantic_metrics()
    exit(1)
    dir_path = '/home2/liren/project/semantic_variety/model_result'
    # 比较SE-GNN和ConvE
    # rank_paths = [join(dir_path, '2_fb_rank'), join(dir_path, '1_fb_rank')]
    # ma.plot_multi_fig(model_name=['ConvE', 'SE-GNN'], rank_load_paths=rank_paths, fig_save_path=join(dir_path, '21_fb_fig.svg'))
    # 整张baseline的大图
    rank_paths = [join(dir_path, '2_fb_rank'), join(dir_path, '7_fb_rank'), join(dir_path, '4_fb_rank'), join(dir_path, '3_fb_rank'), join(dir_path, '6_fb_rank'), join(dir_path, '5_fb_rank')]
    ma.plot_se2rank(model_name=['ConvE', 'CompGCN', 'RotatE', 'TransE', 'DistMult', 'ComplEx'], rank_load_paths=rank_paths, fig_save_path=join(dir_path, '243657_fb_fig.svg'))


