#!/usr/bin/python3

import argparse
import json
import logging
import os
import numpy as np
import math
import random
import torch
from torch.utils.data import DataLoader
import utils
from data_helper import KGTrainDataset, KSGTrainDataset, construct_ksg, construct_kg, construct_kg_directed
import hydra
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
import pickle
from os.path import join
from model_helper import train_step, evaluate
import dgl
from model import KGEModel
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 指定第3块gpu

def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print('save done!')


def save_model(model, save_variables):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    cfg = utils.get_global_config()
    pickle.dump(cfg, open('config.pickle', 'wb'))

    state_dict = {
        'model_state_dict': model.state_dict(),  # 模型的所有parameter
        **save_variables
    }

    torch.save(state_dict, 'checkpoint.torch')


def get_linear_scheduler_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """
    先线性warm up再线性decay的learning rate scheduler.
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        """
        根据当前step返回一个比率, 最终这个比率会乘到optimizer的学习率上, 作为最终学习率
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return (max_steps - current_step) / (max_steps - warmup_steps)

    assert max_steps >= warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_linear_scheduler(optimizer, max_steps: int, final_rate):
    """
    先线性warm up再线性decay的learning rate scheduler.
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        """
        根据当前step返回一个比率, 最终这个比率会乘到optimizer的学习率上, 作为最终学习率
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        # y=ax+b, 过点 (0, 1), (max_steps, final_lr)
        # final_lr是个比例, 不是真的lr
        return ((final_rate - 1) / max_steps) * current_step + 1

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_two_step_scheduler_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """
    先线性warm up再线性decay的learning rate scheduler.
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        """
        根据当前step返回一个比率, 最终这个比率会乘到optimizer的学习率上, 作为最终学习率
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        if current_step < warmup_steps:
            return current_step / warmup_steps
        elif current_step < (max_steps * 0.8):
            return 1.
        else:
            return 0.2

    assert max_steps >= warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_two_step_scheduler(optimizer, max_steps: int):
    """
    先线性warm up再线性decay的learning rate scheduler.
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        """
        根据当前step返回一个比率, 最终这个比率会乘到optimizer的学习率上, 作为最终学习率
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        if current_step < (max_steps * 0.8):
            return 1.
        else:
            return 0.1

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def format_metrics(name, h_metric, t_metric):
    msg_h = name + ' (head) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_t = name + ' (tail) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_avg = name + ' (avg) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_h = msg_h.format(h_metric['MRR'], h_metric['MR'],
                         h_metric['HITS@1'], h_metric['HITS@3'], h_metric['HITS@10'])
    msg_t = msg_t.format(t_metric['MRR'], t_metric['MR'],
                         t_metric['HITS@1'], t_metric['HITS@3'], t_metric['HITS@10'])
    msg_avg = msg_avg.format(
        (h_metric['MRR'] + t_metric['MRR']) / 2,
        (h_metric['MR'] + t_metric['MR']) / 2,
        (h_metric['HITS@1'] + t_metric['HITS@1']) / 2,
        (h_metric['HITS@3'] + t_metric['HITS@3']) / 2,
        (h_metric['HITS@10'] + t_metric['HITS@10']) / 2
    )
    return msg_h, msg_t, msg_avg


def get_kg(src, dst, rel, device):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    kg.edata['rel_id'] = rel
    kg = kg.to(device)
    return kg


def get_random_ksg(ksg_src, ksg_dst, mask_ratio, device):
    cfg = utils.get_global_config()
    n_ent = utils.DATASET_STATISTICS[cfg.dataset]['n_ent']
    n_rel = utils.DATASET_STATISTICS[cfg.dataset]['n_rel']

    ksg_size = ksg_src.shape[0]
    rand_inds = torch.randperm(ksg_size)
    select_num = math.floor(ksg_size*mask_ratio)
    select_ksg_src, select_ksg_dst = ksg_src[rand_inds[select_num:]], ksg_dst[rand_inds[select_num:]]

    g = dgl.graph((select_ksg_src, select_ksg_dst), num_nodes=n_ent).to(device)

    return g

def mask_ksg(src, dst, n_node, mask_ratio):
    # src = []
    # dst = []
    # n_node = ksg.nodes().shape[0]
    new_g = dgl.graph((torch.tensor([0]),torch.tensor([1])), num_nodes=n_node)
    new_g = dgl.remove_edges(new_g, torch.tensor(0))
    edge_pair = list(zip(src, dst))
    # edge_pair = sorted(edge_pair, key=lambda x: x[1])
    new_src = []
    new_dst = []
    last_src = []
    last_dst = []
    last_node = -1
    for k, v in edge_pair:
        if k != last_node:
            last_node = v
            remove_num = int(len(last_src) * mask_ratio)
            random.shuffle(last_src)
            new_src.extend(last_src[remove_num:])
            new_dst.extend(last_dst[remove_num:])
            last_src = [k]
            last_dst = [v]
        else:
            last_src.append(k)
            last_dst.append(v)
    remove_num = int(len(last_src) * mask_ratio)
    new_src.extend(last_src[remove_num:])
    new_dst.extend(last_dst[remove_num:])
    new_g.add_edges(new_src, new_dst)

    return new_g.to('cuda')


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(config: DictConfig):
    utils.set_global_config(config)
    cfg = utils.get_global_config()
    assert cfg.dataset in cfg.dataset_list
    # 去除随机性
    utils.remove_randomness()
    # 打印配置信息
    logging.info('\n------Config------\n {}'.format(utils.filter_config(cfg)))

    # 复制代码和配置, 方便复现
    code_dir_path = os.path.dirname(__file__) # /data/liuyu/project/zlx/code
    project_dir_path = os.path.dirname(code_dir_path) # /data/liuyu/project/zlx
    config_dir_path = os.path.join(project_dir_path, 'config')
    hydra_current_dir = os.getcwd() # /data/liuyu/project/zlx/outputs/FB15k_237/2023-09-07/09-49-27
    logging.info('Code dir path: {}'.format(code_dir_path))
    logging.info('Config dir path: {}'.format(config_dir_path))
    logging.info('Model save path: {}'.format(hydra_current_dir))
    # os.system('cp -r {} {}'.format(code_dir_path, hydra_current_dir))
    # os.system('cp -r {} {}'.format(config_dir_path, hydra_current_dir))

    # 加载模型
    device = torch.device(cfg.device)
    n_ent = utils.DATASET_STATISTICS[cfg.dataset]['n_ent']
    n_rel = utils.DATASET_STATISTICS[cfg.dataset]['n_rel']
    model = KGEModel(cfg.h_dim)
    model = model.to(device)

    kg_src, kg_dst, kg_rel, kg_hr2eid, kg_rt2eid = construct_kg('train')
    kg = dgl.graph((kg_src, kg_dst), num_nodes=n_ent)
    kg.edata['rel_id'] = kg_rel

    ksg_src, ksg_dst, ksg_eweight = construct_ksg(mode='entity', direct='uni')
    ksg = dgl.graph((ksg_src, ksg_dst), num_nodes=n_ent)
    #save_pkl(ksg, '/data/liuyu/project/zlx/similar_graph.pkl')

    ksg_out_deg = ksg.out_degrees(torch.arange(ksg.number_of_nodes()))
    ksg_zero_deg_num = torch.sum(ksg_out_deg < 1).to(torch.int).item()
    logging.info('ksg (raw) # node: {}'.format(ksg.number_of_nodes()))
    logging.info('ksg (raw) # edge: {}'.format(ksg.number_of_edges()))
    logging.info('ksg (raw) # zero deg node: {}'.format(ksg_zero_deg_num))

    # 为ksg中0度节点加边
    # ksg_zero_deg_nodes = torch.arange(ksg.number_of_nodes())[ksg_out_deg < 1]
    # kg_u, kg_v = kg.out_edges(ksg_zero_deg_nodes)
    # ksg.add_edges(kg_u, kg_v)
    # ksg.add_edges(kg_v, kg_u)
    # 根据节点已有边数量选择被选边数量，方向
    # 图的扰动
    kg.add_edges(kg_dst, kg_src)
    # kg = dgl.graph((kg_src, kg_dst), num_nodes=n_ent)
    for node in kg.nodes():
        node_g = dgl.in_subgraph(kg, [node]) # 以node为中心，抽取其一阶入度节点
        node_ksg = dgl.in_subgraph(ksg, [node])
        if node_ksg.edges()[0].shape[0] != 0:
            sup_node_num = math.ceil(cfg.k_selectfunc / node_ksg.edges()[0].shape[0])
        else:
            sup_node_num = cfg.max_selectkg
        sup_node_num = min(cfg.max_selectkg, sup_node_num)
        head_nodes, tail_nodes = node_g.edges()
        head_nodes = head_nodes[torch.randperm(head_nodes.shape[0])]
        selected = min(head_nodes.shape[0], sup_node_num)
        head_nodes = head_nodes[:selected]
        tail_nodes = tail_nodes[:selected]
        ksg_eweight.extend([1]*selected)
        ksg.add_edges(head_nodes, tail_nodes) # 899
        ksg_eweight.extend([1]*selected)
        ksg.add_edges(tail_nodes, head_nodes) # 36

    # 重新统计ksg信息
    ksg_out_deg = ksg.out_degrees(torch.arange(ksg.number_of_nodes()))
    ksg_zero_deg_num = torch.sum(ksg_out_deg < 1).to(torch.int).item()
    logging.info('ksg (modif) # node: {}'.format(ksg.number_of_nodes()))
    logging.info('ksg (modif) # edge: {}'.format(ksg.number_of_edges()))
    logging.info('ksg (modif) # zero deg node: {}'.format(ksg_zero_deg_num)) # 899

    deg = ksg.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    ksg.ndata['d'] = norm

    g = ksg.to(device)
    
    dataset = KSGTrainDataset('train')

    # if cfg.graph == 'kg':
    #     g = kg.to(device)
    #     dataset = KGTrainDataset('train', kg_hr2eid, kg_rt2eid)
    # elif cfg.graph == 'ksg':
    #     g = ksg.to(device)
    #     dataset = KSGTrainDataset('train')
    # else:
    #     raise NotImplementedError

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.cpu_worker_num,
        collate_fn=dataset.collate_fn
    )

    logging.info('-----Model Parameter Configuration-----')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))

    # set optimizer and scheduler
    n_epoch = cfg.epoch
    single_epoch_step = len(train_loader)
    max_steps = n_epoch * single_epoch_step
    warm_up_steps = int(single_epoch_step * cfg.warmup_epoch)
    init_lr = cfg.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=init_lr
    )
    # scheduler = None
    # scheduler = get_two_step_scheduler_with_warmup(optimizer, warm_up_steps, max_steps)
    # scheduler = get_linear_scheduler(optimizer, max_steps, final_rate=max(1-n_epoch/400, 0))
    scheduler = get_linear_scheduler_with_warmup(optimizer, warm_up_steps, max_steps)

    logging.info('Training... total epoch: {0}, step: {1}'.format(n_epoch, max_steps))
    last_improve_epoch = 0
    step = -1
    best_mrr = 0.
    best_val_metrics = None

    print_flag = False
    for epoch in range(n_epoch):
        loss_list = []
        for batch_data in train_loader:
            # break
            step += 1
            # random_ksg = get_random_ksg(ksg_src, ksg_dst, 0.2, device)
            # random_ksg = mask_ksg(ksg_src, ksg_dst, n_ent, 0.2)
            random_ksg = g
            # if print_flag:
            #     # 统计random ksg信息
            #     random_ksg_out_deg = ksg.out_degrees(torch.arange(random_ksg.number_of_nodes()))
            #     random_ksg_zero_deg_num = torch.sum(random_ksg_out_deg < 1).to(torch.int).item()
            #     logging.info('ksg (random) # node: {}'.format(random_ksg.number_of_nodes()))
            #     logging.info('ksg (random) # edge: {}'.format(random_ksg.number_of_edges()))
            #     logging.info('ksg (random) # zero deg node: {}'.format(random_ksg_zero_deg_num))
            #     print_flag = False

            train_log = train_step(model, batch_data, random_ksg, ksg_eweight, optimizer, scheduler)
            loss_list.append(train_log['loss'])

        val_metr = evaluate(model, d_flag='valid', g=g, edge_weight=ksg_eweight)
        val_h_metr, val_t_metr = val_metr['head_batch'], val_metr['tail_batch']
        if (val_h_metr['MRR'] + val_t_metr['MRR']) / 2 > best_mrr:
            best_mrr = (val_h_metr['MRR'] + val_t_metr['MRR']) / 2
            best_val_metrics = val_metr
            save_variables = {
                'best_mrr': best_mrr,
                'best_metrics': best_val_metrics
            }
            save_model(model, save_variables)
            improvement_flag = '*'
            last_improve_epoch = epoch
        else:
            improvement_flag = ''

        val_msg = 'Val - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f} | '
        val_msg = val_msg.format(
            (val_h_metr['MRR'] + val_t_metr['MRR']) / 2,
            (val_h_metr['MR'] + val_t_metr['MR']) / 2,
            (val_h_metr['HITS@1'] + val_t_metr['HITS@1']) / 2,
            (val_h_metr['HITS@3'] + val_t_metr['HITS@3']) / 2,
            (val_h_metr['HITS@10'] + val_t_metr['HITS@10']) / 2
        )

        if improvement_flag != '':
            test_metr = evaluate(model, d_flag='test', g=g, edge_weight=ksg_eweight)
            test_h_metr, test_t_metr = test_metr['head_batch'], test_metr['tail_batch']
            test_mrr = (test_h_metr['MRR'] + test_t_metr['MRR']) / 2
            val_msg += 'Test - MRR: {:5.4f} | '.format(test_mrr)

        val_msg += improvement_flag

        msg = 'Epoch: {:3d} | Loss: {:5.4f} | '
        msg = msg.format(epoch, np.mean(loss_list))
        msg += val_msg
        logging.info(msg)

        # 判断是否提前结束训练
        if epoch - last_improve_epoch > cfg.max_no_improve:
            logging.info("Long time no improvenment, stop training...")
            break

    logging.info('Training end...')

    # 分析train, test的性能
    # 加载最优模型
    checkpoint = torch.load('checkpoint.torch')
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info('Train metrics ...')
    train_metrics = evaluate(model, 'train', g=g, edge_weight=ksg_eweight)
    train_met = format_metrics('Train', train_metrics['head_batch'], train_metrics['tail_batch'])
    logging.info(train_met[0])
    logging.info(train_met[1])
    logging.info(train_met[2] + '\n')

    logging.info('Valid metrics...')
    valid_met = format_metrics('Valid', best_val_metrics['head_batch'], best_val_metrics['tail_batch'])
    logging.info(valid_met[0])
    logging.info(valid_met[1])
    logging.info(valid_met[2] + '\n')

    logging.info('Test metrics...')
    test_metrics = evaluate(model, 'test', g=g, edge_weight=ksg_eweight)
    test_met = format_metrics('Test', test_metrics['head_batch'], test_metrics['tail_batch'])
    logging.info(test_met[0])
    logging.info(test_met[1])
    logging.info(test_met[2] + '\n')

    logging.info('Model save path: {}'.format(os.getcwd()))

    # 将所有模型结果和保存路径记录到一个统一文件中, 便于日后比较
    # save_log_path = join(cfg.working_dir, 'model_save_log')
    # save_log = open(save_log_path, 'a')
    # msg = ''
    # msg += '*** {} ***\n*** {} ***\n*** {} ***\n'.format(test_met[0], test_met[1], test_met[2])
    # msg += '*** {} ***\n*** {} ***\n*** {} ***\n'.format(valid_met[0], valid_met[1], valid_met[2])
    # msg += '*** {} ***\n*** {} ***\n*** {} ***\n'.format(train_met[0], train_met[1], train_met[2])
    # msg += 'Model save path: {} \n'.format(os.getcwd())
    # config_str = utils.format_config(cfg)
    # msg += config_str
    # msg += '\n\n\n'
    # save_log.write(msg)
    # save_log.close()


if __name__ == '__main__':
    main()
