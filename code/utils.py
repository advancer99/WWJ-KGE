# -*- coding=utf-8 -*-
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import random

# 全局设置
CONFIG = OmegaConf.create()

DATASET_STATISTICS = dict(
    FB15k_237=dict(n_ent=14541, n_rel=237, n_train=272115, n_valid=17535, n_test=20466),
    WN18RR=dict(n_ent=40943, n_rel=11, n_train=86835, n_valid=3034, n_test=3134),
    YAGO3_10=dict(n_ent=123182, n_rel=37, n_train=1079040, n_valid=5000, n_test=5000)
)


def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param


def set_global_config(cfg: DictConfig):
    global CONFIG
    CONFIG = cfg


def get_global_config() -> DictConfig:
    global CONFIG
    return CONFIG


def filter_config(cfg: DictConfig):
    """
    过滤无关配置信息, 便于输出.
    """
    filter_keys = ['model_list', 'dataset_list', 'project_dir', 'data_path', 'working_dir']
    new_dict = dict()
    for k, v in cfg.items():
        if k in filter_keys:
            continue
        new_dict[k] = v
    return new_dict


def remove_randomness():
    """
    在一定程度上消除随机因素 (不可能完全消除)
    :return:
    """
    # 固定随机种子生成器
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

