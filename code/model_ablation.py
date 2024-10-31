#!/usr/bin/python3
import torch
import torch.nn as nn
import utils
from utils import get_param
from model import EdgeLayer, SGATLayer, CompLayer
from decoder import ConvE


class LinkPredModel_wo_rel(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.node_layers = nn.ModuleList([SGATLayer(h_dim) for _ in range(self.kg_n_layer)])
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for aggregation
        self.c_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * 2 * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for node_layer, comp_layer, c_rel_emb in zip(self.node_layers, self.comp_layers, self.c_rel_embs):
            ent_emb, c_rel_emb = self.ent_drop(ent_emb), self.rel_drop(c_rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, c_rel_emb)
            ent_emb = ent_emb + node_ent_emb + comp_ent_emb
            rel_emb_list.append(c_rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class LinkPredModel_wo_ent(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for aggregation
        self.e_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])
        self.c_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * 2 * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for edge_layer, comp_layer, e_rel_emb, c_rel_emb in zip(self.edge_layers, self.comp_layers, self.e_rel_embs, self.c_rel_embs):
            ent_emb, c_rel_emb, e_rel_emb = self.ent_drop(ent_emb), self.rel_drop(c_rel_emb), self.rel_drop(e_rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, e_rel_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, c_rel_emb)
            ent_emb = ent_emb + edge_ent_emb + comp_ent_emb
            rel_emb_list.append(c_rel_emb)
            rel_emb_list.append(e_rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class LinkPredModel_wo_comp(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])
        self.node_layers = nn.ModuleList([SGATLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for aggregation
        self.e_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * 2 * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for edge_layer, node_layer, e_rel_emb in zip(self.edge_layers, self.node_layers, self.e_rel_embs):
            ent_emb, e_rel_emb = self.ent_drop(ent_emb), self.rel_drop(e_rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, e_rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb
            rel_emb_list.append(e_rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class LinkPredModel_wo_rel_ent(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for aggregation
        self.c_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * 2 * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for comp_layer, c_rel_emb in zip(self.comp_layers, self.c_rel_embs):
            ent_emb, c_rel_emb = self.ent_drop(ent_emb), self.rel_drop(c_rel_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, c_rel_emb)
            ent_emb = ent_emb + comp_ent_emb
            rel_emb_list.append(c_rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class LinkPredModel_wo_rel_comp(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.node_layers = nn.ModuleList([SGATLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        for node_layer in self.node_layers:
            ent_emb = self.ent_drop(ent_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            ent_emb = ent_emb + node_ent_emb

        pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class LinkPredModel_wo_ent_comp(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)

        # kg agg layer
        self.kg_n_layer = self.cfg.kg_layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation for aggregation
        self.e_rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * 2 * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g):
        """
        输入head和relation (bs, ), 输出每一个实体作为尾节点的分值 (bs, n_ent)。
        两种方案, 一种是输入三元组, 按矩阵来算; 另一种是输入图, 按图来算.
        输入图的话内存并不会小, 还是要存(bs, n_ent, hidden_dim)的数据 (apply_edges)
        :param h_ids:
        :param r_ids:
        :param agg_g:
        :return:
        """
        # 聚合节点embed
        ent_emb, rel_emb = self.aggragate_emb(agg_g)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for edge_layer, e_rel_emb in zip(self.edge_layers, self.e_rel_embs):
            ent_emb, e_rel_emb = self.ent_drop(ent_emb), self.rel_drop(e_rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, e_rel_emb)
            ent_emb = ent_emb + edge_ent_emb
            rel_emb_list.append(e_rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb
