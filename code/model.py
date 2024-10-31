#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import utils
from utils import get_param
from decoder import ConvE
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from model_zlx_add import GPRGNN, FAGCN


class KGEModel(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device

        # 初始化相关embedding
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.n_layer = self.cfg.n_layer
        self.ent_emb = get_param(self.n_ent, h_dim)

        gnn = self.cfg.gnn
        if gnn == 'SGAT':
            self.gnn_layers = nn.ModuleList([SGATLayer(h_dim) for _ in range(self.n_layer)])
        elif gnn == 'GCN':
            self.gnn_layers = nn.ModuleList()
            for _ in range(self.n_layer):
                layer = GraphConv(in_feats=h_dim, out_feats=h_dim,
                                  norm='both', weight=True, bias=True,
                                  activation=nn.Tanh() if self.cfg.act else None,
                                  allow_zero_in_degree=True)
                self.gnn_layers.append(layer)
        elif gnn == 'GAT':
            self.gnn_layers = nn.ModuleList()
            for _ in range(self.n_layer):
                layer = GATConv(in_feats=h_dim, out_feats=h_dim, num_heads=self.cfg.gat_nhead,
                                feat_drop=self.cfg.gat_fdrop, attn_drop=self.cfg.gat_adrop, negative_slope=0.2,
                                activation=nn.Tanh() if self.cfg.act else None,
                                residual=False, allow_zero_in_degree=True, bias=True)
                self.gnn_layers.append(layer)
        elif gnn in ['GraphSAGE_mean', 'GraphSAGE_gcn',
                     'GraphSAGE_pool', 'GraphSAGE_lstm']:
            agg_type = gnn.split('_')[-1]
            self.gnn_layers = nn.ModuleList()
            for _ in range(self.n_layer):
                layer = SAGEConv(in_feats=h_dim, out_feats=h_dim, aggregator_type=agg_type,
                                 activation=nn.Tanh() if self.cfg.act else None,
                                 feat_drop=self.cfg.sage_fdrop, bias=True, norm=None)
                self.gnn_layers.append(layer)
        elif gnn == 'GPRGNN':
            self.gnn_layers = nn.ModuleList([GPRGNN(h_dim, h_dim, self.cfg) for _ in range(self.n_layer)])
        elif gnn == 'FAGCN':
            self.gnn_layers = nn.ModuleList([FAGCN(h_dim, self.cfg) for _ in range(self.n_layer)])
        else:
            raise NotImplementedError

        self.rel_emb = get_param(self.n_rel * 2, h_dim)
        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)

        # loss函数
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)

        self.act = nn.Tanh()

    def forward(self, h_ids, r_ids, agg_g, edge_weight):
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
        ent_emb, rel_emb = self.aggragate_emb(agg_g, edge_weight)

        # 计算分值
        head = ent_emb[h_ids]
        rel = rel_emb[r_ids]
        score = self.predictor(head, rel, ent_emb)

        return score, ent_emb, rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, g, edge_weight):
        """
        聚合所有节点的embedding
        """
        ent_emb = self.ent_emb
        # edge_weight = torch.tensor(edge_weight).to('cuda')

        for layer in self.gnn_layers:
            ent_emb = self.ent_drop(ent_emb)
            node_ent_emb = layer(g, ent_emb)
            if self.cfg.gnn == 'GAT':
                # (n_node, n_head, h)
                node_ent_emb = node_ent_emb.mean(dim=1)
            if self.cfg.gnn not in ['GPRGNN', 'FAGCN']:
                ent_emb = ent_emb + node_ent_emb

        return ent_emb, self.rel_emb


class SGATLayer(nn.Module):
    """
    Simple GAT Layer
    """
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        """
        聚合节点的embedding
        """
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # 计算attention
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            if self.cfg.act:
                neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class CompLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        """
        聚合节点和边的embedding
        """
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            if self.cfg.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.cfg.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # 计算attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = utils.get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        """
        聚合边的embedding
        """
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # 计算attention
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
