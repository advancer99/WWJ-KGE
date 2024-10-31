

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
from dgl.nn import EdgeWeightNorm
import utils


class GPR_prop(nn.Module):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, dropout, Gamma=None, bias=True):
        super(GPR_prop, self).__init__()
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))
        self.gcn_norm = EdgeWeightNorm(norm='both')
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, kg, edge_weight=None):
        norm_edge_weight = self.gcn_norm(kg, edge_weight)
        kg.edata['norm'] = norm_edge_weight
        # print(norm_edge_weight.shape)
        # print(self.temp)
        # print(kg.ndata['emb'].shape)
        # kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))
        # kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
        kg.ndata['hid'] = self.dropout(kg.ndata['emb'] * self.temp[0])
        for k in range(self.K):
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'emb'))
            # kg.ndata['emb'] = kg.ndata['neigh'] * kg.edata['norm']
            gamma = self.temp[k+1]
            kg.ndata['hid'] = kg.ndata['hid'] + self.dropout(gamma * kg.ndata['emb'])
        return kg.ndata['hid']

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    def __init__(self, num_feature, hidden_dim, args):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(num_feature, hidden_dim)

        # if args.ppnp == 'PPNP':
        #     self.prop1 = APPNP(args.inner_k, args.alpha)
        if args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.inner_k, args.alpha, args.Init, args.dropout, args.Gamma)
        else:
            raise NotImplementedError

        self.Init = args.Init
        self.dprate = args.dprate
        # self.dropout = args.dropout
        self.dropout = nn.Dropout(args.dropout)
        self.dp = nn.Dropout(args.dprate)
        self.relu = nn.ReLU()

        self.neigh_w = utils.get_param(hidden_dim, hidden_dim)
        self.act = nn.Tanh()
        if args.bn:
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # kg.ndata['emb'] = self.dropout(kg.ndata['emb'])
            # kg.ndata['emb'] = self.relu(self.lin1(kg.ndata['emb']))
            edge_weight = torch.ones(kg.edges()[0].shape[0]).to('cuda')
            kg.ndata['norm'] = self.prop1(kg, edge_weight)

            # kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['norm']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class FALayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(FALayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['emb'], edges.src['emb']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, kg):
        kg.apply_edges(self.edge_applying)
        kg.update_all(fn.u_mul_e('emb', 'e', '_'), fn.sum('_', 'z'))

        return kg


class FAGCN(nn.Module):
    def __init__(self, hidden_dim, args):
        super(FAGCN, self).__init__()
        self.eps = args.eps
        self.layer_num = args.layer_num
        self.dropout = args.dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(hidden_dim, self.dropout))

        # self.reset_parameters()
        self.neigh_w = utils.get_param(hidden_dim, hidden_dim)
        self.act = nn.Tanh()
        if args.bn:
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            raw = ent_emb
            for i in range(self.layer_num):
                kg = self.layers[i](kg)
                kg.ndata['emb'] = self.eps * raw + kg.ndata['emb']
            neigh_ent_emb = kg.ndata['emb']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
