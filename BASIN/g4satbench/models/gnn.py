import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from g4satbench.models.mlp import MLP
from g4satbench.models.ln_lstm_cell import LayerNormBasicLSTMCell
from torch_scatter import scatter_sum, scatter_mean
from g4satbench.utils.attention import AttentionAggregation

class CryptoANFNet(nn.Module):
    def __init__(self, opts):
        super(CryptoANFNet, self).__init__()
        self.opts = opts
        self.pos_l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim * 2, self.opts.dim,
                                    self.opts.activation)
        self.neg_l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim * 2, self.opts.dim,
                                    self.opts.activation)
        self.pos_c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                    self.opts.activation)
        self.neg_c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                    self.opts.activation)
        self.pos_c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.neg_c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.l_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        
        self.l2l_center_attention = AttentionAggregation(self.opts.dim)
        self.c2l_attention = AttentionAggregation(self.opts.dim)
        self.l2l_attention = AttentionAggregation(self.opts.dim)
        self.pos_l2c_attention = AttentionAggregation(self.opts.dim)
        self.neg_l2c_attention = AttentionAggregation(self.opts.dim)

    def forward(self, l_size, c_size, h_size, l_edge_index, c_edge_index, l1_edge_index,
                l2_edge_index, l_emb, pos_c_emb, neg_c_emb):
        # x_i x_j (i <= j) 
        l2l_size = h_size
        assert c_size % 2 == 0
        # literal node state
        l_state = torch.zeros(l_size, self.opts.dim).to(self.opts.device)
        # clause node state
        pos_c_state = torch.zeros(c_size // 2, self.opts.dim).to(self.opts.device)
        neg_c_state = torch.zeros(c_size // 2, self.opts.dim).to(self.opts.device)
        l_embs = [l_emb]
        pos_c_embs = [pos_c_emb]
        neg_c_embs = [neg_c_emb]

        for i in range(self.opts.n_iterations):

            l2l_msg = l_emb[l1_edge_index]
            # l_emb_center = scatter_sum(l2l_msg, l2_edge_index, dim=0, dim_size=l2l_size)
            l_emb_center = self.l2l_center_attention(l2l_msg, l2_edge_index, l2l_size)

            pos_c_msg_feat = self.pos_c2l_msg_func(pos_c_emb)
            neg_c_msg_feat = self.neg_c2l_msg_func(neg_c_emb)
            c_msg_feat = torch.stack((pos_c_msg_feat, neg_c_msg_feat), dim=1).reshape(-1, self.opts.dim)
            # aggregate clause/negated literal to literal messages
            c2l_msg = c_msg_feat[c_edge_index]
            #c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l2l_size)
            c2l_msg_aggr = self.c2l_attention(c2l_msg, l_edge_index, l2l_size)
            c2l_msg_aggr = c2l_msg_aggr[l2_edge_index]
            #l2l_msg_aggr = scatter_sum(c2l_msg_aggr, l1_edge_index, dim=0, dim_size=l_size)
            l2l_msg_aggr = self.l2l_attention(c2l_msg_aggr, l1_edge_index, l_size)
            l_emb, l_state = self.l_update(l2l_msg_aggr, (l_emb, l_state))
            l_embs.append(l_emb)

            pos_l_msg_feat = self.pos_l2c_msg_func(l_emb_center)
            neg_l_msg_feat = self.neg_l2c_msg_func(l_emb_center)
            # aggregate literal to clause messages
            pos_l2c_msg = pos_l_msg_feat[l_edge_index]
            neg_l2c_msg = neg_l_msg_feat[l_edge_index]

            pos_c_indices = np.where(c_edge_index.cpu().numpy() % 2 == 0)[0]
            neg_c_indices = np.where(c_edge_index.cpu().numpy() % 2 == 1)[0]
            #pos_l2c_msg_aggr = scatter_sum(pos_l2c_msg[pos_c_indices], c_edge_index[pos_c_indices] // 2, dim=0,
            #                               dim_size=c_size // 2)
            pos_l2c_msg_aggr = self.pos_l2c_attention(pos_l2c_msg[pos_c_indices], c_edge_index[pos_c_indices] // 2,c_size // 2)
            #neg_l2c_msg_aggr = scatter_sum(neg_l2c_msg[neg_c_indices], c_edge_index[neg_c_indices] // 2, dim=0,
            #                               dim_size=c_size // 2)
            neg_l2c_msg_aggr = self.neg_l2c_attention(neg_l2c_msg[neg_c_indices], c_edge_index[neg_c_indices] // 2, c_size // 2)
            pos_c_emb, pos_c_state = self.pos_c_update(pos_l2c_msg_aggr, (pos_c_emb, pos_c_state))
            neg_c_emb, neg_c_state = self.neg_c_update(neg_l2c_msg_aggr, (neg_c_emb, neg_c_state))
            pos_c_embs.append(pos_c_emb)
            neg_c_embs.append(neg_c_emb)

        return l_embs, pos_c_embs, neg_c_embs


class NeuroSAT(nn.Module):
    def __init__(self, opts):
        super(NeuroSAT, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.l_update = LayerNormBasicLSTMCell(self.opts.dim * 2, self.opts.dim)

    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        # literal node state
        l_state = torch.zeros(l_size, self.opts.dim).to(self.opts.device)
        # clause node state
        c_state = torch.zeros(c_size, self.opts.dim).to(self.opts.device)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)

            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)

            # update clause embeddings
            c_emb, c_state = self.c_update(l2c_msg_aggr, (c_emb, c_state))
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb, l_state = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1), (l_emb, l_state))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GGNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GGNN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c_update = nn.GRUCell(self.opts.dim, self.opts.dim)
        self.l_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)

    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)

            # update clause embeddings
            c_emb = self.c_update(input=l2c_msg_aggr, hx=c_emb)
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb = self.l_update(input=torch.cat([c2l_msg_aggr, l2l_msg], dim=1), hx=l_emb)

            l_embs.append(l_emb)

        return l_embs, c_embs


class GCN_LCG(nn.Module):
    def __init__(self, opts):
        super(GCN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)

        self.c_update = nn.Linear(self.opts.dim * 2, self.opts.dim)
        self.l_update = nn.Linear(self.opts.dim * 3, self.opts.dim)

    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_one = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
        l_deg = scatter_sum(l_one, l_edge_index, dim=0, dim_size=l_size)
        c_one = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
        c_deg = scatter_sum(c_one, c_edge_index, dim=0, dim_size=c_size)
        degree_norm = l_deg[l_edge_index].pow(0.5) * c_deg[c_edge_index].pow(0.5)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg / degree_norm, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg / degree_norm, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GIN_LCG(nn.Module):
    def __init__(self, opts):
        super(GIN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                self.opts.activation)

        self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim,
                            self.opts.activation)
        self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim,
                            self.opts.activation)

    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GNN_LCG, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))

        if self.opts.model == 'neurosat':
            self.gnn = NeuroSAT(self.opts)
        elif self.opts.model == 'ggnn':
            self.gnn = GGNN_LCG(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_LCG(self.opts)
        elif self.opts.model == 'gin':
            self.gnn = GIN_LCG(self.opts)

        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            # self.opts.task == 'assignment' or self.opts.task == 'core_variable'
            self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, 1, self.opts.activation)

        if hasattr(self.opts, 'use_contrastive_learning'):
            # temperature parameter
            self.tau = 0.5
            # projection head
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

    def forward(self, data):
        batch_size = data.num_graphs
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        else:
            # self.opts.init_emb == 'random'
            l_emb = torch.randn(l_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            c_emb = torch.randn(c_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)

        l_embs, c_embs = self.gnn(l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb)

        if self.opts.task == 'satisfiability':
            l_batch = data.l_batch
            g_emb = scatter_mean(l_embs[-1], l_batch, dim=0, dim_size=batch_size)

            if hasattr(self.opts, 'use_contrastive_learning'):
                # SimCLR
                g_emb = self.proj(g_emb)
                h = F.normalize(g_emb, dim=1)
                sim = torch.exp(torch.mm(h, h.t()) / self.tau)
                # remove the similarity measure between two same objects
                mask = (1 - torch.eye(batch_size, device=self.opts.device))
                sim = sim * mask
                return sim
            else:
                g_logit = self.g_readout(g_emb).reshape(-1)
                return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            if not hasattr(self.opts, 'decoding') or self.opts.decoding == 'standard':
                v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim * 2)).reshape(-1)
                return torch.sigmoid(v_logit)

            elif self.opts.decoding == '2-clustering':
                # perform 2 clustering
                v_assign1 = []
                v_assign2 = []
                idx = 0
                for l_num in data.l_size:
                    l_emb = l_embs[-1][idx:idx + l_num]
                    idx += l_num

                    _, centers = kmeans(X=l_emb, num_clusters=2, distance='euclidean', tqdm_flag=0,
                                        device=self.opts.device)
                    distance1 = l_emb - centers[[0], :]
                    distance1 = (distance1 * distance1).sum(dim=1)

                    distance2 = l_emb - centers[[1], :]
                    distance2 = (distance2 * distance2).sum(dim=1)

                    pl_distance2, nl_distance2 = torch.chunk(distance2.reshape(-1, 2), 2, 1)
                    neg_distance2 = torch.cat([nl_distance2, pl_distance2], dim=1).reshape(-1)

                    distance = (distance1 + neg_distance2).reshape(-1, 2)
                    assignment1 = torch.argmin(distance, dim=1)
                    assignment2 = 1 - assignment1

                    v_assign1.append(assignment1)
                    v_assign2.append(assignment2)

                return [torch.cat(v_assign1, dim=0), torch.cat(v_assign2, dim=0)]

            else:
                assert self.opts.decoding == 'multiple_assignments'
                v_assigns = []
                for l_emb in l_embs:
                    v_logit = self.l_readout(l_emb.reshape(-1, self.opts.dim * 2)).reshape(-1)
                    v_assigns.append(torch.sigmoid(v_logit))
                return v_assigns

        else:
            assert self.opts.task == 'core_variable'
            v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim * 2)).reshape(-1)
            return torch.sigmoid(v_logit)


class GGNN_VCG(nn.Module):
    def __init__(self, opts):
        super(GGNN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)

        self.c_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
        self.v_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)

    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1), c_emb)
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1), v_emb)
            v_embs.append(v_emb)

        return v_embs, c_embs


class GCN_VCG(nn.Module):
    def __init__(self, opts):
        super(GCN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)

        self.c_update = nn.Linear(self.opts.dim * 3, self.opts.dim)
        self.v_update = nn.Linear(self.opts.dim * 3, self.opts.dim)

    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        # calculate the degree of each clause/variable node
        p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
        p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
        p_v_deg[p_v_deg < 1] = 1
        n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
        n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
        n_v_deg[n_v_deg < 1] = 1

        p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
        p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
        p_c_deg[p_c_deg < 1] = 1
        n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
        n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
        n_c_deg[n_c_deg < 1] = 1

        p_norm = p_v_deg[v_edge_index[p_edge_index]].pow(0.5) * p_c_deg[c_edge_index[p_edge_index]].pow(0.5)
        n_norm = n_v_deg[v_edge_index[n_edge_index]].pow(0.5) * n_c_deg[c_edge_index[n_edge_index]].pow(0.5)

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg / p_norm, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg / n_norm, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg / p_norm, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg / n_norm, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([v_emb, p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))
            v_embs.append(v_emb)

        return v_embs, c_embs


class GIN_VCG(nn.Module):
    def __init__(self, opts):
        super(GIN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim,
                                  self.opts.activation)

        self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim,
                            self.opts.activation)
        self.v_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim,
                            self.opts.activation)

    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([v_emb, p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))
            v_embs.append(v_emb)

        return v_embs, c_embs


class GNN_VCG(nn.Module):
    def __init__(self, opts):
        super(GNN_VCG, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.v_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))

        if self.opts.model == 'ggnn':
            self.gnn = GGNN_VCG(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_VCG(self.opts)
        elif self.opts.model == 'gin':
            self.gnn = GIN_VCG(self.opts)

        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            # self.opts.task == 'assignment' or self.opts.task == 'core_variable'
            self.v_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)

        if hasattr(self.opts, 'use_contrastive_learning'):
            # temperature parameter
            self.tau = 0.5
            # projection head
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

    def forward(self, data):
        batch_size = data.num_graphs
        v_size = data.v_size.sum().item()
        c_size = data.c_size.sum().item()

        v_edge_index = data.v_edge_index
        c_edge_index = data.c_edge_index
        p_edge_index = data.p_edge_index
        n_edge_index = data.n_edge_index

        if self.opts.init_emb == 'learned':
            v_emb = (self.v_init).repeat(v_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        else:
            # self.opts.init_emb == 'random'
            v_emb = torch.randn(v_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            c_emb = torch.randn(c_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)

        v_embs, c_embs = self.gnn(v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb)

        if self.opts.task == 'satisfiability':
            v_batch = data.v_batch
            g_emb = scatter_mean(v_embs[-1], v_batch, dim=0, dim_size=batch_size)

            if hasattr(self.opts, 'use_contrastive_learning'):
                # SimCLR
                g_emb = self.proj(g_emb)
                h = F.normalize(g_emb, dim=1)
                sim = torch.exp(torch.mm(h, h.t()) / self.tau)
                # remove the similarity measure between two same objects
                mask = (1 - torch.eye(batch_size, device=self.opts.device))
                sim = sim * mask
                return sim
            else:
                g_logit = self.g_readout(g_emb).reshape(-1)
                return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            # 2-clustering can not be applied to VCG
            if not hasattr(self.opts, 'decoding') or self.opts.decoding == 'standard':
                v_logit = self.v_readout(v_embs[-1]).reshape(-1)
                return torch.sigmoid(v_logit)
            else:
                assert self.opts.decoding == 'multiple_assignments'
                v_assigns = []
                for v_emb in v_embs:
                    v_logit = self.v_readout(v_emb).reshape(-1)
                    v_assigns.append(torch.sigmoid(v_logit))
                return v_assigns

        else:
            assert self.opts.task == 'core_variable'
            v_logit = self.v_readout(v_embs[-1]).reshape(-1)
            return torch.sigmoid(v_logit)


class GNN_ANF(nn.Module):
    def __init__(self, opts):
        super(GNN_ANF, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_pos_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_neg_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))

        if self.opts.model == 'cryptoanfnet':
            self.gnn = CryptoANFNet(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_ANF(self.opts)

        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
            self.g_neg_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)

        if hasattr(self.opts, 'use_contrastive_learning'):
            # temperature parameter
            self.tau = 0.5
            # projection head
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

    def forward(self, data):
        batch_size = data.num_graphs
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        h_size = data.h_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index
        l1_edge_index = data.l1_edge_index
        l2_edge_index = data.l2_edge_index

        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1)
            pos_c_emb = (self.c_pos_init).repeat(c_size // 2, 1)
            neg_c_emb = (self.c_neg_init).repeat(c_size // 2, 1)
        else:
            l_emb = torch.randn(l_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            pos_c_emb = torch.randn(c_size // 2, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            neg_c_emb = torch.randn(c_size // 2, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)

        l_embs, pos_c_embs, neg_c_embs = self.gnn(l_size, c_size, h_size, l_edge_index, c_edge_index, 
                                                  l1_edge_index, l2_edge_index, l_emb, pos_c_emb, neg_c_emb)

        if self.opts.task == 'satisfiability':

            l_batch = data.l_batch
            c_batch = data.c_batch

            # pc_emb, uc_emb = torch.chunk(c_embs[-1].reshape(c_size // 2, -1), 2, 1)
            c_batch, _ = torch.chunk(c_batch.reshape(c_batch.shape[0] // 2, -1), 2, 1)

            g_logit = self.g_readout(pos_c_embs[-1])
            neg_g_logit = self.g_neg_readout(neg_c_embs[-1])

            g_logit = scatter_mean(g_logit, c_batch, dim=0, dim_size=batch_size)
            neg_g_logit = scatter_mean(neg_g_logit, c_batch, dim=0, dim_size=batch_size)

            return torch.sigmoid(((g_logit + neg_g_logit) / 2).reshape(-1))
        elif self.opts.task == 'assignment':
            v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim)).reshape(-1)
            return torch.sigmoid(v_logit)


def GNN(opts):
    if opts.graph == 'lcg':
        return GNN_LCG(opts)
    elif opts.graph == 'vcg':
        return GNN_VCG(opts)
    else:
        return GNN_ANF(opts)
