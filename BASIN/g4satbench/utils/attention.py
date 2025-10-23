import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(dim*2, 1)
        
    def forward(self, src, dst, edge_index):
        # src: source mode features
        # dst: destination node features
        #edge_index: graph connectivity
        edge_src, edge_dsr = edge_index
        
        # compute attention scores
        attn_input = torch.cat([src[edge_src], dst[edge_dst]], dim=1)
        attn_scores = self.attn(attn_input).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=0)

        # Apply attention weights
        weighted_msg = src[edge_src] * attn_weights.unsqueeze(-1)
        
        return weight_msg

class AttentionAggregation(nn.Module):
    def __init__(self, dim):
        super(AttentionAggregation, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x, index, dim_size):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)  #matrix:(num_edges, opts.dim)

        #Compute attention weights
	
        #weighted_msg = torch.zeros_like(x)	#store the results
        #for i in range(weighted_msg.shape[0]):
        #    Q_i = Q[i]
        #    K_i = K[i]
        #    V_i = V[i]	#vector, (dim,)
        #    weight_msg_i = torch.zeros_like(Q_i)
        #    for j in range(weight_msg_i.shape[0]):
        #        w_i = torch.sigmoid(K_i[j] * Q_i)
        #        weight_msg_i += V_i[j] * w_i
        #    weighted_msg[i] = weight_msg_i

        attn_weights = torch.sigmoid(torch.sum(K * Q, dim=-1, keepdim=True))

        weighted_msg = V * attn_weights

        out = scatter_sum(weighted_msg, index, dim=0, dim_size=dim_size)

        return out



