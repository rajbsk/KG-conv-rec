import math
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv

from utils import _load_kg_embeddings

class SoftAttention(nn.Module):
    def __init__(self, dim):
        super(SoftAttention, self).__init__()
        self.dim = dim
        self.W1 = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        self.W2 = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        self.q = nn.Parameter(torch.zeros(size=(1, self.dim)))
        self.output = nn.Linear(self.dim, 1)
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        nn.init.xavier_uniform_(self.q.data, gain=1.414)

    def forward(self, v):
        vn = v[-1]
        att_weights = []
        for i in range(v.size()[0]):
            inp = torch.matmul(self.W1, v[i]) + torch.matmul(self.W2, vn) + self.output.bias
            att_weight =  torch.matmul(self.q, inp)
            att_weights.append(att_weight)
        att_weights = torch.stack(att_weights)
        output = torch.matmul(torch.transpose(att_weights, 0, 1), v)
        return output.squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)/1000.0
        pe[:, 1::2] = torch.cos(position * div_term)/1000.0
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)

class KGCR(nn.Module):
    def __init__(self, n_entity, n_relation, dim, entity_kg_emb):
        super(KGCR, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.entity_emb.weight.data.copy_(entity_kg_emb)
        self.entity_emb.weight.requires_grad_ = False
        
        # nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        self.criterion = nn.CrossEntropyLoss()

        self.soft_attention = SoftAttention(self.dim)
        self.positional_encoding = PositionalEncoding(self.dim)
        self.output = nn.Linear(self.dim, self.n_entity)

    def forward(self, seed_sets, labels):
        u_emb, nodes_features = self.user_representation(seed_sets)
        
        scores = F.linear(u_emb, nodes_features, self.output.bias)

        base_loss = self.criterion(scores, labels) 

        loss = base_loss 

        return dict(scores=scores.detach(), base_loss=base_loss, loss=loss)

    def user_representation(self, seed_sets):
        nodes_features = self.entity_emb.weight

        user_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if len(seed_set)==0:
                user_representation = (torch.zeros(1, self.dim).cuda())
            else:
                user_representation = (nodes_features[seed_set])
            user_representation = self.positional_encoding(user_representation)
            user_representation = self.soft_attention(user_representation)
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list), nodes_features
