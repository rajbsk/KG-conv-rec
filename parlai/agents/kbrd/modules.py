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
from .RGCNMainConv import RGCNConv

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

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

class SparseAutoencoder(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(n_inp, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inp)

    def forward(self, x):
        encoded = F.tanh(self.encoder(x))
        decoded = F.tanh(self.decoder(encoded))
        return encoded, decoded

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

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        N = h.shape[0]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, h)



def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    # edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 10 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 10], len(relation_idx)

class KBRD(nn.Module):
    def __init__(
        self,
        n_entity,
        n_relation,
        dim,
        n_hop,
        kge_weight,
        l2_weight,
        n_memory,
        item_update_mode,
        using_all_hops,
        kg,
        entity_kg_emb,
        entity_text_emb,
        num_bases
    ):
        super(KBRD, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.entity_emb.weight.data.copy_(entity_kg_emb)
        self.entity_emb.weight.requires_grad_ = False
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        
        # nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        self.criterion = nn.CrossEntropyLoss()
        self.zero_shot_criterion = nn.L1Loss()
        self.autoencoder_criterion = nn.L1Loss()
        self.kge_criterion = nn.Softplus()

        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        self.soft_attention = SoftAttention(self.dim)
        self.positional_encoding = PositionalEncoding(self.dim)
        self.kbrd_transformer_encoder = TransformerModel(self.dim, 1, self.dim, 1)
        self.autoencoder = SparseAutoencoder(self.dim, 128)

        self.output = nn.Linear(self.dim, self.n_entity)

        self.kg = kg

        edge_list, self.n_relation = _edge_list(self.kg, self.n_entity, hop=2)
        self.rgcn = RGCNConv(self.dim, self.dim, self.n_relation, num_bases=num_bases)
        edge_list = list(set(edge_list))
        edge_list_tensor = torch.LongTensor(edge_list).cuda()
        self.edge_idx = edge_list_tensor[:, :2].t()
        self.edge_type = edge_list_tensor[:, 2]

    def _get_triples(self, kg):
        triples = []
        for entity in kg:
            for relation, tail in kg[entity]:
                if entity != tail:
                    triples.append([entity, relation, tail])
        return triples

    def forward(
        self,
        seed_sets: list,
        labels: torch.LongTensor,
    ):


        u_emb, nodes_features = self.user_representation(seed_sets)
        input_representation = torch.cat((u_emb, nodes_features), dim=0)
        
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
