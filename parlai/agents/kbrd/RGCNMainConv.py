import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros, uniform


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            x_ju = x_j.unsqueeze(1)
            outs = []
            for i_idx in range(0, x_j.size()[0], 1000):
                w_s = torch.index_select(w, 0, edge_type[i_idx:i_idx+1000])
                x_js = x_ju[i_idx: i_idx+1000, :, :]
                out_s = torch.bmm(x_js, w_s).squeeze(-2)
                outs.append(out_s.data)
#                 del out_s
#                 del w_s
#                 del x_js
#                 torch.cuda.empty_cache()
            out = torch.cat(outs, 0).cuda()

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)