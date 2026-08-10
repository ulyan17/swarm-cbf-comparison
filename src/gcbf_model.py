import torch
import torch.nn as nn

class GCBFPlusModel(nn.Module):
    def __init__(self, node_feat_dim=8, edge_feat_dim=4, hidden=128, dropout=0.2):
        super().__init__()
        self.psi_msg = nn.Sequential(
            nn.Linear(2 * node_feat_dim + edge_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.psi_aggr = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.dropout = nn.Dropout(dropout)
        self.psi_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, node_features, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        h_src = node_features[src]
        h_dst = node_features[dst]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        messages = self.psi_msg(edge_input)

        q = torch.zeros(node_features.size(0), messages.size(1), device=node_features.device)
        q = q.index_add(0, dst, messages)

        q = self.psi_aggr(q)
        q = self.dropout(q)      # регуляризация
        out = self.psi_out(q)
        return out