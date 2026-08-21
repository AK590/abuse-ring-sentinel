import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        
        # Layer 1
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names
        }, aggregate='mean')
        
        # Layer 2
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names
        }, aggregate='mean')
        
        self.fc = nn.Linear(out_feats, 1)

    def forward(self, graph, inputs):
        # inputs is a dictionary of node features {ntype: tensor}
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        
        # We only care about scoring users
        user_h = h['user']
        score = torch.sigmoid(self.fc(user_h))
        return score
