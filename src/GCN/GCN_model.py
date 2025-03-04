import torch
import torch.nn as nn

def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred == y_true) / len(y_true)


import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_channels, out_feats, num_layers=3):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers

        # Initialize graph convolutional layers
        self.pre_Linear = Linear(in_feats, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_feats)

    def forward(self, x, edge_index, batch):
        x = self.pre_Linear(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        """Reset all parameters in the model."""
        self.pre_Linear.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.classifier.reset_parameters()
