import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNModel, self).__init__()
        # Define layers of the GNN model
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = nn.Linear(16, num_classes)

    def forward(self, data):
        # Extract features and edge index from data
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling (to summarize graph-level information)
        x = torch.mean(x, dim=0)

        # Fully connected layer to output classes
        x = self.fc1(x)
        return x