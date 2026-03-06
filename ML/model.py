import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PET_Randoms_GNN(nn.Module):
    def __init__(self, node_in_features=5, hidden_dim=128, num_gat_layers=3, heads=4, dropout_rate=0.2):
        """
        Graph Neural Network for PET List-Mode Coincidence Classification.
        
        Args:
            node_in_features (int): Number of raw features per hit (E, X, Y, Z, T_rel)
            hidden_dim (int): The size of the latent space
            num_gat_layers (int): Number of message-passing hops.
            heads (int): Number of attention heads in GATv2.
            dropout_rate (float): Dropout probability for the MLP head.
        """
        super(PET_Randoms_GNN, self).__init__()
        
        # Node Encoder
        self.node_encoder = nn.Linear(node_in_features, hidden_dim)
        
        # GATv2 Layers
        self.gat_layers = nn.ModuleList()

        for _ in range(num_gat_layers):

            out_channels = hidden_dim // heads
            self.gat_layers.append(GATv2Conv(in_channels=hidden_dim, out_channels=out_channels, heads=heads, concat=True))
            
        # MLP Edge classifier
        classifier_in_dim = (hidden_dim * 2) + 2
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        
        # (Num_Nodes, 5) -> (Num_Nodes, 128)
        h = F.relu(self.node_encoder(x))

        for gat in self.gat_layers:
            h = F.elu(gat(h, edge_index))
            

        row, col = edge_index
        
        node_i_context = h[row]
        node_j_context = h[col]
        
        # Extract physics from the og input tensor 'x'
        # [Energy(0), X(1), Y(2), Z(3), T_rel(4)]
        
        # Calculate delta t
        delta_t = torch.abs(x[row, 4] - x[col, 4]).view(-1, 1)
        
        # Calculate Euclidean Distance between the crystals
        dist = torch.norm(x[row, 1:4] - x[col, 1:4], dim=1).view(-1, 1)
        
        # Shape: (Num_Edges, 258)
        edge_features = torch.cat([node_i_context, node_j_context, delta_t, dist], dim=1)
        
        # (Num_Edges, 258) -> (Num_Edges, 1)
        out = self.edge_classifier(edge_features)
        
        # Flatten 
        return out.view(-1)
