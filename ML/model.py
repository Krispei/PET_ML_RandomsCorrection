import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PET_Randoms_GNN(nn.Module):
    def __init__(self, node_in_features=5, hidden_dim=128, num_gat_layers=3, heads=4, dropout_rate=0.2):
        """
        Graph Neural Network for PET List-Mode Coincidence Classification.
        
        Args:
            node_in_features (int): Number of raw features per hit (E, X, Y, Z, T_rel). Default is 5.
            hidden_dim (int): The size of the latent space for isotope fingerprinting.
            num_gat_layers (int): Number of message-passing hops.
            heads (int): Number of attention heads in GATv2.
            dropout_rate (float): Dropout probability for the MLP head.
        """
        super(PET_Randoms_GNN, self).__init__()
        
        # 1. Node Encoder
        # Projects the 5 raw physics values into the 128-dimensional latent space
        self.node_encoder = nn.Linear(node_in_features, hidden_dim)
        
        # 2. Message Passing Layers (GATv2)
        # ModuleList allows us to stack multiple layers dynamically
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            # The first layer takes hidden_dim, subsequent layers also take hidden_dim
            # We use 'concat=False' or divide the hidden_dim by heads to keep the output size constant at hidden_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // heads, 
                    heads=heads, 
                    concat=True # Concatenating heads keeps the output at exactly hidden_dim
                )
            )
            
        # 3. Edge Classifier (MLP Head)
        # Input size: 128 (Node A) + 128 (Node B) + 1 (Delta T) + 1 (Distance) = 258
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
        """
        The forward pass of the network.
        Takes a PyTorch Geometric Data object (or Batch) and returns edge probabilities.
        """
        x, edge_index = data.x, data.edge_index
        
        # --- Step 1: Encode Nodes ---
        # Shape goes from [Num_Nodes, 5] -> [Num_Nodes, 128]
        h = F.relu(self.node_encoder(x))
        
        # --- Step 2: Message Passing ---
        # Nodes share information to detect neighborhood anomalies (like prompt gammas)
        for gat in self.gat_layers:
            h = F.elu(gat(h, edge_index))
            
        # --- Step 3: Construct Edge Features ---
        # Extract the indices for the start (row) and end (col) of every potential pair
        row, col = edge_index
        
        # Grab the fully updated 128D context vectors for both hits in the pair
        node_i_context = h[row]
        node_j_context = h[col]
        
        # Extract hard physics from the ORIGINAL input tensor 'x'
        # Assuming x is ordered: [Energy(0), X(1), Y(2), Z(3), T_rel(4)]
        
        # Calculate Delta T (Absolute time difference)
        delta_t = torch.abs(x[row, 4] - x[col, 4]).view(-1, 1)
        
        # Calculate Euclidean Distance between the crystals
        dist = torch.norm(x[row, 1:4] - x[col, 1:4], dim=1).view(-1, 1)
        
        # Fuse everything into a single 258-dimensional vector per edge
        # Shape: [Num_Edges, 258]
        edge_features = torch.cat([node_i_context, node_j_context, delta_t, dist], dim=1)
        
        # --- Step 4: Classify ---
        # Shape goes from [Num_Edges, 258] -> [Num_Edges, 1]
        out = self.edge_classifier(edge_features)
        
        # Flatten from [Num_Edges, 1] to [Num_Edges] for easier loss calculation
        return out.view(-1)

# --- Sanity Check / Initialization ---
if __name__ == "__main__":
    # Instantiate the model
    model = PET_Randoms_GNN(
        node_in_features=5, 
        hidden_dim=128, 
        num_gat_layers=1, 
        heads=4, 
        dropout_rate=0.2
    )
    
    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model successfully instantiated.")
    print(f"Total Trainable Parameters: {total_params:,}")


model = PET_Randoms_GNN(
        node_in_features=5, 
        hidden_dim=128, 
        num_gat_layers=1, 
        heads=4, 
        dropout_rate=0.2
    )

