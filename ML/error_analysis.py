"""
    This file performs an error analysis on a model
"""

from model import PET_Randoms_GNN
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from model import PET_Randoms_GNN 
from tqdm import tqdm

MODEL_PATH = ""
VAL_PATH = ""
BATCH_SIZE = 256
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PET_Randoms_GNN(
    node_in_features=5,
    hidden_dim=HIDDEN_DIM,
    num_gat_layers=2,
    heads = 4,
    dropout_rate=0.2
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only = False))
model.eval()

val_dataset = torch.load(VAL_PATH, weights_only=False)
val_load = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

@torch.no_grad()
def evaluate(loader, threshold=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for batch in tqdm(loader, desc="Validating"):
        batch = batch.to(DEVICE)
        out = model(batch)

        predictions = (torch.sigmoid(out) > threshold).float()

        tp += ((predictions == 1) & (batch.y == 1)).sum().item()
        tn += ((predictions == 0) & (batch.y == 0)).sum().item()
        fp += ((predictions == 1) & (batch.y == 0)).sum().item()
        fn += ((predictions == 0) & (batch.y == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f05 = (1 + 0.5**2) (precision * recall) / ((0.5**2) * precision + recall)
    
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)


    return precision, recall, f1, f05

if __name__ == "__main__":
    evaluate(val_load)
