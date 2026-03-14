"""
    This file performs an error analysis on a model
"""

from model import PET_Randoms_GNN
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from model import PET_Randoms_GNN 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\64HD_2GAT_4H.pt"
VAL_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\ML_datasets\testing_dataset"
BATCH_SIZE = 256
HIDDEN_DIM = 64
HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PET_Randoms_GNN(node_in_features=5, hidden_dim=HIDDEN_DIM, num_gat_layers=2, heads = HEADS, dropout_rate=0.2).to(DEVICE)

print("Loading Model...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only = False))
model.eval()

print("Loading validation dataset...")
val = torch.load(VAL_PATH, weights_only=False)

def is_valid(data):
    return (data.x.dim() > 0 and data.y.dim() > 0 and data.edge_index.dim() > 0 and data.edge_index.shape[1] >= 1)

val_dataset = [d for d in val if is_valid(d)]
val_load = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

@torch.no_grad()
def evaluate(loader, threshold=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    #energy_i = []
    #energy_j = []
    triples_checked = 0
    triples_recovered = 0

    for batch in tqdm(loader, desc="Validating"):
        batch = batch.to(DEVICE)
        out = model(batch)

        probs = torch.sigmoid(out)
        preds = (probs > threshold).float()

        energy_floor = 0.350 #set energy floor

        energy_i = batch.x[batch.edge_index[0], 0]
        energy_j = batch.x[batch.edge_index[1], 0]
        energy_mask = (energy_i >= energy_floor) & (energy_j >= energy_floor)

        masked_preds = preds[energy_mask]
        masked_y = batch.y[energy_mask]

        tp += ((masked_preds == 1) & (masked_y  == 1)).sum().item()
        fp += ((masked_preds == 1) & (masked_y  == 0)).sum().item()
        tn += ((masked_preds == 0) & (masked_y  == 0)).sum().item()
        fn += ((masked_preds == 0) & (masked_y  == 1)).sum().item()
        
        
        #tp_mask = ((preds == 1) & (batch.y == 1))

        #fn_edges = batch.edge_index[:, tp_mask]
        #energy_i += batch.x[fn_edges[0], 0].detach().cpu().tolist()
        #energy_j += batch.x[fn_edges[1], 0].detach().cpu().tolist()
        
        
        for i in range(len(batch.ptr) - 1):
            start = batch.ptr[i]
            end = batch.ptr[i+1]
            num_nodes = end - start

            if num_nodes == 3:
                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                
                window_probs = probs[edge_mask]
                window_labels = batch.y[edge_mask]

                if window_labels.sum() == 1:
                    triples_checked += 1
                    
                    # Method 1: Argmax (Winner-take-all)
                    # Does the model give the highest probability to the actual True edge?
                    winner_idx = torch.argmax(window_probs)
                    
                    if window_labels[winner_idx] == 1 and window_probs[winner_idx] > threshold:
                        triples_recovered += 1
        

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2) * precision + recall)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    triple_rate = (triples_recovered / triples_checked * 100) if triples_checked > 0 else 0
    print(triple_rate)
    
    print(f"Accuracy: {accuracy:04f} | f1: {f1:04f} | f05: {f05:04f} | precision: {precision:04f} | recall: {recall:04f}")
    return f"f05: {f05:04f} | precision: {precision:04f} | recall: {recall:04f}"
    '''
    #scale = 1000

    #energy_i = np.array(energy_i) * scale
    #energy_j = np.array(energy_j) * scale
    
    
    plt.hist2d(energy_i, energy_j, bins=100)
    plt.xlabel("Energy of photon 2 (KeV)")
    plt.ylabel("Energy of photon 1 (KeV)")
    plt.title("Energy Distribution of True Positive Pairs")
    plt.colorbar(label="Counts")

    plt.savefig("True_positive_energy_pairs_base_model.png", dpi=300)
    plt.show()
    '''
    return precision, recall, f1, f05

if __name__ == "__main__":
    print(f" Error Analysis (Energy Floor: {0.350*1000} keV)\n")
    lst = []
    for thr in [0.858]:
        print(thr)
        lst.append(evaluate(val_load, threshold=thr))