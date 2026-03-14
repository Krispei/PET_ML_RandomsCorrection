import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from model import PET_Randoms_GNN  # Ensure this matches your filename

MODEL_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\64HD_2GAT_4H.pt" 
VAL_DATA_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\ML_datasets\validation_dataset"
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_pr_curve():
   
    model = PET_Randoms_GNN(node_in_features=5, hidden_dim=64, num_gat_layers=2, heads=4).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    val_list = torch.load(VAL_DATA_PATH, weights_only=False)

    def is_valid(data):
        return (data.x.dim() > 0 and data.y.dim() > 0 and data.edge_index.dim() > 0 and data.edge_index.shape[1] >= 1)

    val_dataset = [d for d in val_list if is_valid(d)]

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(val_list)} windows for evaluation.")

    all_probs = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())

    y_scores = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Purity)')
    plt.title('Precision-Recall Curve: PET Randoms Correction')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.savefig("precision_recall_curve_final.png", dpi=300)
    plt.show()

    print(f"Average Precision Score: {average_precision:.4f}")
    
    target_recall = 0.90
    idx = (np.abs(recall - target_recall)).argmin()
    print(f"--- Threshold Analysis ---")
    print(f"To achieve {recall[idx]*100:.1f}% Recall:")
    print(f"Set Threshold to: {thresholds[idx]:.4f}")
    print(f"Resulting Precision: {precision[idx]*100:.1f}%")

if __name__ == "__main__":
    import numpy as np
    plot_pr_curve()