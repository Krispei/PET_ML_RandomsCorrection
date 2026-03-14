import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import PET_Randoms_GNN  

# COnfig
BATCH_SIZE = 256
EPOCHS = 150
LR = 2.5e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.90
HIDDEN_DIM = 64
NUM_GAT_LAYERS = 2
HEADS = 4
DROPOUT = 0.2
precision_multiplier = 1
CLASSIFICATION_THRESHOLD = 0.5
ENERGY_FLOOR = 0 
SAVE_PATH = f"64HD_2GAT_4H.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load data
train_path = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\ML_datasets\training_dataset"
val_path = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\ML_datasets\validation_dataset"

train = torch.load(train_path, weights_only=False)
val = torch.load(val_path, weights_only=False)

# Fix any graphs with 0-dimensional tensors that break batching
def is_valid(data):
    return (data.x.dim() > 0 and data.y.dim() > 0 and data.edge_index.dim() > 0 and data.edge_index.shape[1] >= 1)

train_dataset = [d for d in train if is_valid(d)]
val_dataset = [d for d in val if is_valid(d)]

#shuffle training data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Total graphs: Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# Solving class imbalance
num_pos, num_neg = 0, 0

for data in train_dataset:
    num_pos += data.y.sum().item()
    num_neg += (data.y == 0).sum().item()

pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(DEVICE)
precision_multiplier = pos_weight

print(f"Pos edges: {int(num_pos):,}, Neg edges: {int(num_neg):,}, pos_weight: {pos_weight.item():.2f}")

#import model
model = PET_Randoms_GNN(node_in_features=5, hidden_dim=HIDDEN_DIM, num_gat_layers=NUM_GAT_LAYERS, heads=HEADS, dropout_rate=DROPOUT).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#print num of parameters
print(f"Trainable parameters: {total_params:,}\n")

def train_one_epoch(loader, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    
    for batch in pbar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(loader, epoch):
    model.eval()
    total_loss = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)

    for batch in pbar:
        batch = batch.to(DEVICE)
        out  = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item()

        preds = (torch.sigmoid(out) > CLASSIFICATION_THRESHOLD).float()
        
        energy_floor = 0 #set energy floor if neeed

        energy_i = batch.x[batch.edge_index[0], 0]
        energy_j = batch.x[batch.edge_index[1], 0]
        energy_mask = (energy_i >= energy_floor) & (energy_j >= energy_floor)

        masked_preds = preds[energy_mask]
        masked_y = batch.y[energy_mask]

        tp += ((masked_preds == 1) & (masked_y  == 1)).sum().item()
        fp += ((masked_preds == 1) & (masked_y  == 0)).sum().item()
        tn += ((masked_preds == 0) & (masked_y  == 0)).sum().item()
        fn += ((masked_preds == 0) & (masked_y  == 1)).sum().item()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss  = total_loss / len(loader)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    return avg_loss, accuracy, precision, recall, f1, f05

history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_f05": []}
best_f05 = 0

epoch_pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")

for epoch in epoch_pbar:
    train_loss = train_one_epoch(train_loader, epoch)
    val_loss, accuracy, precision, recall, f1, f05= evaluate(val_loader, epoch)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(f1)
    history["val_f05"].append(f05)

    #save based on f05 or validation loss
    if f05 > best_f05:
        best_f05 = f05
        torch.save(model.state_dict(), SAVE_PATH)
        saved_marker = " - saved"

    else:
        saved_marker = ""


    epoch_pbar.set_postfix( train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", f1=f"{f1:.3f}")

    print(
        f"\nEpoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {accuracy:.3f} | "
        f"Prec: {precision:.3f} | "
        f"Rec: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"F05: {f05:.3f}" 
        f"{saved_marker}"
    )


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"],   label="Val Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss"); ax1.legend()

ax2.plot(history["val_f1"], label="F1 Score", linestyle='--', alpha=0.6)
ax2.plot(history["val_f05"], label="F0.5 Score (Optimized)", lw=2, color='green')
ax2.set_title("Metric Evolution"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.tight_layout()

training_curve_file = f"training_curves_BS{BATCH_SIZE}_LR{LR}_WD{WEIGHT_DECAY}_M{MOMENTUM}_HD{HIDDEN_DIM}_NGL{NUM_GAT_LAYERS}_H{HEADS}_D{DROPOUT}_PM{2}_CT{CLASSIFICATION_THRESHOLD}.png"
plt.savefig("64HD_2GAT_4H_final", dpi=300)
plt.show()

print(f"\nTraining complete. Best f05: {best_f05:.4f}. saved to '{SAVE_PATH}'") 