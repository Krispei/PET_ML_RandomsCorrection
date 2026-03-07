import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from model import PET_Randoms_GNN  # adjust import to wherever your model is defined

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH       = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\test"
SAVE_PATH       = "best_model.pt"
BATCH_SIZE      = 32
VAL_SPLIT       = 0.2
EPOCHS          = 10
K_FOLDS         = 5
LR              = 1e-3
HIDDEN_DIM      = 128
NUM_GAT_LAYERS  = 2
HEADS           = 4
DROPOUT         = 0.2
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Data ───────────────────────────────────────────────────────────────────────

train_path = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\training_dataset"
val_path = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\validation_dataset"

train = torch.load(train_path, weights_only=False)
val = torch.load(val_path, weights_only=False)

# Shuffle and split

# Fix any graphs with 0-dimensional tensors that break batching
def is_valid(data):
    return (
        data.x.dim() > 0 and
        data.y.dim() > 0 and
        data.edge_index.dim() > 0 and
        data.edge_index.shape[1] >= 1
    )

train_dataset = [d for d in train if is_valid(d)]
val_dataset = [d for d in val if is_valid(d)]


total_dataset = train_dataset + val_dataset

print(f"Total graphs: {len(total_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Total graphs:| Train: {len(train_dataset)} | Val: {len(val_dataset)}")
# ── Class imbalance ────────────────────────────────────────────────────────────
# True coincidences are rare - count pos/neg edges to compute a weight for BCE
num_pos, num_neg = 0, 0
for data in train_dataset:
    num_pos += data.y.sum().item()
    num_neg += (data.y == 0).sum().item()

pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(DEVICE)
print(f"Pos edges: {int(num_pos):,} | Neg edges: {int(num_neg):,} | pos_weight: {pos_weight.item():.2f}")

# ── Model / Optimizer / Loss ───────────────────────────────────────────────────
model = PET_Randoms_GNN(
    node_in_features=5,
    hidden_dim=HIDDEN_DIM,
    num_gat_layers=NUM_GAT_LAYERS,
    heads=HEADS,
    dropout_rate=DROPOUT
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}\n")

# ── Train / Eval functions ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch, fold):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Fold {fold} | Epoch {epoch:03d} [Train]", leave=False)
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
def evaluate(model, loader, criterion, epoch, fold):
    model.eval()
    total_loss = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    pbar = tqdm(loader, desc=f"Fold {fold} | Epoch {epoch:03d} [Val]  ", leave=False)
    for batch in pbar:
        batch = batch.to(DEVICE)
        out  = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item()

        preds = (torch.sigmoid(out) > 0.5).float()
        tp += ((preds == 1) & (batch.y == 1)).sum().item()
        fp += ((preds == 1) & (batch.y == 0)).sum().item()
        tn += ((preds == 0) & (batch.y == 0)).sum().item()
        fn += ((preds == 0) & (batch.y == 1)).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss  = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    return avg_loss, accuracy, precision, recall, f1

# K-Fold Cross Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(total_dataset), 1):
    print(f"\nStarting Fold {fold}/{K_FOLDS}")
    # 1. Create Data Subsets for this fold
    train_sub = [total_dataset[i] for i in train_idx]
    val_sub   = [total_dataset[i] for i in val_idx]
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Recalculate Class Imbalance for ONLY the training subset
    num_pos, num_neg = 0, 0
    for data in train_sub:
        num_pos += data.y.sum().item()
        num_neg += (data.y == 0).sum().item()
    
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)], dtype=torch.float).to(DEVICE)
    
    # 3. Initialize Model, Optimizer, Scheduler newly for each fold
    model = PET_Randoms_GNN(
        node_in_features=5,
        hidden_dim=HIDDEN_DIM,
        num_gat_layers=NUM_GAT_LAYERS,
        heads=HEADS,
        dropout_rate=DROPOUT
    ).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_loss = float("inf")
    fold_history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    # 4. Train loop for this fold
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, fold)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, epoch, fold)
        scheduler.step(val_loss)
        
        fold_history["train_loss"].append(train_loss)
        fold_history["val_loss"].append(val_loss)
        fold_history["val_f1"].append(f1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_save_path = SAVE_PATH.format(fold)
            torch.save(model.state_dict(), current_save_path)
            
    # Save the best metrics for the fold to average later
    fold_results[fold] = best_val_loss

print("\n" + "="*40)
print("Cross-Validation Complete")
for f, loss in fold_results.items():
    print(f"Fold {f} Best Val Loss: {loss:.4f}")
print(f"Average Val Loss: {np.mean(list(fold_results.values())):.4f} ± {np.std(list(fold_results.values())):.4f}")