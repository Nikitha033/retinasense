"""
04_train.py
-----------
Trains RetinaSenseModel jointly on:
  - 6-class disease presence (multi-label with class-weighted focal loss)
  - 5-grade DR severity (masked class-weighted cross-entropy for DR-positive eyes)

Features:
  - Differential learning rates (fine-tunes backbone with lower LR)
  - Positive-weighted Focal Loss for severe class imbalance
  - Cosine annealing scheduler with warm restarts
  - Multi-metric checkpointing (optimizes for Val AUC & Loss)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import importlib.util

def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")

ODIRDataset = ds_mod.ODIRDataset
get_train_transforms = ds_mod.get_train_transforms
get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel
DISEASE_COLS = ds_mod.DISEASE_COLS


def build_model(model_type: str):
    if model_type == "transfer":
        return RetinaSenseModel(pretrained=True, freeze_backbone_layers=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class ClassWeightedFocalLoss(nn.Module):
    """Focal Loss with positive class balancing weights for rare ophthalmic pathologies."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce
        return loss.mean()


def calculate_class_weights(train_csv_path, device):
    """Calculates positive weights for 6 diseases and class weights for 5 DR severity grades."""
    df = pd.read_csv(train_csv_path)
    n_samples = len(df)

    # 1. Disease pos_weights
    pos_counts = df[DISEASE_COLS].sum().values
    neg_counts = n_samples - pos_counts
    # Smooth clamp ratio
    pos_weights = np.clip(np.sqrt(neg_counts / np.maximum(pos_counts, 1)) * 1.5, 1.0, 15.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    # 2. DR severity class weights (grades 0..4)
    dr_eyes = df[df["dr_severity"] >= 0]
    sev_counts = [max(1, (dr_eyes["dr_severity"] == c).sum()) for c in range(5)]
    total_dr = sum(sev_counts)
    sev_weights = [total_dr / (5.0 * cnt) for cnt in sev_counts]
    sev_weights = [w / sum(sev_weights) * 5.0 for w in sev_weights]
    sev_weight_tensor = torch.tensor(sev_weights, dtype=torch.float32, device=device)

    print("\n--- Training Class Weights ---")
    for d, w in zip(DISEASE_COLS, pos_weights):
        print(f"  Disease {d} pos_weight: {w:.2f}")
    for g, w in enumerate(sev_weights):
        print(f"  DR Severity Grade {g} weight: {w:.2f}")

    return pos_weight_tensor, sev_weight_tensor


def masked_severity_loss(severity_logits, severity_targets, severity_weights=None):
    mask = severity_targets >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=severity_logits.device)
    return F.cross_entropy(
        severity_logits[mask],
        severity_targets[mask],
        weight=severity_weights
    )


def train_one_epoch(model, loader, optimizer, disease_loss_fn, sev_weights, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["dr_severity"].to(device)

        optimizer.zero_grad()
        disease_logits, severity_logits = model(images)

        loss_disease = disease_loss_fn(disease_logits, labels)
        loss_severity = masked_severity_loss(severity_logits, severity, sev_weights)
        loss = loss_disease + 0.5 * loss_severity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, disease_loss_fn, sev_weights, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["dr_severity"].to(device)

        disease_logits, severity_logits = model(images)
        loss_disease = disease_loss_fn(disease_logits, labels)
        loss_severity = masked_severity_loss(severity_logits, severity, sev_weights)
        loss = loss_disease + 0.5 * loss_severity
        total_loss += loss.item() * images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(disease_logits).cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    aucs = []
    accs = []
    for i in range(len(DISEASE_COLS)):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            aucs.append(auc)
        except ValueError:
            pass
        acc = accuracy_score(y_true[:, i], (y_prob[:, i] >= 0.5).astype(int))
        accs.append(acc)

    mean_auc = np.mean(aucs) if aucs else 0.0
    mean_acc = np.mean(accs) if accs else 0.0

    return val_loss, mean_auc, mean_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--img_dir", default="dataset/ODIR-5K/Training Images")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--model_type", choices=["transfer", "custom"], default="transfer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=======================================================")
    print(f"           RetinaSense High-Accuracy Training")
    print(f"=======================================================")
    print("Using device:   ", device)
    print("Data directory: ", args.data_dir)
    print("Images directory:", args.img_dir)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Base LR: {args.lr}")

    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "val.csv")

    pos_weights, sev_weights = calculate_class_weights(train_csv, device)

    train_ds = ODIRDataset(train_csv, args.img_dir, get_train_transforms())
    val_ds = ODIRDataset(val_csv, args.img_dir, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model_type).to(device)

    # Differential learning rates: lower for backbone, higher for custom classification heads
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.25},
        {"params": model.disease_head.parameters(), "lr": args.lr},
        {"params": model.severity_head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    disease_loss_fn = ClassWeightedFocalLoss(gamma=2.0, pos_weight=pos_weights)

    best_score = -1.0
    best_val_auc = 0.0
    best_val_acc = 0.0

    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, disease_loss_fn, sev_weights, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, disease_loss_fn, sev_weights, device)
        scheduler.step()

        # Composite evaluation score: combines AUC and Accuracy
        composite_score = 0.6 * val_auc + 0.4 * val_acc

        print(f"Epoch {epoch + 1:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}%")

        if composite_score > best_score:
            best_score = composite_score
            best_val_auc = val_auc
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc*100:.2f}%, Val Acc: {best_val_acc*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete! Best Val AUC: {best_val_auc*100:.2f}%, Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()