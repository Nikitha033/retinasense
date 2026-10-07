"""
04_train.py
-----------
Controlled High-Precision Training Suite for RetinaSense EfficientNet-B0:
  - Backbone: ImageNet-pretrained EfficientNet-B0 ONLY
  - Multi-label disease detection: Asymmetric Loss (ASL) with calibrated positive margin & class weighting
  - DR severity grading: Masked class-weighted cross-entropy with label smoothing
  - Preserved patient-wise train/val/test integrity (test set kept strictly untouched)
  - Validation-only checkpoint selection (highest composite score on val split)
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import importlib.util


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model():
    return RetinaSenseModel(pretrained=True, freeze_backbone_layers=False)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label ophthalmic image classification.
    
    Decouples positive and negative focusing parameters:
      - Positives: gamma_pos = 0 (never suppressed, maximizing sensitivity on rare D/A/H)
      - Negatives: gamma_neg = 2.0 with margin clip = 0.05 (zeros out easy normal backgrounds)
      - Combines with calibrated positive class weighting.
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, pos_weight=None, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)

        # Positive loss
        log_pos = torch.log(p.clamp(min=self.eps))
        loss_pos = targets * log_pos
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1.0 - p) ** self.gamma_pos)
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        # Negative loss with margin clipping
        p_neg = (p - self.clip).clamp(min=0.0)
        log_neg = torch.log((1.0 - p_neg).clamp(min=self.eps))
        loss_neg = (1.0 - targets) * (p_neg ** self.gamma_neg) * log_neg

        loss = - (loss_pos + loss_neg)
        return loss.mean()


def calculate_class_weights(train_csv_path, device):
    """Calculates balanced positive weights for 6 diseases and class weights for 5 DR severity grades."""
    df = pd.read_csv(train_csv_path)
    n_samples = len(df)

    # Calibrated positive weights: D=1.0, rare diseases given moderate boost to prevent under-prediction
    pos_counts = df[DISEASE_COLS].sum().values
    neg_counts = n_samples - pos_counts
    pos_weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1))
    pos_weights[0] = 1.0  # Keep D balanced to maximize accuracy
    pos_weights = np.clip(pos_weights, 1.0, 3.2)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    # DR severity class weights
    dr_eyes = df[df["dr_severity"] >= 0]
    sev_counts = [max(1, (dr_eyes["dr_severity"] == c).sum()) for c in range(5)]
    total_dr = sum(sev_counts)
    sev_weights = [total_dr / (5.0 * cnt) for cnt in sev_counts]
    sev_weights = [np.power(w, 0.45) for w in sev_weights]
    sev_weights = [w / sum(sev_weights) * 5.0 for w in sev_weights]
    sev_weight_tensor = torch.tensor(sev_weights, dtype=torch.float32, device=device)

    print("\n--- Controlled Training Class Weights ---")
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
        weight=severity_weights,
        label_smoothing=0.03
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
        loss = loss_disease + 0.85 * loss_severity

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
    all_sev_true = []
    all_sev_pred = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["dr_severity"].to(device)

        disease_logits, severity_logits = model(images)
        loss_disease = disease_loss_fn(disease_logits, labels)
        loss_severity = masked_severity_loss(severity_logits, severity, sev_weights)
        loss = loss_disease + 0.85 * loss_severity
        total_loss += loss.item() * images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(disease_logits).cpu().numpy())

        mask = (severity >= 0).cpu().numpy()
        if mask.sum() > 0:
            all_sev_true.extend(severity[mask].cpu().numpy().tolist())
            all_sev_pred.extend(severity_logits[mask].argmax(dim=1).cpu().numpy().tolist())

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
        acc = accuracy_score(y_true[:, i], (y_prob[:, i] >= 0.50).astype(int))
        accs.append(acc)

    mean_auc = np.mean(aucs) if aucs else 0.0
    mean_acc = np.mean(accs) if accs else 0.0
    sev_acc = accuracy_score(all_sev_true, all_sev_pred) if all_sev_true else 0.0

    return val_loss, mean_auc, mean_acc, sev_acc


def main():
    ap = argparse.ArgumentParser(description="Train RetinaSense EfficientNet-B0")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--img_dir", default="dataset/ODIR-5K/Training Images")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./models")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=======================================================")
    print(f"    RetinaSense Controlled EfficientNet-B0 Training")
    print(f"=======================================================")
    print("Backbone:        EfficientNet-B0 (Standard)")
    print("Using device:   ", device)
    print("Data directory: ", args.data_dir)
    print("Images directory:", args.img_dir)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Base LR: {args.lr}, Seed: {args.seed}")

    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "val.csv")

    pos_weights, sev_weights = calculate_class_weights(train_csv, device)

    train_ds = ODIRDataset(train_csv, args.img_dir, get_train_transforms())
    val_ds = ODIRDataset(val_csv, args.img_dir, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)

    # Differential learning rates: stable fine-tuning of backbone with responsive heads
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.25},
        {"params": model.disease_head.parameters(), "lr": args.lr},
        {"params": model.severity_head.parameters(), "lr": args.lr * 1.20},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    disease_loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, pos_weight=pos_weights)

    best_score = -1.0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_sev_acc = 0.0

    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, disease_loss_fn, sev_weights, device)
        val_loss, val_auc, val_acc, val_sev_acc = validate(model, val_loader, disease_loss_fn, sev_weights, device)
        scheduler.step()

        # Selection criterion strictly on validation split: 55% Acc, 30% AUC, 15% Severity
        composite_score = 0.55 * val_acc + 0.30 * val_auc + 0.15 * val_sev_acc

        print(f"Epoch {epoch + 1:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | Sev Acc: {val_sev_acc*100:5.2f}%")

        if composite_score > best_score:
            best_score = composite_score
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_sev_acc = val_sev_acc
            checkpoint_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best checkpoint (Val AUC: {best_val_auc*100:.2f}%, Val Acc: {best_val_acc*100:.2f}%, Sev Acc: {best_sev_acc*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete! Best Val AUC: {best_val_auc*100:.2f}%, Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()