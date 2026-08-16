"""
04_train.py
-----------
Trains RetinaSenseModel jointly on:
  - disease presence (multi-label, focal loss to handle imbalance)
  - DR severity (5-way CE, masked so it only contributes for DR-positive
    samples with a known severity label)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from importlib import import_module

dataset_mod = import_module("02_dataset".replace("-", "_")) if False else None
# direct imports (files are numeric-prefixed, so import via exec-friendly names)
import importlib.util

def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")
#custom_cnn_mod = _load(os.path.join(_here, "03b_custom_cnn.py"), "custom_cnn_mod")

ODIRDataset = ds_mod.ODIRDataset
get_train_transforms = ds_mod.get_train_transforms
get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel
#CustomRetinaCNN = custom_cnn_mod.CustomRetinaCNN


def build_model(model_type: str):
    """model_type: 'transfer' -> pretrained EfficientNet-B0 backbone (03_model.py)
                    'custom'   -> CNN built from scratch (03b_custom_cnn.py)"""
    if model_type == "transfer":
        return RetinaSenseModel(pretrained=True, freeze_backbone_layers=True)
    #elif model_type == "custom":
        return CustomRetinaCNN()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class FocalLossMultiLabel(nn.Module):
    """Focal loss for multi-label classification — down-weights easy
    (already well-classified) examples so the model keeps learning from
    the rarer disease classes (e.g. G, A, H are far rarer than N, D)."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def masked_severity_loss(severity_logits, severity_targets):
    mask = severity_targets >= 0

    print("Valid severity samples:", mask.sum().item())

    if mask.sum() == 0:
        return torch.tensor(0.0, device=severity_logits.device)

    return F.cross_entropy(
        severity_logits[mask],
        severity_targets[mask]
    )


def train_one_epoch(model, loader, optimizer, disease_loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["dr_severity"].to(device)

        optimizer.zero_grad()
        disease_logits, severity_logits = model(images)

        loss_disease = disease_loss_fn(disease_logits, labels)
        loss_severity = masked_severity_loss(severity_logits, severity)
        loss = loss_disease + 0.5 * loss_severity  # severity is auxiliary, weighted lower

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, disease_loss_fn, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="val"):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["dr_severity"].to(device)

        disease_logits, severity_logits = model(images)
        loss_disease = disease_loss_fn(disease_logits, labels)
        loss_severity = masked_severity_loss(severity_logits, severity)
        loss = loss_disease + 0.5 * loss_severity
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--model_type", choices=["transfer", "custom"], default="transfer",
                     help="'transfer' = pretrained EfficientNet-B0, 'custom' = CNN built from scratch")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = ODIRDataset(os.path.join(args.data_dir, "train.csv"), args.img_dir, get_train_transforms())
    val_ds = ODIRDataset(os.path.join(args.data_dir, "val.csv"), args.img_dir, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model_type).to(device)
    print(f"Using model_type={args.model_type}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    disease_loss_fn = FocalLossMultiLabel()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, disease_loss_fn, device)
        val_loss = validate(model, val_loader, disease_loss_fn, device)
        scheduler.step(val_loss)
        print(f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print("  -> saved new best checkpoint")

    print("\nTraining complete. Best val_loss:", best_val_loss)


if __name__ == "__main__":
    main()