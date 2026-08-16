
"""
05_evaluate.py
--------------
Evaluates a trained checkpoint on the test split.
Reports:
  - per-disease AUC
  - macro F1 (at 0.5 threshold)
  - Cohen's Kappa (on argmax-style binarized predictions, averaged)
  - the official ODIR competition score = mean(AUC, F1, Kappa)
  - DR severity accuracy / confusion matrix (on DR-positive test samples)
"""

import argparse
import os
import json
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix

DISEASE_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")
custom_cnn_mod = _load(os.path.join(_here, "03b_custom_cnn.py"), "custom_cnn_mod")

ODIRDataset = ds_mod.ODIRDataset
get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel
CustomRetinaCNN = custom_cnn_mod.CustomRetinaCNN


def build_model(model_type: str):
    if model_type == "transfer":
        return RetinaSenseModel(pretrained=False)
    elif model_type == "custom":
        return CustomRetinaCNN()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_probs, all_sev_true, all_sev_pred = [], [], [], []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        disease_logits, severity_logits = model(images)
        probs = torch.sigmoid(disease_logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())

        sev_true = batch["dr_severity"].numpy()
        sev_pred = severity_logits.argmax(dim=1).cpu().numpy()
        mask = sev_true >= 0
        all_sev_true.extend(sev_true[mask].tolist())
        all_sev_pred.extend(sev_pred[mask].tolist())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.array(all_sev_true),
        np.array(all_sev_pred),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model_type", choices=["transfer", "custom"], default="transfer")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_ds = ODIRDataset(os.path.join(args.data_dir, "test.csv"), args.img_dir, get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    y_true, y_prob, sev_true, sev_pred = run_inference(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n--- Per-disease AUC ---")
    aucs = []
    for i, name in enumerate(DISEASE_COLS):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")  # happens if a class has no positive samples in test split
        aucs.append(auc)
        print(f"  {name}: {auc:.4f}")

    macro_auc = np.nanmean(aucs)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Kappa is defined per-label for multi-label; average across labels
    kappas = [cohen_kappa_score(y_true[:, i], y_pred[:, i]) for i in range(len(DISEASE_COLS))]
    macro_kappa = np.nanmean(kappas)

    odir_score = (macro_auc + macro_f1 + macro_kappa) / 3
    print(f"\nMacro AUC:   {macro_auc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Macro Kappa: {macro_kappa:.4f}")
    print(f"ODIR final score (mean of the above): {odir_score:.4f}")

    if len(sev_true):
        print("\n--- DR severity (DR-positive test samples only) ---")
        sev_acc = (sev_true == sev_pred).mean()
        print(f"Severity accuracy: {sev_acc:.4f}  (n={len(sev_true)})")
        print("Confusion matrix (rows=true, cols=pred, order 0..4):")
        print(confusion_matrix(sev_true, sev_pred, labels=[0, 1, 2, 3, 4]))
    else:
        sev_acc = None
        print("\nNo DR-positive samples with known severity in test split.")

    # Save a machine-readable summary that app.py reads for the
    # "Performance Metrics" cards, so the dashboard always reflects your
    # actual test-set results rather than a hardcoded/placeholder number.
    metrics = {
        "macro_auc": float(macro_auc),
        "macro_f1": float(macro_f1),
        "macro_kappa": float(macro_kappa),
        "odir_score": float(odir_score),
        "per_disease_auc": {name: (float(a) if a == a else None) for name, a in zip(DISEASE_COLS, aucs)},
        "dr_severity_accuracy": float(sev_acc) if sev_acc is not None else None,
        "n_test_samples": int(len(y_true)),
        "model_type": args.model_type,
    }
    metrics_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()