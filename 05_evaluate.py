"""
05_evaluate.py
--------------
Comprehensive Evaluation & Calibration Suite for RetinaSense EfficientNet-B0:
  1. Validation-based Disease Threshold Calibration (optimizes F1 & balanced accuracy on val split)
  2. Multi-label Disease Evaluation on Test Set (Accuracy >90%, AUC, Precision, Recall, Specificity, F1, Kappa)
  3. Dedicated Normal Retinal Image Screening Verification (raw probabilities across all 6 diseases)
  4. Balanced DR Severity Evaluation (Confusion matrix, Balanced Accuracy, Macro F1, Per-class metrics)
  5. Exports models/metrics.json and models/thresholds.json
"""

import argparse
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
)

DISEASE_COLS = ["D", "G", "C", "A", "H", "M"]

DISEASE_NAMES = {
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertensive Retinopathy",
    "M": "Myopia",
}

SEVERITY_NAMES = [
    "Unspecified DR (0)",
    "Mild NPDR (1)",
    "Moderate NPDR (2)",
    "Severe NPDR (3)",
    "Proliferative DR (4)",
]


def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")

ODIRDataset = ds_mod.ODIRDataset
get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel


def build_model():
    return RetinaSenseModel(pretrained=False)


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels = []
    all_probs = []
    all_sev_true = []
    all_sev_pred = []
    all_filenames = []

    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        disease_logits, severity_logits = model(images)

        probs = torch.sigmoid(disease_logits).cpu().numpy()
        labels = batch["labels"].numpy()
        sev_true = batch["dr_severity"].numpy()
        sev_pred = severity_logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)
        all_sev_true.extend(sev_true.tolist())
        all_sev_pred.extend(sev_pred.tolist())
        all_filenames.extend(batch["filename"])

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.array(all_sev_true),
        np.array(all_sev_pred),
        all_filenames,
    )


def calibrate_thresholds_on_val(val_true, val_prob):
    """Calibrate optimal decision threshold for each disease on the VALIDATION split."""
    calibrated_thresholds = {}
    print("\n=======================================================")
    print("      VALIDATION THRESHOLD CALIBRATION (OPTIMAL)")
    print("=======================================================")

    for i, name in enumerate(DISEASE_COLS):
        best_score = -1.0
        best_th = 0.50
        y_t = val_true[:, i]
        y_p = val_prob[:, i]

        if y_t.sum() == 0:
            calibrated_thresholds[name] = 0.50
            continue

        for th in np.arange(0.15, 0.85, 0.02):
            y_pred_th = (y_p >= th).astype(int)
            f1 = f1_score(y_t, y_pred_th, zero_division=0)
            acc = accuracy_score(y_t, y_pred_th)
            # Composite objective balancing F1 and high accuracy
            score = 0.6 * f1 + 0.4 * acc
            if score > best_score:
                best_score = score
                best_th = float(th)

        calibrated_thresholds[name] = round(best_th, 2)
        print(f"  {DISEASE_NAMES[name]:32s} ({name}): Optimal Threshold = {best_th:.2f} (Val Composite Score = {best_score:.4f})")

    return calibrated_thresholds


def evaluate_diseases(y_true, y_prob, thresholds):
    """Computes Accuracy, AUC, Precision, Recall, Specificity, F1 per disease using calibrated thresholds."""
    results = {}
    accuracies, aucs, precisions, recalls, specificities, f1s, kappas = [], [], [], [], [], [], []

    print("\n=======================================================")
    print("         PER-DISEASE TEST SET PERFORMANCE")
    print("=======================================================")
    print(f"{'Disease':30s} | {'Threshold':9s} | {'Accuracy':8s} | {'AUC':6s} | {'Prec':6s} | {'Recall':6s} | {'Spec':6s} | {'F1':6s} | {'Kappa':6s}")
    print("-" * 105)

    for i, name in enumerate(DISEASE_COLS):
        d_name = DISEASE_NAMES[name]
        th = thresholds.get(name, 0.50)
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        y_pred = (y_p >= th).astype(int)

        acc = accuracy_score(y_t, y_pred)
        try:
            auc = roc_auc_score(y_t, y_p)
        except ValueError:
            auc = float("nan")

        prec = precision_score(y_t, y_pred, zero_division=0)
        rec = recall_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        try:
            kappa = cohen_kappa_score(y_t, y_pred)
        except ValueError:
            kappa = float("nan")

        # Specificity
        tn = ((y_t == 0) & (y_pred == 0)).sum()
        fp = ((y_t == 0) & (y_pred == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        accuracies.append(acc)
        aucs.append(auc)
        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)
        f1s.append(f1)
        kappas.append(kappa)

        results[name] = {
            "disease_name": d_name,
            "threshold": th,
            "accuracy": float(acc),
            "auc": float(auc) if not np.isnan(auc) else None,
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec) if not np.isnan(spec) else None,
            "f1": float(f1),
            "kappa": float(kappa) if not np.isnan(kappa) else None,
        }

        print(f"{d_name:30s} | {th:9.2f} | {acc*100:7.2f}% | {auc:6.4f} | {prec:6.4f} | {rec:6.4f} | {spec:6.4f} | {f1:6.4f} | {kappa:6.4f}")

    macro_acc = float(np.nanmean(accuracies))
    macro_auc = float(np.nanmean(aucs))
    macro_f1 = float(np.nanmean(f1s))
    macro_prec = float(np.nanmean(precisions))
    macro_rec = float(np.nanmean(recalls))
    macro_kappa = float(np.nanmean(kappas))
    odir_score = float((macro_auc + macro_f1 + macro_kappa) / 3)

    print("\n-------------------------------------------------------")
    print(f"Macro Overall Accuracy: {macro_acc*100:.2f}%")
    print(f"Macro AUC:              {macro_auc:.4f} ({macro_auc*100:.2f}%)")
    print(f"Macro Precision:        {macro_prec:.4f}")
    print(f"Macro Recall:           {macro_rec:.4f}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print(f"Macro Cohen's Kappa:    {macro_kappa:.4f}")
    print(f"ODIR Overall Score:     {odir_score:.4f}")
    print("-------------------------------------------------------")

    return results, {
        "macro_accuracy": macro_acc,
        "macro_auc": macro_auc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_kappa": macro_kappa,
        "odir_score": odir_score,
    }


def evaluate_normal_images(y_true, y_prob, filenames, thresholds):
    """Tests known normal fundus images and reports raw predicted probabilities."""
    print("\n=======================================================")
    print("        NORMAL RETINAL IMAGE SCREENING EVALUATION")
    print("=======================================================")

    normal_mask = (y_true.sum(axis=1) == 0)
    n_normals = int(normal_mask.sum())
    print(f"Total verified normal test images: {n_normals}")

    if n_normals == 0:
        print("No normal images found in test set.")
        return {}

    normal_probs = y_prob[normal_mask]
    normal_filenames = [filenames[idx] for idx, is_n in enumerate(normal_mask) if is_n]

    print("\nMean Predicted Disease Probabilities on Known Normal Images (Lower is Better):")
    mean_normal_probs = {}
    for i, name in enumerate(DISEASE_COLS):
        m_prob = float(normal_probs[:, i].mean())
        max_prob = float(normal_probs[:, i].max())
        mean_normal_probs[name] = m_prob
        print(f"  {DISEASE_NAMES[name]:32s} ({name}): Mean = {m_prob:.4f}, Max = {max_prob:.4f}")

    all_below_th = np.all(
        [normal_probs[:, i] < thresholds.get(name, 0.50) for i, name in enumerate(DISEASE_COLS)],
        axis=0
    )
    correct_normal_count = int(all_below_th.sum())
    normal_screening_accuracy = correct_normal_count / n_normals * 100

    print(f"\nNormal Screening Specificity: {correct_normal_count}/{n_normals} ({normal_screening_accuracy:.2f}%) "
          f"images correctly yielded zero disease alarms.")

    print("\nSample 5 Known Normal Test Images and Raw Probabilities:")
    for idx in range(min(5, n_normals)):
        fn = normal_filenames[idx]
        probs_str = " | ".join([f"{name}:{normal_probs[idx, i]:.3f}" for i, name in enumerate(DISEASE_COLS)])
        is_clean = "✔ Clean" if all_below_th[idx] else "⚠ Borderline"
        print(f"  [{is_clean}] {fn:15s} -> {probs_str}")

    return {
        "n_normal_test_samples": n_normals,
        "correct_normal_classifications": correct_normal_count,
        "normal_screening_specificity": float(normal_screening_accuracy),
        "mean_disease_probabilities_on_normal": mean_normal_probs,
    }


def evaluate_dr_severity(sev_true, sev_pred):
    """Balanced evaluation of DR severity grading on DR-positive samples."""
    print("\n=======================================================")
    print("             DR SEVERITY EVALUATION")
    print("=======================================================")

    mask = (sev_true >= 0)
    y_true = sev_true[mask]
    y_pred = sev_pred[mask]
    n_dr = len(y_true)

    print(f"Total DR-positive test samples evaluated: {n_dr}")
    if n_dr == 0:
        print("No DR-positive test samples found.")
        return {}

    print("\nDR Severity Class Distribution in Test Set:")
    for c in range(5):
        cnt = int((y_true == c).sum())
        pct = cnt / n_dr * 100 if n_dr > 0 else 0
        print(f"  Grade {c} [{SEVERITY_NAMES[c]:22s}]: {cnt:3d} ({pct:5.1f}%)")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print("\nDR Severity Confusion Matrix (Rows=True, Cols=Predicted):")
    print("        [G0] [G1] [G2] [G3] [G4]")
    for r_idx, row in enumerate(cm):
        print(f"  [G{r_idx}]   {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d}")

    pred_counts = np.bincount(y_pred, minlength=5)
    print("\nModel Prediction Counts across Grades:")
    for c in range(5):
        print(f"  Predicted Grade {c}: {pred_counts[c]} times")

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raw_acc = float((y_true == y_pred).mean())

    print(f"\nRaw Severity Accuracy:      {raw_acc:.4f} ({raw_acc*100:.2f}%)")
    print(f"Balanced Severity Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"Macro Severity F1:           {macro_f1:.4f}")

    return {
        "n_dr_samples": n_dr,
        "raw_accuracy": raw_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {c: int((y_true == c).sum()) for c in range(5)},
        "predicted_distribution": {c: int(pred_counts[c]) for c in range(5)},
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate RetinaSense EfficientNet-B0")
    ap.add_argument("--checkpoint", default="./models/best_model.pt", help="Path to model checkpoint")
    ap.add_argument("--data_dir", default="./data", help="Directory containing train.csv, val.csv, test.csv")
    ap.add_argument("--img_dir", default="dataset/ODIR-5K/Training Images", help="Directory containing images")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("      RetinaSense Comprehensive Model Evaluation")
    print("=======================================================")
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Data directory:  {args.data_dir}")
    print(f"Image directory: {args.img_dir}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val_csv = os.path.join(args.data_dir, "val.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")

    # 1. Validation Set Inference & Threshold Calibration
    val_ds = ODIRDataset(val_csv, args.img_dir, get_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_true, val_prob, _, _, _ = run_inference(model, val_loader, device)

    calibrated_thresholds = calibrate_thresholds_on_val(val_true, val_prob)

    thresholds_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(calibrated_thresholds, f, indent=2)
    print(f"\n[SAVED] Calibrated thresholds saved to: {thresholds_path}")

    # 2. Test Set Inference
    test_ds = ODIRDataset(test_csv, args.img_dir, get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_true, test_prob, sev_true, sev_pred, test_filenames = run_inference(model, test_loader, device)

    # 3. Evaluate Disease Classification
    per_disease_results, macro_results = evaluate_diseases(test_true, test_prob, calibrated_thresholds)

    # 4. Evaluate Normal Image Screening
    normal_results = evaluate_normal_images(test_true, test_prob, test_filenames, calibrated_thresholds)

    # 5. Evaluate DR Severity
    severity_results = evaluate_dr_severity(sev_true, sev_pred)

    # 6. Export models/metrics.json
    metrics_summary = {
        "model": "EfficientNet-B0",
        "num_diseases": 6,
        "disease_targets": DISEASE_COLS,
        "n_test_samples": int(len(test_true)),
        "macro_metrics": macro_results,
        "per_disease_metrics": per_disease_results,
        "normal_screening_metrics": normal_results,
        "dr_severity_metrics": severity_results,
        "calibrated_thresholds": calibrated_thresholds,
    }

    metrics_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[SAVED] Comprehensive metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    main()