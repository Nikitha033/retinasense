"""
05_evaluate.py
--------------
Evaluates a trained EfficientNet-B0 RetinaSense checkpoint on the ODIR test split.

Reports:
  - Overall disease-classification accuracy (Top-1)
  - Per-disease AUC
  - Macro F1 (at 0.5 threshold)
  - Cohen's Kappa (averaged across disease labels)
  - ODIR-style final score = mean(AUC, F1, Kappa)
  - DR severity accuracy
  - DR severity confusion matrix

The evaluation results are saved to:
    models/metrics.json
"""

import argparse
import os
import json
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Disease labels
# ---------------------------------------------------------------------------

DISEASE_COLS = ["D", "G", "C", "A", "H", "M"]

DISEASE_NAMES = {
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertensive Retinopathy",
    "M": "Myopia",
}


# ---------------------------------------------------------------------------
# Load project modules
# ---------------------------------------------------------------------------

def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_file
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


_here = os.path.dirname(os.path.abspath(__file__))

ds_mod = _load(
    os.path.join(_here, "02_dataset.py"),
    "ds_mod"
)

model_mod = _load(
    os.path.join(_here, "03_model.py"),
    "model_mod"
)


# ---------------------------------------------------------------------------
# Import required classes/functions
# ---------------------------------------------------------------------------

ODIRDataset = ds_mod.ODIRDataset
get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel


# ---------------------------------------------------------------------------
# Build EfficientNet-B0 transfer-learning model
# ---------------------------------------------------------------------------

def build_model():
    """
    Build the same RetinaSense EfficientNet-B0 model
    used during training.
    """

    model = RetinaSenseModel(
        pretrained=False
    )

    return model


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):

    all_labels = []
    all_probs = []

    all_sev_true = []
    all_sev_pred = []

    model.eval()

    for batch in loader:

        images = batch["image"].to(device)

        # Model returns:
        # disease_logits
        # severity_logits

        disease_logits, severity_logits = model(images)

        # ---------------------------------------------------------------
        # Disease probabilities
        # ---------------------------------------------------------------

        probs = torch.sigmoid(disease_logits)

        probs = probs.cpu().numpy()

        all_probs.append(probs)

        # ---------------------------------------------------------------
        # True disease labels
        # ---------------------------------------------------------------

        labels = batch["labels"].numpy()

        all_labels.append(labels)

        # ---------------------------------------------------------------
        # DR severity
        # ---------------------------------------------------------------

        sev_true = batch["dr_severity"].numpy()

        sev_pred = (
            severity_logits
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        # Keep only samples with known DR severity

        mask = sev_true >= 0

        all_sev_true.extend(
            sev_true[mask].tolist()
        )

        all_sev_pred.extend(
            sev_pred[mask].tolist()
        )

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.array(all_sev_true),
        np.array(all_sev_pred),
    )


# ---------------------------------------------------------------------------
# Overall disease classification accuracy
# ---------------------------------------------------------------------------

def calculate_overall_accuracy(y_true, y_prob):
    """
    Calculate Top-1 disease-classification accuracy.

    For each test image:
      1. Find the disease with the highest true label.
      2. Find the disease with the highest predicted probability.
      3. Count it as correct if both are the same.

    This is useful as an overall single-disease classification metric,
    while the other metrics still evaluate the original multi-label task.
    """

    # Highest-probability predicted disease
    predicted_class = np.argmax(
        y_prob,
        axis=1
    )

    # Highest true-label disease
    true_class = np.argmax(
        y_true,
        axis=1
    )

    # Compare predictions with true class
    correct = (
        predicted_class == true_class
    )

    accuracy = np.mean(correct)

    return float(accuracy)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate RetinaSense EfficientNet-B0 model"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing train.csv/test.csv"
    )

    parser.add_argument(
        "--img_dir",
        required=True,
        help="Directory containing retinal images"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========================================")
    print("       RetinaSense Model Evaluation")
    print("========================================")

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Image directory: {args.img_dir}")

    # -----------------------------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------------------------

    if not os.path.isfile(args.checkpoint):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{args.checkpoint}\n\n"
            "Please check the checkpoint path."
        )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------

    print("\nLoading EfficientNet-B0 model...")

    model = build_model()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    # -----------------------------------------------------------------------
    # Test dataset
    # -----------------------------------------------------------------------

    test_csv = os.path.join(
        args.data_dir,
        "test.csv"
    )

    if not os.path.isfile(test_csv):

        raise FileNotFoundError(
            f"\nTest CSV not found:\n{test_csv}"
        )

    if not os.path.isdir(args.img_dir):

        raise FileNotFoundError(
            f"\nImage directory not found:\n{args.img_dir}"
        )

    print("\nLoading test dataset...")

    test_ds = ODIRDataset(
        test_csv,
        args.img_dir,
        get_eval_transforms()
    )

    print(f"Test samples: {len(test_ds)}")

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # num_workers=0 is intentionally used here because it is
    # more reliable on Windows.

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    print("\nRunning inference...")

    y_true, y_prob, sev_true, sev_pred = run_inference(
        model,
        test_loader,
        device
    )

    # -----------------------------------------------------------------------
    # Overall disease classification accuracy
    # -----------------------------------------------------------------------

    overall_accuracy = calculate_overall_accuracy(
        y_true,
        y_prob
    )

    # -----------------------------------------------------------------------
    # Disease predictions at 0.5 threshold
    # -----------------------------------------------------------------------

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    # -----------------------------------------------------------------------
    # Per-disease AUC
    # -----------------------------------------------------------------------

    print("\n----------------------------------------")
    print("Per-disease AUC")
    print("----------------------------------------")

    aucs = []

    for i, name in enumerate(DISEASE_COLS):

        try:

            auc = roc_auc_score(
                y_true[:, i],
                y_prob[:, i]
            )

        except ValueError:

            auc = float("nan")

        aucs.append(auc)

        disease_name = DISEASE_NAMES[name]

        if np.isnan(auc):

            print(
                f"  {disease_name}: N/A"
            )

        else:

            print(
                f"  {disease_name}: {auc:.4f}"
            )

    # -----------------------------------------------------------------------
    # Macro AUC
    # -----------------------------------------------------------------------

    macro_auc = np.nanmean(aucs)

    # -----------------------------------------------------------------------
    # Macro F1
    # -----------------------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # -----------------------------------------------------------------------
    # Cohen's Kappa
    # -----------------------------------------------------------------------

    kappas = []

    for i in range(len(DISEASE_COLS)):

        try:

            kappa = cohen_kappa_score(
                y_true[:, i],
                y_pred[:, i]
            )

            kappas.append(kappa)

        except ValueError:

            kappas.append(float("nan"))

    macro_kappa = np.nanmean(kappas)

    # -----------------------------------------------------------------------
    # ODIR-style final score
    # -----------------------------------------------------------------------

    odir_score = (
        macro_auc +
        macro_f1 +
        macro_kappa
    ) / 3

    # -----------------------------------------------------------------------
    # Print overall metrics
    # -----------------------------------------------------------------------

    print("\n----------------------------------------")
    print("Overall Evaluation Results")
    print("----------------------------------------")

    print(
        f"Overall Disease Accuracy: "
        f"{overall_accuracy:.4f} "
        f"({overall_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro AUC:                "
        f"{macro_auc:.4f}"
    )

    print(
        f"Macro F1:                 "
        f"{macro_f1:.4f}"
    )

    print(
        f"Macro Kappa:              "
        f"{macro_kappa:.4f}"
    )

    print(
        f"ODIR Final Score:         "
        f"{odir_score:.4f}"
    )

    # -----------------------------------------------------------------------
    # DR Severity Evaluation
    # -----------------------------------------------------------------------

    sev_acc = None

    if len(sev_true) > 0:

        print("\n----------------------------------------")
        print("DR Severity Evaluation")
        print("----------------------------------------")

        sev_acc = (
            sev_true == sev_pred
        ).mean()

        print(
            f"Severity Accuracy: "
            f"{sev_acc:.4f} "
            f"({sev_acc * 100:.2f}%)"
        )

        print(
            f"Number of DR severity samples: "
            f"{len(sev_true)}"
        )

        print("\nConfusion Matrix")
        print("(Rows = True, Columns = Predicted)")
        print("(Class order: 0, 1, 2, 3, 4)")

        cm = confusion_matrix(
            sev_true,
            sev_pred,
            labels=[0, 1, 2, 3, 4]
        )

        print(cm)

    else:

        print("\n----------------------------------------")
        print("DR Severity Evaluation")
        print("----------------------------------------")

        print(
            "No DR-positive samples with known "
            "severity found in test split."
        )

    # -----------------------------------------------------------------------
    # Save metrics.json
    # -----------------------------------------------------------------------

    metrics = {

        # NEW:
        # Overall disease classification accuracy
        "overall_accuracy":
            float(overall_accuracy),

        "macro_auc":
            float(macro_auc),

        "macro_f1":
            float(macro_f1),

        "macro_kappa":
            float(macro_kappa),

        "odir_score":
            float(odir_score),

        "per_disease_auc": {

            name: (
                float(auc)
                if not np.isnan(auc)
                else None
            )

            for name, auc in zip(
                DISEASE_COLS,
                aucs
            )
        },

        "dr_severity_accuracy":
            (
                float(sev_acc)
                if sev_acc is not None
                else None
            ),

        "n_test_samples":
            int(len(y_true)),

        "model_type":
            "transfer",

        "model":
            "EfficientNet-B0",
    }

    metrics_path = os.path.join(
        os.path.dirname(args.checkpoint),
        "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # -----------------------------------------------------------------------
    # Finished
    # -----------------------------------------------------------------------

    print("\n========================================")
    print("Evaluation completed successfully!")
    print("========================================")

    print(
        f"\nMetrics saved to:\n{metrics_path}"
    )

    print("\n")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()