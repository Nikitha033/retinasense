"""
07_inference_demo.py
---------------------
End-to-end demo:
  - Single image  -> predicted diseases (with probabilities) + DR severity
                     if DR is predicted positive
  - Two images    -> also runs progression comparison (06_image_comparison)

Usage:
  python 07_inference_demo.py --image path.jpg --checkpoint models/best_model.pt
  python 07_inference_demo.py --image baseline.jpg --followup followup.jpg \
      --checkpoint models/best_model.pt
"""

import argparse
import os
import importlib.util
import cv2
import torch

DISEASE_NAMES = {
    "N": "Normal", "D": "Diabetic Retinopathy", "G": "Glaucoma",
    "C": "Cataract", "A": "Age-related Macular Degeneration",
    "H": "Hypertensive Retinopathy", "M": "Myopia",
}
DISEASE_COLS = ["N", "D", "G", "C", "A", "H", "M"]
SEVERITY_NAMES = ["No/Unspecified", "Mild", "Moderate", "Severe", "Proliferative"]


def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")
#custom_cnn_mod = _load(os.path.join(_here, "03b_custom_cnn.py"), "custom_cnn_mod")
cmp_mod = _load(os.path.join(_here, "06_image_comparison.py"), "cmp_mod")

get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel
#CustomRetinaCNN = custom_cnn_mod.CustomRetinaCNN


def build_model(model_type):
    if model_type == "transfer":
        return RetinaSenseModel(pretrained=False)
    else:
        raise ValueError("Only transfer model is supported.")


def predict_single(image_path, model, device, threshold=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_eval_transforms()
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        disease_logits, severity_logits = model(tensor)
        probs = torch.sigmoid(disease_logits).cpu().numpy()[0]
        severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]

    results = {DISEASE_NAMES[c]: float(probs[i]) for i, c in enumerate(DISEASE_COLS)}
    predicted = {k: v for k, v in results.items() if v >= threshold}

    severity_result = None
    if probs[DISEASE_COLS.index("D")] >= threshold:
        sev_idx = int(severity_probs.argmax())
        severity_result = {
            "grade": SEVERITY_NAMES[sev_idx],
            "confidence": float(severity_probs[sev_idx]),
        }

    return results, predicted, severity_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="fundus image to analyze")
    ap.add_argument("--followup", default=None, help="optional second image, same eye, later date")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model_type", choices=["transfer", "custom"], default="transfer")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_probs, predicted, severity = predict_single(args.image, model, device, args.threshold)

    print("\n=== Disease probabilities ===")
    for name, p in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {name:35s}: {p:.3f}")

    print("\n=== Predicted positive (>= threshold) ===")
    if predicted:
        for name, p in predicted.items():
            print(f"  {name}: {p:.3f}")
    else:
        print("  None above threshold")

    if severity:
        print(f"\n=== DR severity: {severity['grade']} (confidence {severity['confidence']:.3f}) ===")

    if args.followup:
        print("\n=== Progression comparison ===")
        cmp_mod.compare_progression(args.image, args.followup, out_prefix="progression")


if __name__ == "__main__":
    main()