# RetinaSense — Multi-Disease Retinal Screening System

Predicts multiple retinal diseases from fundus images, grades severity, and
compares two images of the same eye over time to track disease progression.

Dataset: **ODIR-5K** (Ocular Disease Intelligent Recognition), Kaggle:
`andrewmvd/ocular-disease-recognition-odir5k`

---

## 1. Project Pipeline (step-by-step)

```
Step 1  Environment setup (Kaggle/Colab, GPU, libraries)
Step 2  Download & understand ODIR-5K (images + full_df.csv / data.xlsx)
Step 3  Data cleaning & EDA (class distribution, missing files, image sizes)
Step 4  Preprocess: extract multi-labels (N, D, G, C, A, H, M), excluding
        the unspecific/noisy "Other" (O) category + extract DR severity
        from diagnostic keyword text
Step 5  Train/val/test split (patient-wise, NOT image-wise, to avoid leakage
        — left+right eye of same patient must stay in the same split)
Step 6  Build tf.data / torch Dataset with augmentation (CLAHE, rotation,
        flips, brightness/contrast — fundus images tolerate horizontal
        flip but NOT vertical flip well since orientation matters clinically)
Step 7  Model A: multi-label disease classifier
        (EfficientNet-B0 backbone, pretrained ImageNet, 7-way sigmoid head:
        Normal, DR, Glaucoma, Cataract, AMD, Hypertension, Myopia)
Step 8  Model B: DR severity classifier
        (shares backbone features, 5-way softmax head: No_DR/Mild/
        Moderate/Severe/Proliferative) — trained only on DR-positive samples
Step 9  Train with class-weighted focal loss (ODIR is heavily imbalanced)
Step 10 Evaluate with ODIR's official metric: mean(AUC, F1, Kappa),
        plus per-class AUC and confusion matrices
Step 11 Progression tracking module: image registration (ORB + homography)
        + SSIM difference heatmap between two images of the same eye
        taken at different times
Step 12 Wrap everything into a simple inference script / Streamlit demo
Step 13 (Optional) Deploy as a web app
```

## 2. File Guide

| File | Purpose |
|---|---|
| `01_data_preprocessing.py` | Loads `full_df.csv` / `data.xlsx`, excludes "Other" (`O=1`) images, builds 7-disease multi-labels (`N,D,G,C,A,H,M`), extracts DR severity from keywords, does patient-wise split, saves clean CSVs |
| `02_dataset.py` | PyTorch `Dataset`/`DataLoader` with augmentation pipeline for 7 disease classes |
| `03_model.py` | Model A: EfficientNet-B0 (pretrained, transfer learning) + 7-way disease & 5-way severity heads |
| `03b_custom_cnn.py` | Model B: CNN built from scratch (5 conv blocks, no pretrained weights) + same heads — use this to show/explain your own CNN architecture |
| `04_train.py` | Training loop, loss functions, checkpointing |
| `05_evaluate.py` | Metrics: AUC, F1, Kappa (ODIR official score), confusion matrix over 7 disease classes |
| `06_image_comparison.py` | Registration + SSIM diff for progression tracking |
| `07_inference_demo.py` | End-to-end inference on a new image / image pair (7 disease categories) |
| `requirements.txt` | Dependencies |

## 3. How to run (Kaggle Notebook is easiest — dataset is already there)

```bash
pip install -r requirements.txt

python 01_data_preprocessing.py \
    --csv_path /kaggle/input/ocular-disease-recognition-odir5k/full_df.csv \
    --img_dir  /kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images \
    --out_dir  ./data

python 04_train.py --data_dir ./data --img_dir <img_dir> --epochs 25 --batch_size 32 \
    --model_type transfer   # or: --model_type custom

python 05_evaluate.py --checkpoint ./models/best_model.pt --data_dir ./data \
    --img_dir <img_dir> --model_type transfer   # match whichever you trained

python 07_inference_demo.py --image path/to/fundus.jpg --checkpoint ./models/best_model.pt \
    --model_type transfer
```

`--model_type transfer` uses the pretrained EfficientNet-B0 backbone (`03_model.py`) —
best accuracy with limited data, standard for a working system.
`--model_type custom` uses a CNN built entirely from scratch (`03b_custom_cnn.py`,
5 conv blocks with BatchNorm/ReLU/MaxPool/Dropout) — use this if your project
requires you to design and explain your own architecture. A good report
trains both and compares them (transfer learning almost always wins on a
dataset this size — that comparison itself is a nice result to present).

## 4. Important honesty notes for your report / viva

- ODIR-5K originally provides **multi-label disease** ground truth across 8 categories (N,D,G,C,A,H,M,O). In this project, the ambiguous **"Other" (O)** category and its associated images are filtered out during preprocessing to focus the network on the **7 specific, clinically actionable retinal conditions** (Normal, DR, Glaucoma, Cataract, AMD, Hypertension, Myopia).
- ODIR-5K does **not** provide a clean severity column. Severity here is
  derived by **rule-based keyword parsing** of the free-text diagnostic
  keywords field, and only reliably works for **Diabetic Retinopathy**,
  because that's the disease whose grading vocabulary (mild/moderate/
  severe NPDR, PDR) is consistently present in the text. Say this
  explicitly in your report — don't claim severity grading for all
  diseases, a reviewer will ask.
- ODIR-5K has **no longitudinal (same patient, multiple time points) data**,
  so "progression tracking" cannot be trained/validated on ODIR itself.
  The registration+diff module is a general image-processing tool you
  demo on two images (e.g. a baseline and a synthetically-altered or
  externally sourced follow-up image) rather than something with an
  ODIR-based accuracy number. Frame it as a *decision-support visualization
  tool*, not a trained/validated model.