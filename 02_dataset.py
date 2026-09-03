"""
02_dataset.py
-------------
PyTorch Dataset + augmentation pipeline for ODIR fundus images.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DISEASE_COLS = ["N", "D", "G", "C", "A", "H", "M"]
IMG_SIZE = 224


def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),          # left/right eye flip is valid
        A.Rotate(limit=15, p=0.5),        # small rotation only — fundus anatomy is orientation-sensitive
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),   # boosts vessel/lesion contrast, common for fundus images
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_eval_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class ODIRDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform or get_eval_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Could not load image: '{img_path}'. "
                f"Please ensure --img_dir points to the correct image folder "
                f"(e.g., 'dataset/ODIR-5K/Training Images')."
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img_t = augmented["image"]

        labels = torch.tensor(row[DISEASE_COLS].values.astype("float32"))
        dr_severity = int(row["dr_severity"])  # -1 if not applicable

        return {
            "image": img_t,
            "labels": labels,
            "dr_severity": dr_severity,
            "filename": row["filename"],
        }