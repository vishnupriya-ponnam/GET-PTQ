"""Test/calibration data loading that matches GET's Image_Dataset (test transform).

GET's test transform is: divide by 255, resize to 224 x 224 (bilinear for the image,
nearest for the mask). The image is then mapped to [-1, 1] and the mask is binarized
as mean(RGB) >= 0.5. We reproduce that here without the albumentations dependency.
"""
from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 224
SPLIT_FILES = {
    "bus": "bus/bus_train_test_names.pkl",
    "busi": "busi/busi_train_test_names.pkl",
    "glas": "glas/glas_train_test_names.pkl",
    "ham10000": "ham10000/ham10000_train_test_names.pkl",
    "kvasir-instrument": "kvasir-instrument/kvasir_train_test_names.pkl",
}


def split_names(get_root: str, dataset: str, split: str) -> list[str]:
    with open(os.path.join(get_root, "Dataset", SPLIT_FILES[dataset]), "rb") as f:
        return list(pickle.load(f)[split]["name_list"])


def load_pair(data_dir: str, name: str):
    """Return (image in [-1, 1] as (3, 224, 224) float32, GT mask (224, 224) uint8 {0,1})."""
    img = np.array(Image.open(os.path.join(data_dir, "images", name + ".png")).convert("RGB"), np.float32) / 255.0
    msk = np.array(Image.open(os.path.join(data_dir, "masks", name + ".png")).convert("RGB"), np.float32)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = (2 * img - 1).transpose(2, 0, 1).copy()
    gt = ((msk / 255.0).mean(-1) >= 0.5).astype(np.uint8)
    return img, gt


def check_dataset(get_root: str, dataset: str, data_dir: str) -> None:
    """Fail early, with a clear message, if images or masks are missing."""
    missing = []
    for split in ("train", "test"):
        for n in split_names(get_root, dataset, split):
            for sub in ("images", "masks"):
                if not os.path.isfile(os.path.join(data_dir, sub, n + ".png")):
                    missing.append(f"{sub}/{n}.png")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files missing in {data_dir} (e.g. {missing[:3]}). "
            "Run scripts/prepare_dataset.py on the raw download first.")
