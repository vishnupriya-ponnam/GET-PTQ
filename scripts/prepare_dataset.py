"""Convert a raw dataset download into the layout GET expects.

GET reads   <out>/images/<name>.png   and   <out>/masks/<name>.png
where <name> comes from GET's split file (e.g. Dataset/busi/busi_train_test_names.pkl).

Supported raw downloads (point --raw at the folder you downloaded / unzipped):
  busi              Kaggle "Breast Ultrasound Images Dataset" (Dataset_BUSI_with_GT):
                    benign/benign (3).png + benign (3)_mask.png [+ _mask_1.png ...]  -> benign_3
                    Multiple masks for one image are merged (union).
  bus               BUS "Dataset B" (Yap et al.): original/000130.png + GT/000130.png
  glas              GlaS (Warwick QU): testA_1.bmp + testA_1_anno.bmp (instance labels -> binary)
  ham10000          ISIC_0024306.jpg + ISIC_0024306_segmentation.png
  kvasir-instrument images/<id>.jpg + masks/<id>.png

Usage:
  python scripts/prepare_dataset.py --dataset busi --raw /path/to/Dataset_BUSI_with_GT \
      --get_root /path/to/GET --out /path/to/GET/Dataset/busi
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

SPLITS = {
    "bus": "bus/bus_train_test_names.pkl",
    "busi": "busi/busi_train_test_names.pkl",
    "glas": "glas/glas_train_test_names.pkl",
    "ham10000": "ham10000/ham10000_train_test_names.pkl",
    "kvasir-instrument": "kvasir-instrument/kvasir_train_test_names.pkl",
}
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_files(root):
    files = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(EXTS):
                files.append(os.path.join(d, f))
    return files


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def find_pairs(dataset, files):
    """Map GET name -> (image path, [mask paths])."""
    pairs = defaultdict(lambda: [None, []])
    if dataset == "busi":
        for p in files:
            m = re.match(r"^(benign|malignant|normal) \((\d+)\)(_mask(_\d+)?)?$", stem(p))
            if not m:
                continue
            name = f"{m.group(1)}_{m.group(2)}"
            if m.group(3):
                pairs[name][1].append(p)
            else:
                pairs[name][0] = p
    elif dataset == "bus":
        for p in files:
            parent = os.path.basename(os.path.dirname(p)).lower()
            if parent in ("gt", "mask", "masks", "groundtruth"):
                pairs[stem(p)][1].append(p)
            elif parent in ("original", "images", "image"):
                pairs[stem(p)][0] = p
    elif dataset == "glas":
        for p in files:
            s = stem(p)
            if s.endswith("_anno"):
                pairs[s[:-5]][1].append(p)
            elif re.match(r"^(train|testA|testB)_\d+$", s):
                pairs[s][0] = p
    elif dataset == "ham10000":
        for p in files:
            s = stem(p)
            if s.endswith("_segmentation"):
                pairs[s[: -len("_segmentation")]][1].append(p)
            elif s.startswith("ISIC_"):
                pairs[s][0] = p
    elif dataset == "kvasir-instrument":
        for p in files:
            parent = os.path.basename(os.path.dirname(p)).lower()
            if "mask" in parent:
                pairs[stem(p)][1].append(p)
            elif "image" in parent:
                pairs[stem(p)][0] = p
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPLITS))
    ap.add_argument("--raw", required=True, help="folder with the raw download (searched recursively)")
    ap.add_argument("--get_root", required=True, help="path to the cloned GET repo")
    ap.add_argument("--out", help="output folder (default: <get_root>/Dataset/<dataset>)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.get_root, "Dataset", a.dataset)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)

    with open(os.path.join(a.get_root, "Dataset", SPLITS[a.dataset]), "rb") as f:
        split = pickle.load(f)
    needed = [n for s in ("train", "test") for n in split[s]["name_list"]]

    pairs = find_pairs(a.dataset, index_files(a.raw))
    missing = []
    for n in needed:
        img_p, mask_ps = pairs.get(n, (None, []))
        if img_p is None or not mask_ps:
            missing.append(n)
            continue
        Image.open(img_p).convert("RGB").save(os.path.join(out, "images", n + ".png"))
        mask = None
        for mp in sorted(mask_ps):
            m = np.array(Image.open(mp).convert("L")) > 0   # binary; GlaS instance ids -> foreground
            mask = m if mask is None else (mask | m)
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out, "masks", n + ".png"))

    print(f"{a.dataset}: wrote {len(needed) - len(missing)} / {len(needed)} image-mask pairs to {out}")
    if missing:
        print(f"MISSING {len(missing)} names, e.g. {missing[:5]}. Check that --raw points at the full download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
