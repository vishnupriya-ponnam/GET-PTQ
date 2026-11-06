"""Regenerate GET_PTQ_Colab.ipynb from the cell list below (keeps the notebook easy to diff)."""
import json
import os

MD, CODE = "markdown", "code"
cells = [
(MD, """# GET + INT8 Post-Training Quantization: whole test set + qualitative results

Applies the PTQ pipeline from our replication notebook (Qu et al., 2025: ModelOpt Q/DQ → strongly typed TensorRT INT8, FP32 TensorRT baseline with TF32 off) to **GET** (Rahman et al., 2026).

**What this notebook produces (the two things asked for this week):**
1. **Quantitative results on the whole test set**: Dice, IoU and HD95 for every test image, averaged per precision configuration, plus latency and engine size.
2. **Qualitative results**: input image, true mask, FP32 mask and the mask after quantization, side by side.

**How GET is split:** GET = frozen SD-VAE encoder → ETN (trained) → frozen SD-VAE decoder. Each part becomes its own TensorRT engine, so each part can run in FP32, FP16 or INT8. That gives the **mixed-precision** configurations.

| Config | Encoder | ETN | Decoder |
|---|---|---|---|
| `fp32` (baseline) | FP32 | FP32 | FP32 |
| `fp16` | FP16 | FP16 | FP16 |
| `int8_all` | INT8 | INT8 | INT8 |
| `int8_enc` / `int8_etn` / `int8_dec` | one part INT8, rest FP32 | | |
| `mix_vae8_etn16` | INT8 | FP16 | INT8 |
| `mix_enc8_fp16` | INT8 | FP16 | FP16 |
| `mix_int8_fp16` | INT8 + FP16 for non-quantized layers, all parts | | |

**How to run:** *Runtime → Change runtime type → T4 GPU* → *Runtime → Run all*. If Colab asks to restart after installing packages, restart, then run from the **Settings** cell down.
"""),
(MD, "## 1. Settings"),
(CODE, """DATASET   = "busi"      # busi | bus | glas | ham10000 | kvasir-instrument
CONFIGS   = ["fp32", "fp16", "int8_all", "int8_enc", "int8_etn", "int8_dec",
             "mix_vae8_etn16", "mix_enc8_fp16", "mix_int8_fp16"]
N_CALIB   = 64            # unlabeled training images used for INT8 calibration
CALIB_METHOD = "max"      # same as the PTQ paper; "entropy" is the other option
MAX_TEST  = None          # None = WHOLE test set (what we report). Use e.g. 5 only for a quick dry run.

# Where this repo lives on GitHub (leave as is once it is pushed; or upload the folder to /content/GET-PTQ)
GET_PTQ_REPO = "https://github.com/vishnupriya-ponnam/GET-PTQ.git"

# Raw dataset location. BUSI downloads automatically from Kaggle; for the others, put the raw
# download in Google Drive and set the path, e.g. "/content/drive/MyDrive/datasets/GlaS"
RAW_DATA_DIR = None

SAVE_TO_DRIVE = True      # copy results (tables + figures) to Google Drive
DRIVE_OUT     = f"/content/drive/MyDrive/GET-PTQ-results/{DATASET}"
"""),
(MD, "## 2. Install packages and get the code"),
(CODE, """!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "nvidia-modelopt[onnx]" tensorrt onnx medpy einops omegaconf gdown kagglehub tabulate opencv-python-headless
"""),
(CODE, """import os, sys, shutil, subprocess
os.chdir("/content")
if not os.path.isdir("/content/GET"):
    !git clone -q --depth 1 https://github.com/maklachur/GET.git /content/GET
if not os.path.isdir("/content/GET-PTQ"):
    !git clone -q --depth 1 {GET_PTQ_REPO} /content/GET-PTQ
sys.path.insert(0, "/content/GET-PTQ")
GET_ROOT = "/content/GET"
print("GET checkpoints:", sorted(os.listdir(f"{GET_ROOT}/Results")))
"""),
(MD, """## 3. Frozen SD-VAE weights

GET's README links this file on Google Drive; the Hugging Face copy is the fallback."""),
(CODE, """VAE_CKPT = f"{GET_ROOT}/SD-VAE-weight/768-v-ema-first-stage-VAE.ckpt"
if not os.path.isfile(VAE_CKPT):
    !gdown -q 1NM9pjv7IJe-HBLrqo1IGATYHHkyT9jTw -O "{VAE_CKPT}"
if not os.path.isfile(VAE_CKPT) or os.path.getsize(VAE_CKPT) < 50e6:
    !wget -q -O "{VAE_CKPT}" https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt
print(f"VAE checkpoint: {os.path.getsize(VAE_CKPT)/1e6:.0f} MB")
"""),
(MD, """## 4. Dataset

GET's split files list exactly which images are train and test. `prepare_dataset.py` converts the raw download into GET's layout (`images/<name>.png`, `masks/<name>.png`) and stops with a clear message if anything is missing."""),
(CODE, """if SAVE_TO_DRIVE or (RAW_DATA_DIR or "").startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

DATA_DIR = f"{GET_ROOT}/Dataset/{DATASET}"
raw = RAW_DATA_DIR
if raw is None and DATASET == "busi":
    import kagglehub
    raw = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
assert raw, "Set RAW_DATA_DIR to the raw download for this dataset (see the README)."
!python /content/GET-PTQ/scripts/prepare_dataset.py --dataset {DATASET} --raw "{raw}" --get_root {GET_ROOT} --out {DATA_DIR}
"""),
(MD, """## 5. Run every configuration on the whole test set

Steps (cached, so re-running is fast): export each part to ONNX → calibration inputs from FP32 → ModelOpt INT8 Q/DQ → build TensorRT engines → run every test image through every configuration → metrics.
The first run takes a while, mostly building the VAE decoder engines."""),
(CODE, """from get_ptq.experiments import DEFAULT_CONFIGS, run_experiments
from get_ptq.report import summary_markdown, summary_table

OUT = f"/content/results/{DATASET}"
configs = {k: DEFAULT_CONFIGS[k] for k in CONFIGS}
res = run_experiments(GET_ROOT, DATASET, DATA_DIR, VAE_CKPT, OUT, work_dir="/content/work/" + DATASET,
                      configs=configs, n_calib=N_CALIB, calib_method=CALIB_METHOD, max_test=MAX_TEST)
open(f"{OUT}/summary.md", "w").write(summary_markdown(res["summary"], res["env"]))
"""),
(MD, """## 6. Quantitative results (whole test set)

- **ΔDice / ΔHD95**: change vs. the FP32 TensorRT engine (negative ΔDice = worse).
- **Mask agreement**: Dice between the quantized mask and the FP32 mask (1.0 = identical masks).
- **PyTorch FP32** row: sanity check that our numbers match GET's own inference."""),
(CODE, """from IPython.display import display, Markdown
print(res["env"])
display(summary_table(res["summary"]))
"""),
(CODE, """# Which layers actually run in INT8 / FP16 inside each engine
display(res["engines"])
"""),
(CODE, """# Per-image results (first rows); the full table is per_image.csv
display(res["per_image"].head(12))
"""),
(MD, "## 7. Qualitative results: true mask vs. mask after quantization"),
(CODE, """from get_ptq.figures import make_all_figures
from IPython.display import Image as IPImage
focus = [c for c in ("int8_all", "mix_int8_fp16", "fp16") if c in configs]
fig_paths = make_all_figures(OUT, DATA_DIR, configs, focus_configs=focus, k=6, dataset=DATASET)
for p in fig_paths:
    print(p); display(IPImage(p))
"""),
(MD, """Rows are picked automatically: the images whose Dice dropped most after quantization, then typical and best cases. Columns: input, ground truth, FP32 mask, quantized mask, a difference map (white = both, blue = only FP32, red = only quantized) and all three outlines on the image."""),
(MD, "## 8. Save results"),
(CODE, """from get_ptq.figures import save_mask_pngs
save_mask_pngs(OUT, configs=[c for c in ("fp32", "int8_all") if c in configs])   # every predicted mask as PNG
if SAVE_TO_DRIVE:
    shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True)
    print("Copied to", DRIVE_OUT)
!cd /content/results && zip -qr {DATASET}_results.zip {DATASET} && ls -la /content/results
"""),
]

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == CODE:
        c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GET_PTQ_Colab.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
