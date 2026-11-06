# GET-PTQ: real INT8 / mixed-precision post-training quantization for GET

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishnupriya-ponnam/GET-PTQ/blob/main/GET_PTQ_Colab.ipynb)

This repo applies the real post-training quantization (PTQ) pipeline of
**Qu et al. (2025)** ([arXiv:2501.17343](https://arxiv.org/abs/2501.17343)) to **GET**, a
generative medical image segmentation model (**Rahman et al., 2026**,
[arXiv:2608.22619](https://arxiv.org/abs/2608.22619), [code](https://github.com/maklachur/GET)).

It runs on the **whole test set** and produces:

- **Quantitative results**: Dice, IoU and HD95 for every test image and their averages for each precision
  setup. It also reports how much each mask changed compared with FP32, plus latency and engine size.
- **Qualitative results**: the input image, true mask, FP32 mask and quantized mask side by side,
  with a difference map and outlines.

## How it works

GET has three parts in sequence. Each one is exported and built as **its own TensorRT engine**,
so each can run at its own precision:

```
image ──► SD-VAE encoder (frozen) ──► ETN (trained, 1.07M params) ──► SD-VAE decoder (frozen) ──► mask
            34.2M params                                              49.5M params
```

For each part:

1. Export the FP32 PyTorch module to ONNX (fixed input shape, batch size 1).
2. For INT8: build calibration inputs from unlabeled **training** images run through the FP32 model.
   NVIDIA ModelOpt then inserts QuantizeLinear/DequantizeLinear nodes using `max` calibration
   (fake quantization), as in our replication notebook.
3. Build a TensorRT engine:
   - The FP32 engine has TF32 turned off, so it's a true FP32 baseline.
   - The FP16 engine is built with TensorRT's FP16 flag.
   - The INT8 engines are strongly typed builds from the Q/DQ model, so they use real INT8 kernels.
4. Run every test image through every configuration and compare the masks with the ground truth and
   with the FP32 TensorRT masks.

### Precision configurations

| Config | Encoder | ETN | Decoder | Question it answers |
|---|---|---|---|---|
| `fp32` | FP32 | FP32 | FP32 | Baseline |
| `fp16` | FP16 | FP16 | FP16 | Is plain half precision already enough? |
| `int8_all` | INT8 | INT8 | INT8 | Fastest and smallest: how much accuracy is lost? |
| `int8_enc` | INT8 | FP32 | FP32 | Sensitivity of the encoder |
| `int8_etn` | FP32 | INT8 | FP32 | Sensitivity of the ETN |
| `int8_dec` | FP32 | FP32 | INT8 | Sensitivity of the decoder (mask boundaries) |
| `mix_vae8_etn16` | INT8 | FP16 | INT8 | Mixed: quantize the heavy VAE, keep the small ETN precise |
| `mix_enc8_fp16` | INT8 | FP16 | FP16 | Mixed: INT8 encoder only |
| `mix_int8_fp16` | INT8+FP16 | INT8+FP16 | INT8+FP16 | Mixed: INT8 convolutions, FP16 for every non-quantized layer |

A `pytorch_fp32` row is also reported as a sanity check that we reproduce GET's own inference.

## Run it in Google Colab (recommended)

1. Open `GET_PTQ_Colab.ipynb` in Colab (*File → Upload notebook*, or open it from GitHub).
2. *Runtime → Change runtime type → T4 GPU*.
3. In the **Settings** cell, set `GET_PTQ_REPO` to this repo's URL and choose `DATASET`.
4. *Runtime → Run all*.

Results are written to `/content/results/<dataset>/` and copied to Google Drive:

| File | Contents |
|---|---|
| `summary.csv`, `summary.md` | One row per configuration: Dice/IoU/HD95 mean ± std, Δ vs FP32, mask agreement, Wilcoxon p-value, latency per part, speedup, engine size |
| `per_image.csv` | Every test image × every configuration |
| `engines.csv` | Engine size and how many layers run in INT8 / FP16 |
| `figures/qualitative_<config>.png` | Input, ground truth, FP32 mask, quantized mask, difference map, outlines |
| `figures/qualitative_all_configs.png` | The same images with one column per configuration |
| `figures/accuracy_vs_latency.png` | Accuracy/speed trade-off |
| `masks/<config>/*.png` | Every predicted mask |
| `predictions.npz` | All masks (bit-packed), so figures can be remade without the GPU |

## Datasets

GET's split files (`GET/Dataset/<dataset>/*_train_test_names.pkl`) define the exact train/test images.
`scripts/prepare_dataset.py` converts a raw download into GET's layout, and stops if any image is missing.

| Dataset | Test images | Raw download | Notes |
|---|---|---|---|
| BUSI | 130 | Kaggle `aryashah2k/breast-ultrasound-images-dataset` (downloaded automatically in the notebook) | `benign (3).png` → `benign_3`; multiple masks per image are merged |
| BUS | 31 | Dataset B, Yap et al. (on request) | `original/` + `GT/` folders |
| GlaS | 80 | Warwick GlaS contest | `*_anno.bmp` instance labels → binary |
| HAM10000 | 2000 | ISIC / Harvard Dataverse | images + `*_segmentation.png` |
| Kvasir-Instrument | 118 | Simula | `images/` + `masks/` |

For anything other than BUSI, put the raw download in Google Drive and set `RAW_DATA_DIR` in the notebook.

## Run it from the command line (any machine with an NVIDIA GPU)

```bash
git clone https://github.com/maklachur/GET.git
pip install -r requirements.txt
# put the SD-VAE weights in GET/SD-VAE-weight/ (see GET's README)
python scripts/prepare_dataset.py --dataset busi --raw /path/to/Dataset_BUSI_with_GT --get_root GET
python scripts/run_experiments.py --get_root GET --dataset busi \
    --vae_ckpt GET/SD-VAE-weight/768-v-ema-first-stage-VAE.ckpt --out results/busi
```

## Code layout

```
get_ptq/models.py       load GET; split it into EncoderPart, ETNPart, DecoderPart
get_ptq/data.py         test-time preprocessing identical to GET (resize 224, image to [-1, 1], mask >= 0.5)
get_ptq/metrics.py      Dice, IoU, HD95 exactly as GET's valid.py
get_ptq/quantize.py     ONNX export, calibration sets, ModelOpt INT8 Q/DQ
get_ptq/trt_engine.py   TensorRT build/run, latency, per-layer precision (from our replication notebook)
get_ptq/experiments.py  configurations and the whole-test-set run
get_ptq/figures.py      qualitative figures
get_ptq/report.py       summary table for slides
```

## What has been checked, and what hasn't

Checked on CPU, without TensorRT:

- ONNX export of all three parts matches PyTorch (max abs. difference < 1e-4).
- ModelOpt INT8 Q/DQ insertion works for the ETN and VAE parts, with FP32 or FP16 for non-quantized layers.
- HD95 matches medpy exactly.
- The full pipeline, tables and figures run end to end on synthetic data.

Not checked yet: TensorRT engine builds and the real numbers. Those need the Colab GPU run.

## Notes and caveats

- GET's checkpoints are taken from GET's repo (`Results/valid_<dataset>/best_model.pth`). The paper
  selected those checkpoints using test-set Dice. We use them as released.
- Calibration uses training images only, never test images.
- Latency is the median over 100 runs at batch size 1. It includes the small casts between engines.
- The SD-VAE file from GET's Google Drive link is used first. The Hugging Face `sd-vae-ft-ema` file is the fallback.
