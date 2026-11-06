"""Run GET under several precision configurations on the WHOLE test set.

For every configuration we record, per test image: Dice, IoU, HD95 against the
ground truth, plus how much the mask changed relative to the FP32 TensorRT mask.
We also record per-part latency and engine size, and save every predicted mask so
the qualitative figures can be made (or remade) later.
"""
from __future__ import annotations

import os
import platform
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from .data import check_dataset, load_pair, split_names
from .metrics import all_metrics, dice_iou
from .models import INPUT_SHAPES, PART_NAMES, load_get
from .quantize import export_onnx, make_calibration_sets, modelopt_fp16, modelopt_int8

# slug: (label, encoder precision, ETN precision, decoder precision)
# precisions: fp32 | fp16 | int8 (INT8 Q/DQ, other layers FP32) | int8_fp16 (INT8 Q/DQ, other layers FP16)
DEFAULT_CONFIGS = OrderedDict([
    ("fp32",           ("FP32 (TensorRT baseline)",            "fp32", "fp32", "fp32")),
    ("fp16",           ("FP16 all parts",                      "fp16", "fp16", "fp16")),
    ("int8_all",       ("INT8 all parts",                      "int8", "int8", "int8")),
    ("int8_enc",       ("INT8 encoder only",                   "int8", "fp32", "fp32")),
    ("int8_etn",       ("INT8 ETN only",                       "fp32", "int8", "fp32")),
    ("int8_dec",       ("INT8 decoder only",                   "fp32", "fp32", "int8")),
    ("mix_vae8_etn16", ("Mixed: INT8 VAE + FP16 ETN",          "int8", "fp16", "int8")),
    ("mix_enc8_fp16",  ("Mixed: INT8 encoder + FP16 ETN/dec",  "int8", "fp16", "fp16")),
    ("mix_int8_fp16",  ("Mixed: INT8 + FP16 fallback, all",    "int8_fp16", "int8_fp16", "int8_fp16")),
])


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_engines(parts, configs, onnx_dir, engine_dir, calib_dir, train_images,
                     calib_method, log, device="cuda"):
    """Export ONNX, quantize and build every (part, precision) engine the configs need."""
    from .trt_engine import build_engine, layer_precision_counts, mb

    needed = sorted({(part, prec) for _, *precs in configs.values() for part, prec in zip(PART_NAMES, precs)})
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(engine_dir, exist_ok=True)

    onnx_fp32 = {}
    for part in PART_NAMES:
        path = os.path.join(onnx_dir, f"{part}_fp32.onnx")
        if not os.path.isfile(path):
            log(f"Exporting {part} to ONNX")
            export_onnx(parts[part], part, path)
        onnx_fp32[part] = path

    calib = None
    if any(p.startswith("int8") for _, p in needed):
        log(f"Making calibration sets from {len(train_images)} unlabeled training images")
        calib = make_calibration_sets(parts, train_images, calib_dir, device=device)

    engines, info = {}, []
    for part, prec in needed:
        src = onnx_fp32[part]
        if prec == "fp16":
            src = os.path.join(onnx_dir, f"{part}_fp16.onnx")
            if not os.path.isfile(src):
                log(f"ModelOpt AutoCast to FP16: {part}")
                modelopt_fp16(onnx_fp32[part], src)
        elif prec.startswith("int8"):
            hp = "fp16" if prec == "int8_fp16" else "fp32"
            src = os.path.join(onnx_dir, f"{part}_{prec}_{calib_method}_qdq.onnx")
            if not os.path.isfile(src):
                log(f"ModelOpt INT8 ({calib_method} calibration, {hp} for other layers): {part}")
                modelopt_int8(onnx_fp32[part], calib[part], src, high_precision=hp,
                              calibration_method=calib_method,
                              calibration_eps=["cuda:0", "cpu"] if device == "cuda" else ["cpu"])
        tag = f"{prec}_{calib_method}" if prec.startswith("int8") else prec
        eng = os.path.join(engine_dir, f"{part}_{tag}.engine")
        if not os.path.isfile(eng):
            log(f"Building TensorRT engine: {part} {prec}")
            t0 = time.time()
            build_engine(src, eng, prec)
            log(f"   done in {time.time() - t0:.0f} s, {mb(eng):.1f} MB")
        engines[(part, prec)] = eng
        info.append({"part": part, "precision": prec, "engine_mb": mb(eng), **layer_precision_counts(eng)})
    return engines, pd.DataFrame(info)


def run_experiments(get_root: str, dataset: str, data_dir: str, vae_ckpt: str, out_dir: str,
                    work_dir: str | None = None, configs: dict | None = None, n_calib: int = 64,
                    calib_method: str = "max", seed: int = 0, max_test: int | None = None,
                    latency_iters: int = 100, verbose: bool = True, device: str = "cuda") -> dict:
    """Full pipeline. Returns a dict of DataFrames; also writes CSVs and predictions to out_dir."""
    from .trt_engine import TRTRunner, median_latency_ms

    log = print if verbose else (lambda *a, **k: None)
    configs = configs or DEFAULT_CONFIGS
    if "fp32" not in configs:
        configs = OrderedDict([("fp32", DEFAULT_CONFIGS["fp32"]), *configs.items()])
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(out_dir, exist_ok=True)

    # True FP32 everywhere (same rule as the replication notebook)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ---------------------------------------------------------------- data
    check_dataset(get_root, dataset, data_dir)
    test_names = split_names(get_root, dataset, "test")
    if max_test:
        test_names = test_names[:max_test]
    train_names = split_names(get_root, dataset, "train")
    rng = np.random.RandomState(seed)
    calib_names = [train_names[i] for i in rng.permutation(len(train_names))[:n_calib]]
    log(f"{dataset}: {len(test_names)} test images, {len(calib_names)} calibration images (from train split)")

    test_imgs, test_gts = [], []
    for n in tqdm(test_names, desc="Loading test set", disable=not verbose):
        img, gt = load_pair(data_dir, n)
        test_imgs.append(img.astype(np.float16) if len(test_names) > 500 else img)
        test_gts.append(gt)
    train_images = np.stack([load_pair(data_dir, n)[0] for n in calib_names]).astype(np.float32)

    # ---------------------------------------------------------------- models and engines
    parts = load_get(get_root, dataset, vae_ckpt)
    engines, engine_info = _prepare_engines(
        parts, configs, os.path.join(work_dir, "onnx"), os.path.join(work_dir, "engines"),
        os.path.join(work_dir, "calib"), train_images, calib_method, log, device)
    engine_info.to_csv(os.path.join(out_dir, "engines.csv"), index=False)

    preds = OrderedDict()
    per_image = []
    latency_rows = []

    # ---------------------------------------------------------------- PyTorch FP32 reference
    torch_parts = {k: v.to(device) for k, v in parts.items()}

    @torch.no_grad()
    def torch_chain(x):
        return torch_parts["decoder"](torch_parts["etn"](torch_parts["encoder"](x)))

    def run_chain(chain, slug):
        out = []
        for i, n in enumerate(tqdm(test_names, desc=slug, disable=not verbose)):
            x = torch.from_numpy(test_imgs[i].astype(np.float32))[None].to(device)
            prob = torch.nan_to_num(chain(x), nan=0.0, posinf=1.0, neginf=0.0)
            out.append((prob >= 0.5).squeeze().cpu().numpy().astype(np.uint8))
        return np.stack(out)

    preds["pytorch_fp32"] = run_chain(torch_chain, "pytorch_fp32")
    x0 = torch.from_numpy(test_imgs[0].astype(np.float32))[None].to(device)
    latency_rows.append({"config": "pytorch_fp32",
                         "latency_total_ms": median_latency_ms(torch_chain, x0, iters=latency_iters)})
    for p in torch_parts.values():
        p.cpu()
    del torch_parts
    _free_gpu()

    # ---------------------------------------------------------------- TensorRT configurations
    for slug, (label, *precs) in configs.items():
        runners = {part: TRTRunner(engines[(part, prec)]) for part, prec in zip(PART_NAMES, precs)}

        def chain(x, r=runners):
            return r["decoder"](r["etn"](r["encoder"](x)))

        preds[slug] = run_chain(chain, slug)
        row = {"config": slug}
        # per-part latency on realistic inputs
        with torch.no_grad():
            mu = runners["encoder"](x0).clone()
            out = runners["etn"](mu).clone()
        for part, inp in zip(PART_NAMES, (x0, mu, out)):
            row[f"latency_{part}_ms"] = median_latency_ms(runners[part], inp, iters=latency_iters)
        row["latency_total_ms"] = median_latency_ms(chain, x0, iters=latency_iters)
        latency_rows.append(row)
        del runners
        _free_gpu()

    # ---------------------------------------------------------------- metrics
    ref = preds["fp32"]
    for slug, P in preds.items():
        for i, n in enumerate(test_names):
            m = all_metrics(P[i], test_gts[i])
            agree, _ = dice_iou(P[i], ref[i])
            if P[i].sum() == 0 and ref[i].sum() == 0:
                agree = 1.0
            per_image.append({"config": slug, "name": n, **m, "agreement_dice_vs_fp32": agree,
                              "pixels_changed_vs_fp32": int((P[i] != ref[i]).sum())})
    per_image = pd.DataFrame(per_image)

    labels = {"pytorch_fp32": ("PyTorch FP32 (reference)", "pytorch", "pytorch", "pytorch"), **configs}
    lat = pd.DataFrame(latency_rows).set_index("config")
    sizes = engine_info.set_index(["part", "precision"])["engine_mb"]
    fp32_rows = per_image[per_image.config == "fp32"].set_index("name")
    rows = []
    for slug in preds:
        g = per_image[per_image.config == slug].set_index("name")
        label, *precs = labels[slug]
        r = {"config": slug, "label": label, "encoder": precs[0], "etn": precs[1], "decoder": precs[2],
             "n_images": len(g),
             "dice_mean": g.dice.mean(), "dice_std": g.dice.std(ddof=1),
             "iou_mean": g.iou.mean(), "iou_std": g.iou.std(ddof=1),
             "hd95_mean": g.hd95.mean(), "hd95_std": g.hd95.std(ddof=1),
             "delta_dice_vs_fp32": g.dice.mean() - fp32_rows.dice.mean(),
             "delta_hd95_vs_fp32": g.hd95.mean() - fp32_rows.hd95.mean(),
             "mask_agreement_dice": g.agreement_dice_vs_fp32.mean(),
             "images_changed": int((g.pixels_changed_vs_fp32 > 0).sum()),
             "pct_pixels_changed": 100 * g.pixels_changed_vs_fp32.sum() / (len(g) * 224 * 224)}
        diff = (g.dice - fp32_rows.dice).values
        r["wilcoxon_p_dice"] = (float(wilcoxon(diff).pvalue) if slug != "fp32" and np.any(diff != 0)
                                else np.nan)
        r.update(lat.loc[slug].dropna().to_dict())
        if slug != "pytorch_fp32":
            r["engine_mb_total"] = sum(sizes[(part, prec)] for part, prec in zip(PART_NAMES, precs))
        rows.append(r)
    summary = pd.DataFrame(rows)
    base = summary.set_index("config").loc["fp32"]
    summary["speedup_vs_fp32"] = base["latency_total_ms"] / summary["latency_total_ms"]
    summary["size_ratio_vs_fp32"] = base["engine_mb_total"] / summary["engine_mb_total"]

    # ---------------------------------------------------------------- save
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_image.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), names=np.array(test_names),
                        gt=np.packbits(np.stack(test_gts), axis=-1),
                        **{k: np.packbits(v, axis=-1) for k, v in preds.items()})
    n_full = len(split_names(get_root, dataset, "test"))
    env = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "python": platform.python_version(),
           "dataset": dataset, "n_test": len(test_names), "n_test_full": n_full, "n_calib": len(calib_names),
           "calib_method": calib_method, "seed": seed}
    try:
        import tensorrt
        env["tensorrt"] = tensorrt.__version__
        import modelopt
        env["modelopt"] = modelopt.__version__
    except Exception:
        pass
    pd.Series(env).to_csv(os.path.join(out_dir, "environment.csv"), header=False)
    log(f"Saved results to {out_dir}")
    return {"summary": summary, "per_image": per_image, "engines": engine_info, "env": env}


def load_predictions(out_dir: str) -> dict:
    """Read predictions.npz back as {config: (N, 224, 224) uint8} plus names and gt."""
    z = np.load(os.path.join(out_dir, "predictions.npz"))
    out = {"names": list(z["names"])}
    for k in z.files:
        if k != "names":
            out[k] = np.unpackbits(z[k], axis=-1)[..., :224]
    return out
