"""ONNX export and ModelOpt INT8 post-training quantization for each GET part."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch

from .models import INPUT_SHAPES


def export_onnx(module: torch.nn.Module, part: str, path: str, opset: int = 17) -> str:
    """Export one GET part to ONNX with a fixed batch-1 input shape."""
    dummy = torch.randn(*INPUT_SHAPES[part])
    kw = dict(input_names=["input"], output_names=["output"], opset_version=opset)
    with torch.no_grad():
        try:
            torch.onnx.export(module.cpu(), dummy, path, dynamo=False, **kw)
        except TypeError:  # older torch has no dynamo argument
            torch.onnx.export(module.cpu(), dummy, path, **kw)
    return path


@torch.no_grad()
def make_calibration_sets(parts: dict, images: np.ndarray, out_dir: str, device: str = "cuda") -> dict:
    """Calibration inputs for each part, produced by the FP32 PyTorch model.

    images: (N, 3, 224, 224) float32 in [-1, 1], unlabeled training images.
    encoder sees the images; ETN sees FP32 encoder outputs; decoder sees FP32 ETN outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    feeds = {"encoder": [], "etn": [], "decoder": []}
    for i in range(len(images)):
        x = torch.from_numpy(images[i:i + 1]).to(device)
        mu = parts["encoder"].to(device)(x)
        out = parts["etn"].to(device)(mu)
        feeds["encoder"].append(images[i:i + 1])
        feeds["etn"].append(mu.cpu().numpy())
        feeds["decoder"].append(out.cpu().numpy())
    paths = {}
    for part, arrs in feeds.items():
        paths[part] = os.path.join(out_dir, f"calib_{part}.npy")
        np.save(paths[part], np.concatenate(arrs).astype(np.float32))
    return paths


def modelopt_int8(onnx_path: str, calib_npy: str, out_path: str, high_precision: str = "fp32",
                  calibration_method: str = "max", calibration_eps: list[str] | None = None,
                  extra_args: list[str] | None = None) -> str:
    """Insert QuantizeLinear/DequantizeLinear nodes with NVIDIA ModelOpt (fake quantization).

    Same command as our replication notebook. Runs as a separate process so ModelOpt
    can't patch ONNX Runtime in the notebook. high_precision = dtype for layers that are
    not quantized (fp32, or fp16 for an INT8 + FP16 mixed model).
    """
    cmd = [sys.executable, "-m", "modelopt.onnx.quantization",
           "--onnx_path", onnx_path,
           "--quantize_mode", "int8",
           "--calibration_data", calib_npy,
           "--calibration_method", calibration_method,
           "--output_path", out_path,
           "--high_precision_dtype", high_precision]
    if calibration_eps:  # e.g. ["cuda:0", "cpu"]: calibrate on the GPU, fall back to CPU
        cmd += ["--calibration_eps", *calibration_eps]
    cmd += list(extra_args or [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"ModelOpt failed (exit code {res.returncode}) for {onnx_path}:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return out_path


def modelopt_fp16(onnx_path: str, out_path: str) -> str:
    """Convert an FP32 ONNX model to FP16 with NVIDIA ModelOpt AutoCast.

    Inputs and outputs stay FP32 (--keep_io_types). AutoCast keeps a node in FP32 when its
    values would overflow FP16. Runs as a separate process, like modelopt_int8.
    """
    cmd = [sys.executable, "-m", "modelopt.onnx.autocast", "--onnx_path", onnx_path,
           "--output_path", out_path, "--low_precision_type", "fp16", "--keep_io_types",
           "--log_level", "WARNING"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"ModelOpt AutoCast failed (exit code {res.returncode}) for {onnx_path}:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return out_path
