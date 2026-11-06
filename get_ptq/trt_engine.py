"""TensorRT engine building and running (adapted from our PTQ replication notebook).

Precisions:
  fp32 : FP32 engine with TF32 turned off (the true FP32 baseline, as in the PTQ paper)
  fp16 : FP16 engine built from the FP32 ONNX file
  int8 / int8_fp16 : strongly typed engine built from a ModelOpt Q/DQ ONNX file, so
         TensorRT follows the Q/DQ types exactly (real INT8 kernels)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_MAJOR = int(trt.__version__.split(".")[0])
NDF = trt.NetworkDefinitionCreationFlag


def build_engine(onnx_path: str, engine_path: str, precision: str, workspace_gb: float = 4.0) -> str:
    builder = trt.Builder(LOGGER)
    flags = 0 if TRT_MAJOR >= 10 else 1 << int(NDF.EXPLICIT_BATCH)
    strongly_typed = precision.startswith("int8") and hasattr(NDF, "STRONGLY_TYPED")
    if strongly_typed:
        flags |= 1 << int(NDF.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(onnx_path):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not strongly_typed:
        config.clear_flag(trt.BuilderFlag.TF32)  # true FP32 baseline
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision.startswith("int8"):  # fallback for TensorRT without strong typing
            config.set_flag(trt.BuilderFlag.INT8)
            if precision == "int8_fp16":
                config.set_flag(trt.BuilderFlag.FP16)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path} ({precision})")
    with open(engine_path, "wb") as f:
        f.write(data)
    return engine_path


def _torch_dtype(trt_dtype):
    return torch.from_numpy(np.zeros(1, dtype=trt.nptype(trt_dtype))).dtype


class TRTRunner:
    """Loads an engine once; takes and returns CUDA torch tensors (output as float32)."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = _torch_dtype(self.engine.get_tensor_dtype(self.inp))
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(out)),
                               dtype=_torch_dtype(self.engine.get_tensor_dtype(out)), device="cuda")
        self.ctx.set_tensor_address(out, self.out.data_ptr())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.in_dtype).contiguous()
        s = torch.cuda.current_stream()
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
        out = self.out.float()
        # always hand back a copy, so the result survives the next call of this engine
        return out.clone() if out.data_ptr() == self.out.data_ptr() else out


def median_latency_ms(fn, x, warmup: int = 20, iters: int = 100) -> float:
    """Median latency at batch size 1, same method as our replication notebook."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def layer_precision_counts(engine_path: str) -> dict:
    """How many engine layers TensorRT built, and how many run in Int8 / Half."""
    with open(engine_path, "rb") as f:
        eng = trt.Runtime(LOGGER).deserialize_cuda_engine(f.read())
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = [L for L in info.get("Layers", []) if isinstance(L, dict)]
    text = [json.dumps(L) for L in layers]
    return {"layers": len(layers),
            "int8_layers": sum("Int8" in t for t in text),
            "fp16_layers": sum("Half" in t for t in text)}


def mb(path: str) -> float:
    return os.path.getsize(path) / 1e6
