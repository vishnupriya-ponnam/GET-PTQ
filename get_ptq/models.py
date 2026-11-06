"""Load GET (ETN + frozen SD-VAE) and split it into three exportable parts.

GET inference (see GET's valid.py):
    mu   = scale * VAE.encode(img).mean          # encoder part
    out  = ETN(mu)['out']                          # ETN part
    prob = decode(out)                             # decoder part
where decode() = clamp(out/scale) -> VAE decoder -> mean over RGB -> [0, 1].

Each part becomes its own ONNX file and TensorRT engine, so every part can run
at its own precision (FP32, FP16 or INT8). That is what makes the mixed-precision
experiments possible.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import yaml


def _import_get(get_root: str):
    """Put GET's repo on the path and return its modules."""
    get_root = os.path.abspath(get_root)
    if get_root not in sys.path:
        sys.path.insert(0, get_root)
    import networks.vae_setup.diffusion_model as dm

    # xformers attention cannot be exported to ONNX; force the plain attention block
    # (numerically the same computation).
    dm.XFORMERS_IS_AVAILBLE = False
    from networks.embedding_translation import MainPipeline

    return dm, MainPipeline


class SDVAE(nn.Module):
    """Same layout as GET's AutoencoderKL, minus the pytorch-lightning dependency."""

    def __init__(self, dm, ddconfig: dict, embed_dim: int):
        super().__init__()
        self.encoder = dm.Encoder(**ddconfig)
        self.decoder = dm.Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


class EncoderPart(nn.Module):
    """image in [-1, 1], (1, 3, 224, 224)  ->  scaled latent mean, (1, 4, 28, 28)."""

    def __init__(self, vae: SDVAE, scale: float):
        super().__init__()
        self.encoder, self.quant_conv, self.scale = vae.encoder, vae.quant_conv, float(scale)

    def forward(self, img):
        moments = self.quant_conv(self.encoder(img))
        mean, _logvar = torch.chunk(moments, 2, dim=1)
        return mean * self.scale


class ETNPart(nn.Module):
    """image latent (1, 4, 28, 28)  ->  predicted mask latent (1, 4, 28, 28)."""

    def __init__(self, etn: nn.Module):
        super().__init__()
        self.etn = etn

    def forward(self, mu):
        return self.etn(mu)["out"]


class DecoderPart(nn.Module):
    """predicted mask latent (1, 4, 28, 28)  ->  mask probability in [0, 1], (1, 1, 224, 224)."""

    def __init__(self, vae: SDVAE, scale: float):
        super().__init__()
        self.decoder, self.post_quant_conv, self.scale = vae.decoder, vae.post_quant_conv, float(scale)

    def forward(self, mu_pred):
        z = (mu_pred / self.scale).clamp(-10, 10)
        seg = self.decoder(self.post_quant_conv(z)).mean(1, keepdim=True).clamp(-1, 1)
        return ((seg + 1) * 0.5).clamp(0, 1)


PART_NAMES = ("encoder", "etn", "decoder")
INPUT_SHAPES = {"encoder": (1, 3, 224, 224), "etn": (1, 4, 28, 28), "decoder": (1, 4, 28, 28)}


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def load_get(get_root: str, dataset: str, vae_ckpt: str, etn_ckpt: str | None = None):
    """Build the three GET parts on CPU, in eval mode, with trained weights loaded.

    dataset: one of bus, busi, glas, ham10000, kvasir-instrument (selects GET's
    configs/<dataset>_valid.yaml and Results/valid_<dataset>/best_model.pth).
    """
    dm, MainPipeline = _import_get(get_root)
    cfg = yaml.safe_load(open(os.path.join(get_root, "configs", f"{dataset}_valid.yaml")))
    etn_ckpt = etn_ckpt or os.path.join(get_root, cfg["model_weight"])

    etn = MainPipeline(in_channel=cfg["in_channel"], out_channels=cfg["out_channels"],
                       ch=cfg["ch"], ch_mult=tuple(cfg["ch_mult"]))
    etn.load_state_dict(_load(etn_ckpt)["model"])

    vae_cfg = yaml.safe_load(open(os.path.join(get_root, cfg["vae_config_path"])))["first_stage_config"]
    vae = SDVAE(dm, dict(vae_cfg["params"]["ddconfig"]), vae_cfg["params"]["embed_dim"])
    state = _load(vae_ckpt)
    state = state.get("state_dict", state)
    keep = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
    state = {k: v for k, v in state.items() if k.startswith(keep)}
    vae.load_state_dict(state, strict=True)  # fails loudly if the VAE file is wrong
    scale = float(vae_cfg["scale_factor"])

    parts = {
        "encoder": EncoderPart(vae, scale).eval(),
        "etn": ETNPart(etn).eval(),
        "decoder": DecoderPart(vae, scale).eval(),
    }
    for p in parts.values():
        p.requires_grad_(False)
    return parts
