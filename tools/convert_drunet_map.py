"""Convert DRUNet color denoiser (cszn/KAIR, MIT) to Core ML with a
per-pixel noise-level map input.

Produces ../Models/DRUNetMap.mlpackage:
    input  "image":     512x512 RGB noisy tile
    input  "noise_map": 512x512 grayscale, pixel value = sigma on the
                        0-255 scale (trained range 0-50)
    output "denoised":  512x512 RGB

DRUNet's fourth input channel IS a noise-level map; convert_drunet.py
collapsed it to one scalar per tile. Film grain amplitude in gamma-encoded
pixels grows strongly with brightness (5x shadows-to-highlights on the
issue-1 cinema frame), so a single sigma always over-smooths the shadows
while leaving grain in the highlights. The app estimates the grain level
per brightness and feeds it here pixel by pixel.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import download

TOOLS = Path(__file__).resolve().parent
MODELS_DIR = TOOLS.parent / "Models"
VENDOR = TOOLS / "vendor" / "KAIR"
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
SIZE = 512


def ensure_repo():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/cszn/KAIR", str(VENDOR)], check=True)
    sys.path.insert(0, str(VENDOR))


class DRUNetMapWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, noise_map):
        x = torch.cat([image, noise_map], dim=1)
        return torch.clamp(self.net(x), 0, 1) * 255.0


def main(force: bool):
    out_path = MODELS_DIR / "DRUNetMap.mlpackage"
    if out_path.exists() and not force:
        print(f"{out_path.name} exists, skipping (use --force to redo)")
        return

    ensure_repo()
    from models.network_unet import UNetRes  # noqa: E402

    # DRUNet is bias-free.
    net = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                  act_mode="R", downsample_mode="strideconv",
                  upsample_mode="convtranspose", bias=False)
    weights_file = download(WEIGHTS_URL, TOOLS / "weights")
    net.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True), strict=True)
    net.eval()

    wrapped = DRUNetMapWrapper(net).eval()
    example = (torch.rand(1, 3, SIZE, SIZE), torch.full((1, 1, SIZE, SIZE), 10 / 255))
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE),
                         scale=1 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.ImageType(name="noise_map", shape=(1, 1, SIZE, SIZE),
                         scale=1 / 255.0, color_layout=ct.colorlayout.GRAYSCALE),
        ],
        outputs=[ct.ImageType(name="denoised", color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.short_description = "DRUNet color denoiser, 512x512 tile + per-pixel noise map (MIT)"
    MODELS_DIR.mkdir(exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")

    # Validate on a gradient with brightness-dependent noise: Core ML vs
    # PyTorch, and the spatially varying map must beat any single sigma.
    from PIL import Image
    rng = np.random.default_rng(9)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    # Gradient plus fine texture everywhere: over-smoothing must cost
    # something (a bare gradient would reward the largest sigma).
    texture = 0.04 * np.sin(x * SIZE / 3.0) * np.sin(y * SIZE / 4.0)
    clean = np.repeat((x * 0.8 + 0.1 + texture)[..., None], 3, -1)
    sigma_px = (4 + 36 * clean[..., 0]) / 255           # 7.6 .. 32 on 0-255
    noisy = np.clip(clean + rng.normal(0, 1, clean.shape) * sigma_px[..., None], 0, 1)
    noisy8 = (noisy * 255).astype(np.uint8)
    map8 = np.clip(sigma_px * 255, 0, 255).astype(np.uint8)

    def coreml(noise_map8):
        out = mlmodel.predict({"image": Image.fromarray(noisy8),
                               "noise_map": Image.fromarray(noise_map8, mode="L")})
        return np.asarray(out["denoised"], dtype=np.float32)[..., :3]

    coreml_out = coreml(map8)
    with torch.no_grad():
        tensor = torch.from_numpy(noisy8.astype(np.float32) / 255).permute(2, 0, 1)[None]
        nm = torch.from_numpy(map8.astype(np.float32) / 255)[None, None]
        torch_out = wrapped(tensor, nm)[0].permute(1, 2, 0).numpy()
    diff = np.abs(coreml_out - torch_out).mean()
    err_map = np.abs(coreml_out / 255 - clean).mean()
    dark = clean[..., 0] < 0.35
    def dark_detail_loss(out):
        # How much of the clean image's dark-band gradient survives.
        return np.abs(out[dark] / 255 - clean[dark]).mean()
    err_const = {s: np.abs(coreml(np.full_like(map8, s)) / 255 - clean).mean() for s in (10, 20, 32)}
    print(f"CoreML-vs-torch mean diff {diff:.3f}/255; error vs clean: map {err_map:.4f}, "
          + ", ".join(f"const{s} {e:.4f}" for s, e in err_const.items())
          + f"; dark-band error map {dark_detail_loss(coreml_out):.4f}")
    if diff > 2.0:
        sys.exit("FAILED: Core ML output deviates from PyTorch")
    if err_map > min(err_const.values()):
        sys.exit("FAILED: spatially varying map should beat every constant sigma")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)
