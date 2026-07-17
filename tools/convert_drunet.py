"""Convert DRUNet color denoiser (cszn/KAIR, MIT) to Core ML.

Produces ../Models/DRUNet.mlpackage:
    input  "image": 512x512 RGB noisy tile
    input  "sigma": (1,) float32 noise level, 0-255 scale (e.g. 10 = mild film grain)
    output "denoised": 512x512 RGB

DRUNet is a bias-free residual U-Net with a noise-level map input — one model
covers the whole strength range.
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


class DRUNetWrapper(torch.nn.Module):
    """Builds the noise-level map channel from a scalar sigma input."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, sigma):
        noise_map = (sigma / 255.0).reshape(1, 1, 1, 1).expand(1, 1, image.shape[2], image.shape[3])
        x = torch.cat([image, noise_map], dim=1)
        return torch.clamp(self.net(x), 0, 1) * 255.0


def main(force: bool):
    out_path = MODELS_DIR / "DRUNet.mlpackage"
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

    wrapped = DRUNetWrapper(net).eval()
    example = (torch.rand(1, 3, SIZE, SIZE), torch.tensor([10.0]))
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE),
                         scale=1 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.TensorType(name="sigma", shape=(1,)),
        ],
        outputs=[ct.ImageType(name="denoised", color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.short_description = "DRUNet blind color denoiser, 512x512 tile + sigma (MIT)"
    MODELS_DIR.mkdir(exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")

    # Validate: denoise a noisy gradient, compare Core ML vs PyTorch.
    from PIL import Image
    rng = np.random.default_rng(9)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    clean = np.stack([x, (x + y) / 2, y], -1)
    noisy = np.clip(clean + rng.normal(0, 10 / 255, clean.shape), 0, 1)
    noisy8 = (noisy * 255).astype(np.uint8)

    coreml_out = np.asarray(mlmodel.predict({
        "image": Image.fromarray(noisy8),
        "sigma": np.array([10.0], dtype=np.float32),
    })["denoised"], dtype=np.float32)[..., :3]
    with torch.no_grad():
        tensor = torch.from_numpy(noisy8.astype(np.float32) / 255).permute(2, 0, 1)[None]
        torch_out = wrapped(tensor, torch.tensor([10.0]))[0].permute(1, 2, 0).numpy()
    diff = np.abs(coreml_out - torch_out)
    residual_noise = np.abs(coreml_out / 255.0 - clean).mean()
    print(f"CoreML-vs-torch mean diff: {diff.mean():.3f}/255; residual vs clean: {residual_noise:.4f} "
          f"(input noise was {np.abs(noisy - clean).mean():.4f})")
    if diff.mean() > 2.0:
        sys.exit("FAILED: Core ML output deviates from PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)
