"""Convert GFPGAN v1.4 (TencentARC/GFPGAN, Apache-2.0) to Core ML.

Produces ../Models/GFPGAN.mlpackage:
    input  "face":     512x512 RGB aligned face crop
    output "restored": 512x512 RGB restored face

The "clean" architecture variant has no custom CUDA ops and traces cleanly.
Face detection/alignment happens in Swift via the Vision framework.
"""
import argparse
import subprocess
import sys
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import download

TOOLS = Path(__file__).resolve().parent
MODELS_DIR = TOOLS.parent / "Models"
VENDOR = TOOLS / "vendor" / "GFPGAN"
WEIGHTS_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
SIZE = 512


def shim_torchvision():
    """basicsr 1.4.x imports torchvision.transforms.functional_tensor, removed
    in torchvision 0.17 — provide a compatibility shim."""
    import torchvision.transforms.functional as F
    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module


def ensure_repo():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/TencentARC/GFPGAN", str(VENDOR)], check=True)
    sys.path.insert(0, str(VENDOR))


class GFPGANWrapper(torch.nn.Module):
    """Input 0..1 → model domain [-1,1] → output back to 0..255."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, face):
        x = face * 2.0 - 1.0
        # randomize_noise=False → deterministic stored noise buffers
        # (random `normal_` ops are not convertible to Core ML).
        out = self.net(x, return_rgb=False, randomize_noise=False, weight=0.5)[0]
        return torch.clamp((out + 1.0) / 2.0, 0, 1) * 255.0


def main(force: bool):
    out_path = MODELS_DIR / "GFPGAN.mlpackage"
    if out_path.exists() and not force:
        print(f"{out_path.name} exists, skipping (use --force to redo)")
        return

    shim_torchvision()
    ensure_repo()
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # noqa: E402

    net = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    weights_file = download(WEIGHTS_URL, TOOLS / "weights")
    state = torch.load(weights_file, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in state else "params"
    net.load_state_dict(state[key], strict=True)
    net.eval()

    wrapped = GFPGANWrapper(net).eval()
    example = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="face", shape=(1, 3, SIZE, SIZE),
                             scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="restored", color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.short_description = "GFPGAN v1.4 face restoration, 512x512 aligned crop, FP32 (Apache-2.0)"
    MODELS_DIR.mkdir(exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")

    # Validate against PyTorch on a synthetic blurry face-ish patch.
    from PIL import Image
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    patch = np.stack([0.6 + 0.2 * np.sin(x * 6), 0.5 + 0.2 * np.sin(y * 6), 0.45 + 0.1 * x], -1)
    patch += rng.normal(0, 0.03, patch.shape).astype(np.float32)
    patch8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)

    coreml_out = np.asarray(
        mlmodel.predict({"face": Image.fromarray(patch8)})["restored"],
        dtype=np.float32)[..., :3]
    with torch.no_grad():
        tensor = torch.from_numpy(patch8.astype(np.float32) / 255).permute(2, 0, 1)[None]
        torch_out = wrapped(tensor)[0].permute(1, 2, 0).numpy()
    diff = np.abs(coreml_out - torch_out)
    print(f"restored max abs diff: {diff.max():.2f}, mean: {diff.mean():.4f} (0-255)")
    if diff.mean() > 3.0:
        sys.exit("FAILED: Core ML output deviates from PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    main(args.force)
