"""Convert DDColor (piddnad/DDColor, Apache-2.0) to Core ML.

Produces ../Models/DDColor.mlpackage:
    input  "image": 512x512 RGB (the L channel rendered as gray RGB, 0-255)
    output "ab":    MLMultiArray (1, 2, 512, 512), a/b chroma in Lab units

The official repo is cloned into tools/vendor for its model definition.
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
VENDOR = TOOLS / "vendor" / "DDColor"

VARIANTS = {
    "DDColor": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
    "DDColor_artistic": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_artistic.pth",
}


def ensure_repo():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/piddnad/DDColor", str(VENDOR)], check=True)
    sys.path.insert(0, str(VENDOR))


def main(force: bool):
    for name, url in VARIANTS.items():
        convert_variant(name, url, force)


def convert_variant(name: str, weights_url: str, force: bool):
    out_path = MODELS_DIR / f"{name}.mlpackage"
    if out_path.exists() and not force:
        print(f"{out_path.name} exists, skipping (use --force to redo)")
        return

    ensure_repo()
    from basicsr.archs.ddcolor_arch import DDColor  # noqa: E402

    net = DDColor(
        encoder_name="convnext-l",
        decoder_name="MultiScaleColorDecoder",
        input_size=[512, 512],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )
    weights_file = download(weights_url, TOOLS / "weights")
    state = torch.load(weights_file, map_location="cpu", weights_only=False)
    if "params" in state:
        state = state["params"]
    # The modelscope checkpoint ships an unused classification head.
    state = {k: v for k, v in state.items() if not k.startswith("encoder.arch.head_cls.")}
    net.load_state_dict(state, strict=True)
    net.eval()

    example = torch.rand(1, 3, 512, 512)
    traced = torch.jit.trace(net, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 512, 512),
                             scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="ab")],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.short_description = f"{name}: predicts Lab ab chroma from a gray 512x512 image"
    MODELS_DIR.mkdir(exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")

    # Validate: same gray input through PyTorch and Core ML.
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:512, 0:512].astype(np.float32) / 512
    gray = np.clip(0.2 + 0.6 * x + 0.15 * np.sin(y * 20), 0, 1)
    gray_rgb = (np.stack([gray] * 3, axis=-1) * 255).astype(np.uint8)

    from PIL import Image
    coreml_ab = mlmodel.predict({"image": Image.fromarray(gray_rgb)})["ab"]
    with torch.no_grad():
        tensor = torch.from_numpy(gray_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        torch_ab = net(tensor).numpy()

    diff = np.abs(coreml_ab.astype(np.float32) - torch_ab)
    print(f"ab max abs diff: {diff.max():.3f}, mean: {diff.mean():.4f} (Lab units)")
    if diff.mean() > 1.0:
        sys.exit("FAILED: mean ab difference exceeds 1 Lab unit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)
