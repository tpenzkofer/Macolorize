"""Convert big-lama inpainting to Core ML (adapted from mallman/CoreMLaMa,
Apache-2.0, simplified to load the TorchScript checkpoint directly).

Produces ../Models/LaMa.mlpackage:
    input  "image": 800x800 RGB
    input  "mask":  800x800 grayscale (>0 = inpaint here)
    output "output": 800x800 RGB inpainted

FP32 + CPU_AND_GPU: LaMa's Fourier convolutions don't run on the ANE.
"""
import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from common import download

SIZE = 800
MODELS_DIR = Path(__file__).resolve().parent.parent / "Models"
WEIGHTS_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


class CoreMLaMa(torch.nn.Module):
    def __init__(self, lama):
        super().__init__()
        self.lama = lama

    def forward(self, image, mask):
        normalized_mask = (mask > 0).to(torch.float32)
        lama_out = self.lama(image, normalized_mask)
        return torch.clamp(lama_out * 255, min=0, max=255)


def main(force: bool):
    out_path = MODELS_DIR / "LaMa.mlpackage"
    if out_path.exists() and not force:
        print(f"{out_path.name} exists, skipping (use --force to redo)")
        return

    weights_file = download(WEIGHTS_URL, Path(__file__).parent / "weights")
    lama = torch.jit.load(weights_file, map_location="cpu").eval()
    model = CoreMLaMa(lama).eval()

    example_image = torch.rand(1, 3, SIZE, SIZE)
    example_mask = (torch.rand(1, 1, SIZE, SIZE) > 0.9).float()
    traced = torch.jit.trace(model, (example_image, example_mask))

    coreml_model = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        inputs=[
            ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255.0),
            ct.ImageType(name="mask", shape=(1, 1, SIZE, SIZE),
                         color_layout=ct.colorlayout.GRAYSCALE),
        ],
        outputs=[ct.ImageType(name="output")],
        minimum_deployment_target=ct.target.macOS15,
    )
    coreml_model.short_description = "big-lama inpainting, 800x800 image+mask (Apache-2.0)"
    MODELS_DIR.mkdir(exist_ok=True)
    coreml_model.save(str(out_path))
    print(f"saved {out_path}")

    # Validate: inpaint a synthetic defect, compare Core ML vs PyTorch.
    from PIL import Image
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    scene = np.stack([x, 0.5 + 0.3 * y, 1 - x], axis=-1)
    scene8 = (np.clip(scene, 0, 1) * 255).astype(np.uint8)
    mask8 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask8[380:420, 300:500] = 255

    coreml_out = coreml_model.predict({
        "image": Image.fromarray(scene8),
        "mask": Image.fromarray(mask8, mode="L"),
    })["output"]
    coreml_arr = np.asarray(coreml_out, dtype=np.float32)[..., :3]

    with torch.no_grad():
        t_img = torch.from_numpy(scene8.astype(np.float32) / 255).permute(2, 0, 1)[None]
        t_mask = torch.from_numpy((mask8 > 0).astype(np.float32))[None, None]
        torch_out = model(t_img, t_mask)[0].permute(1, 2, 0).numpy()

    diff = np.abs(coreml_arr - torch_out)
    print(f"output max abs diff: {diff.max():.2f}, mean: {diff.mean():.4f} (0-255 scale)")
    if diff.mean() > 2.0:
        sys.exit("FAILED: Core ML output deviates from PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)
