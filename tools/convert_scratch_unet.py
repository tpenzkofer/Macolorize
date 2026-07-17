"""Convert the scratch-detection U-Net from Microsoft's Bringing Old Photos
Back to Life (MIT) to Core ML. Weights via the zeroscratches mirror on HF.

Produces ../Models/ScratchNet.mlpackage:
    input  "image": 1024x1024 grayscale
    output "mask":  1024x1024 grayscale scratch probability (0-255)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
MODELS_DIR = TOOLS.parent / "Models"
VENDOR = TOOLS / "vendor" / "BOPBTL"
SIZE = 1024


def ensure_repo():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life",
                        str(VENDOR)], check=True)
    sys.path.insert(0, str(VENDOR / "Global"))


def fetch_weights() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("leonelhs/zeroscratches", "detection/FT_Epoch_latest.pt"))


class ScratchWrapper(torch.nn.Module):
    """Normalizes ((x-0.5)/0.5) like the official transform, sigmoids the
    logits, and scales to 0-255 for an image output."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        x = (image - 0.5) / 0.5
        logits = self.net(x)
        return torch.sigmoid(logits) * 255.0


def main(force: bool):
    out_path = MODELS_DIR / "ScratchNet.mlpackage"
    if out_path.exists() and not force:
        print(f"{out_path.name} exists, skipping (use --force to redo)")
        return

    ensure_repo()
    # The repo expects a Synchronized-BatchNorm-PyTorch submodule; at
    # inference plain BatchNorm2d is equivalent — shim the import.
    import types
    import torch.nn as nn
    shim = types.ModuleType("detection_models.sync_batchnorm")
    shim.DataParallelWithCallback = nn.DataParallel
    shim.SynchronizedBatchNorm2d = nn.BatchNorm2d
    sys.modules["detection_models.sync_batchnorm"] = shim

    from detection_models import networks  # noqa: E402

    net = networks.UNet(
        in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6,
        padding=True, batch_norm=True, up_mode="upsample",
        with_tanh=False, sync_bn=False, antialiasing=True)
    weights_file = fetch_weights()
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()

    wrapped = ScratchWrapper(net).eval()
    example = torch.rand(1, 1, SIZE, SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 1, SIZE, SIZE),
                             scale=1 / 255.0, color_layout=ct.colorlayout.GRAYSCALE)],
        outputs=[ct.ImageType(name="mask", color_layout=ct.colorlayout.GRAYSCALE)],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.short_description = "BOPBTL scratch detection U-Net, 1024x1024 grayscale (MIT)"
    MODELS_DIR.mkdir(exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")

    # Validate vs PyTorch on a synthetic scratched patch.
    from PIL import Image
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    patch = 0.35 + 0.3 * x
    for cy in range(100, 900):
        cx = 512 + int(30 * np.sin(cy / 60))
        patch[cy, cx - 2:cx + 2] = 0.95
    patch8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)

    coreml_out = np.asarray(mlmodel.predict(
        {"image": Image.fromarray(patch8, mode="L")})["mask"], dtype=np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(patch8.astype(np.float32) / 255)[None, None]
        torch_out = wrapped(tensor)[0, 0].numpy()
    diff = np.abs(coreml_out - torch_out)
    on_scratch = coreml_out[500, 480:544].max()
    print(f"CoreML-vs-torch mean diff: {diff.mean():.3f}/255; "
          f"scratch response: {on_scratch:.0f}/255")
    if diff.mean() > 4.0:
        sys.exit("FAILED: Core ML output deviates from PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)
