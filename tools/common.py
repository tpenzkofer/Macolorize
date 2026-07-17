"""Shared helpers for the conversion scripts."""
import math
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.split("/")[-1]
    if dest.exists():
        return dest
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def test_image(size: int) -> np.ndarray:
    """Deterministic gradient + structure test patch, uint8 HWC RGB."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([x, y, (x + y) / 2], axis=-1)
    img += 0.15 * np.sin(x[..., None] * 40) * np.cos(y[..., None] * 40)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def validate_psnr(mlmodel, torch_model, tile: int, scale: int,
                  input_name: str = "image", output_name: str = "upscaled") -> float:
    """Runs the same test patch through Core ML and PyTorch, returns PSNR."""
    patch = test_image(tile)
    pil = Image.fromarray(patch)

    coreml_out = mlmodel.predict({input_name: pil})[output_name]
    coreml_arr = np.asarray(coreml_out, dtype=np.float32)
    if coreml_arr.shape[-1] == 4:  # RGBA image output
        coreml_arr = coreml_arr[..., :3]

    with torch.no_grad():
        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        torch_out = torch_model(tensor)[0].permute(1, 2, 0).numpy()

    mse = float(np.mean((coreml_arr - torch_out) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))
