# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "tifffile"]
# ///
"""Generate synthetic 16-bit TIFF test scans for Macolorize development:
a faded color slide, a B&W slide, and a dusty slide."""
import numpy as np
import tifffile

OUT = "/Users/tpenzkofer/Desktop/Vibeprojects/Macolorize/SampleImages"
W, H = 3000, 2000
rng = np.random.default_rng(42)


def base_scene(w, h):
    """A plausible landscape-ish scene in float [0,1] RGB."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = y / h, x / w
    # sky gradient
    r = 0.55 - 0.25 * yn
    g = 0.65 - 0.20 * yn
    b = 0.85 - 0.15 * yn
    img = np.stack([r, g, b], axis=-1)
    # ground
    ground = yn > 0.62
    img[ground] = np.stack([0.35 + 0.1 * xn[ground], 0.45 + 0.05 * xn[ground],
                            0.20 + 0.02 * xn[ground]], axis=-1)
    # sun disk
    d = np.sqrt((x - w * 0.75) ** 2 + (y - h * 0.25) ** 2)
    sun = np.clip(1 - d / 180, 0, 1)[..., None]
    img = img + sun * np.array([0.45, 0.40, 0.20])
    # dark "building" block
    img[int(h * 0.45):int(h * 0.65), int(w * 0.15):int(w * 0.35)] = [0.15, 0.12, 0.10]
    # film grain
    img += rng.normal(0, 0.012, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def to_u16(img):
    return (np.clip(img, 0, 1) * 65535).astype(np.uint16)


scene = base_scene(W, H)

# 1. Faded slide: cyan dye loss -> red cast, lifted blacks, compressed range
faded = scene.copy()
faded[..., 0] = 0.18 + faded[..., 0] * 0.78          # red lifted
faded[..., 1] = 0.10 + faded[..., 1] * 0.62          # green compressed
faded[..., 2] = 0.08 + faded[..., 2] * 0.45          # blue heavily faded
tifffile.imwrite(f"{OUT}/faded_slide.tif", to_u16(faded), photometric="rgb",
                 resolution=(2400, 2400),
                 extratags=[(272, "s", 0, "SyntheticScanner", False)])  # TIFF Model

# 2. B&W slide (grayscale luminance of the scene)
lum = scene @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
tifffile.imwrite(f"{OUT}/bw_slide.tif", to_u16(lum), photometric="minisblack",
                 resolution=(2400, 2400))

# 3. Dusty slide: scene + white dust specks and a scratch line
dusty = scene.copy()
for _ in range(120):
    cx, cy = rng.integers(0, W), rng.integers(0, H)
    rr = rng.integers(1, 5)
    yy, xx = np.ogrid[-cy:H - cy, -cx:W - cx]
    mask = xx * xx + yy * yy <= rr * rr
    dusty[mask] = 0.98
x0 = W // 2
for yy in range(int(H * 0.1), int(H * 0.9)):
    xx = x0 + int(18 * np.sin(yy / 90))
    dusty[yy, xx:xx + 3] = 0.97
tifffile.imwrite(f"{OUT}/dusty_slide.tif", to_u16(dusty), photometric="rgb",
                 resolution=(2400, 2400))

print("wrote faded_slide.tif, bw_slide.tif, dusty_slide.tif")
