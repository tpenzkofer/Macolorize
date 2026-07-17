#!/bin/zsh
# Downloads weights and converts all Macolorize models to Core ML (.mlpackage in ../Models).
# Dust detection is classical (no pretrained dust model is publicly available) — no script needed.
set -e
cd "$(dirname "$0")"
uv sync
uv run python convert_realesrgan.py "$@"
uv run python convert_ddcolor.py "$@"
uv run python convert_lama.py "$@"
uv run python convert_gfpgan.py "$@"
uv run python convert_drunet.py "$@"
uv run python convert_scratch_unet.py "$@"
