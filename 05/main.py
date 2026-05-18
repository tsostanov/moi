"""main.py — Lab 5 bilateral filter CLI.

Usage examples:
    python 05/main.py --aov 05/outputs/aov.npz \\
        --mode mean --sigma-s 3 --sigma-n 0.3 --sigma-z 0.1 \\
        --split-direct-indirect \\
        --reference 05/outputs/reference.npz \\
        --output 05/outputs/filtered_mean.png

    python 05/main.py --aov 05/outputs/aov.npz \\
        --mode median --radius 3 --energy-normalize object \\
        --output 05/outputs/filtered_median.png

    python 05/main.py --aov 05/outputs/aov.npz --dump-debug
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from bilateral import bilateral_filter, make_spatial_kernel


# ---------------------------------------------------------------------------
# Image I/O helpers (standalone, no 04/ dependency needed at runtime)
# ---------------------------------------------------------------------------

GAMMA = 2.2


def _to_uint8(arr: np.ndarray, white_point: float | None = None) -> np.ndarray:
    """Convert linear float (H,W,3) → gamma-corrected uint8."""
    a = arr.astype(np.float64)
    wp = white_point if white_point and white_point > 0.0 else float(a.max())
    wp = max(wp, 1e-12)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a / wp, 0.0, 1.0)
    a = a ** (1.0 / GAMMA)
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def write_png(path: Path, rgb8: np.ndarray) -> None:
    """Write H×W×3 uint8 array as PNG (pure Python, no PIL required)."""
    H, W = rgb8.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    scanlines = bytearray()
    for row in rgb8:
        scanlines.append(0)  # filter type None
        scanlines.extend(row.tobytes())

    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6)))
        f.write(chunk(b"IEND", b""))


def write_ppm(path: Path, rgb8: np.ndarray) -> None:
    H, W = rgb8.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def save_image(path: Path, arr: np.ndarray, white_point: float | None = None) -> None:
    """Save float (H,W,3) as .png and .ppm."""
    rgb8 = _to_uint8(arr, white_point)
    png_path = path.with_suffix(".png")
    ppm_path = path.with_suffix(".ppm")
    write_png(png_path, rgb8)
    write_ppm(ppm_path, rgb8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-30:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def mean_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# ---------------------------------------------------------------------------
# AOV debug visualizations
# ---------------------------------------------------------------------------

_OBJ_PALETTE = np.array([
    [0.9, 0.9, 0.9],  # 0 floor
    [0.8, 0.8, 0.6],  # 1 ceiling
    [0.5, 0.5, 0.9],  # 2 back wall
    [0.9, 0.2, 0.2],  # 3 left wall (red)
    [0.2, 0.8, 0.2],  # 4 right wall (green)
    [1.0, 1.0, 0.0],  # 5 light
    [0.3, 0.6, 0.9],  # 6 box 1
    [0.9, 0.6, 0.3],  # 7 box 2 / pyramid
], dtype=np.float32)


def debug_obj_id(obj_id: np.ndarray) -> np.ndarray:
    H, W = obj_id.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    for oid in range(len(_OBJ_PALETTE)):
        mask = obj_id == oid
        out[mask] = _OBJ_PALETTE[oid]
    return out


def debug_normal(normal: np.ndarray) -> np.ndarray:
    return ((normal + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)


def debug_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    d = depth.copy()
    if finite.any():
        d_min = d[finite].min()
        d_max = d[finite].max()
        d_range = max(d_max - d_min, 1e-12)
        d[finite] = 1.0 - (d[finite] - d_min) / d_range
    d[~finite] = 0.0
    return np.stack([d, d, d], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 5: bilateral denoising filter")
    p.add_argument("--aov", type=Path, required=True, help="Input aov.npz")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "filtered.png",
        help="Output path (suffix replaced for .png, .ppm, .txt)",
    )
    p.add_argument("--mode", choices=("mean", "median"), default="mean")
    p.add_argument("--sigma-s", type=float, default=2.0, help="Spatial Gaussian sigma (pixels)")
    p.add_argument("--sigma-n", type=float, default=0.3, help="Normal edge-stop sigma")
    p.add_argument("--sigma-z", type=float, default=0.1, help="Depth edge-stop sigma")
    p.add_argument("--sigma-c", type=float, default=0.0, help="Color edge-stop sigma (0=off)")
    p.add_argument("--radius", type=int, default=3, help="Filter window half-size")
    p.add_argument(
        "--split-direct-indirect", action="store_true",
        help="Filter direct and indirect channels separately (recommended)",
    )
    p.add_argument(
        "--energy-normalize", choices=("none", "object"), default="none",
        help="Per-object energy normalization after filtering",
    )
    p.add_argument("--reference", type=Path, default=None, help="Reference aov.npz for PSNR/L1")
    p.add_argument("--dump-debug", action="store_true", help="Save AOV visualization images")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.aov.exists():
        raise SystemExit(f"error: AOV file not found: {args.aov}")

    aov = dict(np.load(str(args.aov)))
    required = {"direct", "indirect", "depth", "obj_id", "normal", "color"}
    missing = required - set(aov.keys())
    if missing:
        raise SystemExit(f"error: AOV file missing keys: {missing}")

    H, W = aov["color"].shape[:2]
    print(f"loaded AOV {W}x{H} from {args.aov}", flush=True)

    # Apply filter
    print(
        f"filtering mode={args.mode} sigma_s={args.sigma_s} sigma_n={args.sigma_n} "
        f"sigma_z={args.sigma_z} radius={args.radius} "
        f"split={args.split_direct_indirect} energy={args.energy_normalize}",
        flush=True,
    )
    filtered = bilateral_filter(
        aov,
        mode=args.mode,
        sigma_s=args.sigma_s,
        sigma_n=args.sigma_n,
        sigma_z=args.sigma_z,
        sigma_c=args.sigma_c,
        radius=args.radius,
        split_direct_indirect=args.split_direct_indirect,
        energy_normalize_mode=args.energy_normalize,
    )

    # Save outputs
    out_base = args.output.with_suffix("")
    save_image(out_base.with_suffix(".png"), filtered)
    print(f"saved {out_base.with_suffix('.png')}", flush=True)
    print(f"saved {out_base.with_suffix('.ppm')}", flush=True)

    # Metrics
    noisy_color = aov["color"].astype(np.float64)
    filtered_f64 = filtered.astype(np.float64)

    ref_psnr_noisy: float | None = None
    ref_l1_noisy: float | None = None
    ref_psnr_filtered: float | None = None
    ref_l1_filtered: float | None = None
    ref_color = None

    if args.reference is not None:
        if not args.reference.exists():
            print(f"warning: reference file not found: {args.reference}", flush=True)
        else:
            ref = dict(np.load(str(args.reference)))
            ref_color = ref["color"].astype(np.float64)
            max_ref = max(float(ref_color.max()), 1e-12)

            ref_psnr_noisy = psnr(noisy_color, ref_color, max_ref)
            ref_l1_noisy = mean_l1(noisy_color, ref_color)
            ref_psnr_filtered = psnr(filtered_f64, ref_color, max_ref)
            ref_l1_filtered = mean_l1(filtered_f64, ref_color)

            print(f"PSNR  noisy->ref : {ref_psnr_noisy:.2f} dB", flush=True)
            print(f"PSNR  filt->ref  : {ref_psnr_filtered:.2f} dB", flush=True)
            print(f"L1    noisy->ref : {ref_l1_noisy:.6f}", flush=True)
            print(f"L1    filt->ref  : {ref_l1_filtered:.6f}", flush=True)

    # Energy report per object
    obj_id = aov["obj_id"].astype(np.int32)
    energy_lines: list[str] = []
    for oid in sorted(np.unique(obj_id)):
        if oid < 0:
            continue
        mask = obj_id == oid
        e_in = float(noisy_color[mask].sum())
        e_out = float(filtered_f64[mask].sum())
        energy_lines.append(f"  obj_id={oid}  energy_in={e_in:.4f}  energy_out={e_out:.4f}  ratio={e_out/(e_in+1e-30):.4f}")

    # Write stats file
    txt_path = out_base.with_suffix(".txt")
    lines = [
        "Lab 5 bilateral filter run",
        f"input_aov: {args.aov}",
        f"output: {out_base.with_suffix('.png')}",
        f"mode: {args.mode}",
        f"sigma_s: {args.sigma_s}",
        f"sigma_n: {args.sigma_n}",
        f"sigma_z: {args.sigma_z}",
        f"sigma_c: {args.sigma_c}",
        f"radius: {args.radius}",
        f"split_direct_indirect: {args.split_direct_indirect}",
        f"energy_normalize: {args.energy_normalize}",
        f"image_size: {W}x{H}",
        "",
    ]
    if ref_psnr_filtered is not None:
        lines += [
            f"reference: {args.reference}",
            f"psnr_noisy_db: {ref_psnr_noisy:.2f}",
            f"psnr_filtered_db: {ref_psnr_filtered:.2f}",
            f"l1_noisy: {ref_l1_noisy:.6f}",
            f"l1_filtered: {ref_l1_filtered:.6f}",
            "",
        ]
    lines += ["Energy per object (noisy vs filtered):", *energy_lines, ""]

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {txt_path}", flush=True)

    # Debug visualizations
    if args.dump_debug:
        out_dir = args.output.parent
        debug_img = debug_obj_id(obj_id)
        save_image(out_dir / "debug_objid.png", debug_img, white_point=1.0)
        print(f"saved {out_dir / 'debug_objid.png'}")

        debug_img = debug_normal(aov["normal"])
        save_image(out_dir / "debug_normal.png", debug_img, white_point=1.0)
        print(f"saved {out_dir / 'debug_normal.png'}")

        debug_img = debug_depth(aov["depth"])
        save_image(out_dir / "debug_depth.png", debug_img, white_point=1.0)
        print(f"saved {out_dir / 'debug_depth.png'}")

        # Noisy preview
        save_image(out_dir / "noisy.png", noisy_color, white_point=float(noisy_color.max()))
        print(f"saved {out_dir / 'noisy.png'}")

        # Diff image
        diff = np.abs(noisy_color - filtered_f64) * 5.0
        save_image(out_dir / "diff.png", diff, white_point=float(diff.max()) + 1e-12)
        print(f"saved {out_dir / 'diff.png'}")


if __name__ == "__main__":
    main()
