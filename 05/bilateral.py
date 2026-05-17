"""bilateral.py — bilateral denoising filter with AOV-guided edge stopping.

Implements:
  - make_spatial_kernel   : normalized discrete Gaussian kernel
  - bilateral_mean        : mean variant (vectorized over shift offsets)
  - bilateral_median      : median variant (vectorized over shift offsets)
  - energy_normalize      : per-object energy conservation
  - bilateral_filter      : top-level dispatcher

Formula (from lecture, slide 45):
    g_p = (1/W_p) * sum_{q in S} f_q * G_s(p-q) * G_r(p,q)
where:
    G_s — spatial Gaussian (pre-normalized so sum_q G_s = 1, hence W_p = sum_q G_r)
    G_r = w_obj * w_normal * w_depth  (range/edge-stop weights)
"""
from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Spatial kernel
# ---------------------------------------------------------------------------

def make_spatial_kernel(sigma_s: float, radius: int) -> np.ndarray:
    """Return a (2r+1, 2r+1) float64 Gaussian kernel normalized to sum=1."""
    size = 2 * radius + 1
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(y.astype(np.float64) ** 2 + x.astype(np.float64) ** 2) / (2.0 * sigma_s ** 2))
    kernel /= kernel.sum()
    return kernel


# ---------------------------------------------------------------------------
# Mean bilateral filter
# ---------------------------------------------------------------------------

def bilateral_mean(
    image: np.ndarray,
    obj_id: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    sigma_s: float = 2.0,
    sigma_n: float = 0.3,
    sigma_z: float = 0.1,
    sigma_c: float = 0.0,
    radius: int = 3,
) -> np.ndarray:
    """
    Mean bilateral filter on a single RGB image.

    Args:
        image   : (H, W, 3) float array
        obj_id  : (H, W) int array, -1 = background
        normal  : (H, W, 3) float array of unit surface normals
        depth   : (H, W) float array of distances
        sigma_s : spatial Gaussian sigma (pixels)
        sigma_n : normal edge-stop sigma
        sigma_z : depth edge-stop sigma
        sigma_c : color edge-stop sigma (0 = disabled)
        radius  : filter window half-size
    Returns:
        filtered image (H, W, 3)
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape[:2]
    kernel_s = make_spatial_kernel(sigma_s, radius)

    pad = radius
    img_p = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    oid_p = np.pad(obj_id, pad, mode="edge")
    nrm_p = np.pad(normal, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    dep_p = np.pad(depth, pad, mode="edge")

    numerator = np.zeros((H, W, 3), dtype=np.float64)
    denominator = np.zeros((H, W), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            qs = kernel_s[dy + radius, dx + radius]
            if qs == 0.0:
                continue

            sy = slice(pad + dy, pad + dy + H)
            sx = slice(pad + dx, pad + dx + W)

            f_q = img_p[sy, sx]
            oid_q = oid_p[sy, sx]
            n_q = nrm_p[sy, sx]
            d_q = dep_p[sy, sx]

            # Object edge-stop: hard mask for different objects
            w_obj = (obj_id == oid_q).astype(np.float64)

            # Normal edge-stop
            cos_sim = np.clip(np.sum(normal * n_q, axis=-1), 0.0, 1.0)
            w_norm = np.exp(-(1.0 - cos_sim) / (sigma_n ** 2))

            # Depth edge-stop (inf depth = background → zero weight)
            depth_safe = np.where(np.isfinite(depth), depth, 1e30)
            d_q_safe = np.where(np.isfinite(d_q), d_q, 1e30)
            dz = depth_safe - d_q_safe
            w_dep = np.exp(-(dz ** 2) / (2.0 * sigma_z ** 2))

            G_r = w_obj * w_norm * w_dep

            # Optional color edge-stop
            if sigma_c > 0.0:
                diff_c = np.sum((image - f_q) ** 2, axis=-1)
                G_r = G_r * np.exp(-diff_c / (2.0 * sigma_c ** 2))

            weight = qs * G_r  # (H, W)
            numerator += f_q * weight[:, :, np.newaxis]
            denominator += weight

    denom_safe = np.maximum(denominator, 1e-30)
    return (numerator / denom_safe[:, :, np.newaxis]).astype(image.dtype)


# ---------------------------------------------------------------------------
# Median bilateral filter (vectorized)
# ---------------------------------------------------------------------------

def bilateral_median(
    image: np.ndarray,
    obj_id: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    sigma_n: float = 0.3,
    sigma_z: float = 0.1,
    radius: int = 3,
    weight_threshold: float = 0.05,
) -> np.ndarray:
    """
    Median bilateral filter.

    For each pixel p, collects the set of neighbors q in window S that:
      1. Belong to the same object (same obj_id)
      2. Have sufficient normal and depth similarity

    Returns the component-wise median of valid neighbors.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape[:2]
    window = 2 * radius + 1
    n_offsets = window * window

    pad = radius
    img_p = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    oid_p = np.pad(obj_id, pad, mode="edge")
    nrm_p = np.pad(normal, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    dep_p = np.pad(depth, pad, mode="edge")

    # Collect neighbor values with NaN for excluded pixels
    neighbors = np.full((n_offsets, H, W, 3), np.nan, dtype=np.float64)

    k = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sy = slice(pad + dy, pad + dy + H)
            sx = slice(pad + dx, pad + dx + W)

            f_q = img_p[sy, sx]
            oid_q = oid_p[sy, sx]
            n_q = nrm_p[sy, sx]
            d_q = dep_p[sy, sx]

            valid_obj = obj_id == oid_q

            cos_sim = np.clip(np.sum(normal * n_q, axis=-1), 0.0, 1.0)
            w_n = np.exp(-(1.0 - cos_sim) / (sigma_n ** 2))

            depth_safe = np.where(np.isfinite(depth), depth, 1e30)
            d_q_safe = np.where(np.isfinite(d_q), d_q, 1e30)
            dz = depth_safe - d_q_safe
            w_d = np.exp(-(dz ** 2) / (2.0 * sigma_z ** 2))

            w = valid_obj.astype(np.float64) * w_n * w_d
            include = w > weight_threshold  # (H, W)

            vals = f_q.copy()
            vals[~include] = np.nan
            neighbors[k] = vals
            k += 1

    # nanmedian over neighbor dimension; background pixels (all-NaN) fall back to original
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        result = np.nanmedian(neighbors, axis=0)
    all_nan = np.all(np.isnan(neighbors), axis=0).any(axis=-1)
    result[all_nan] = image[all_nan]

    return result.astype(image.dtype)


# ---------------------------------------------------------------------------
# Energy normalization (per object)
# ---------------------------------------------------------------------------

def energy_normalize(
    filtered: np.ndarray,
    original: np.ndarray,
    obj_ids: np.ndarray,
) -> np.ndarray:
    """
    Scale filtered values so that for each object O:
        sum_{p in O} g_p  ==  sum_{p in O} f_p  (per channel)

    This is the energy-conservation requirement from slide 45.
    """
    result = filtered.copy()
    for oid in np.unique(obj_ids):
        if oid < 0:
            continue
        mask = obj_ids == oid
        for c in range(filtered.shape[2]):
            s_in = float(original[mask, c].sum())
            s_out = float(filtered[mask, c].sum())
            if s_out > 1e-30:
                result[mask, c] = result[mask, c] * (s_in / s_out)
    return result


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def bilateral_filter(
    aov: dict,
    mode: str = "mean",
    sigma_s: float = 2.0,
    sigma_n: float = 0.3,
    sigma_z: float = 0.1,
    sigma_c: float = 0.0,
    radius: int = 3,
    split_direct_indirect: bool = True,
    energy_normalize_mode: str = "none",
) -> np.ndarray:
    """
    Apply bilateral filter to AOV data.

    Args:
        aov   : dict with keys 'direct', 'indirect', 'depth', 'obj_id', 'normal'
        mode  : 'mean' or 'median'
        split_direct_indirect: filter direct and indirect separately then sum
        energy_normalize_mode: 'none' or 'object'

    Returns:
        Filtered color image (H, W, 3).
    """
    obj_id = aov["obj_id"].astype(np.int32)
    normal = aov["normal"].astype(np.float64)
    depth = aov["depth"].astype(np.float64)

    def _filter(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float64)
        if mode == "mean":
            return bilateral_mean(img, obj_id, normal, depth, sigma_s, sigma_n, sigma_z, sigma_c, radius)
        elif mode == "median":
            return bilateral_median(img, obj_id, normal, depth, sigma_n, sigma_z, radius)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    if split_direct_indirect:
        direct = aov["direct"].astype(np.float64)
        indirect = aov["indirect"].astype(np.float64)
        f_direct = _filter(direct)
        f_indirect = _filter(indirect)

        if energy_normalize_mode == "object":
            f_direct = energy_normalize(f_direct, direct, obj_id)
            f_indirect = energy_normalize(f_indirect, indirect, obj_id)

        return np.clip(f_direct + f_indirect, 0.0, None).astype(np.float32)
    else:
        color = aov["color"].astype(np.float64)
        filtered = _filter(color)

        if energy_normalize_mode == "object":
            filtered = energy_normalize(filtered, color, obj_id)

        return np.clip(filtered, 0.0, None).astype(np.float32)
