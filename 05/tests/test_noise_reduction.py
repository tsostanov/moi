"""test_noise_reduction.py — bilateral filter must reduce L1 error on smooth+noise signal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import bilateral_mean, bilateral_median


def _make_smooth_noisy(H=50, W=50, noise_sigma=0.2, seed=7):
    rng = np.random.default_rng(seed)

    # Smooth ground truth: slow sinusoidal variation
    y_idx = np.linspace(0, np.pi, H)[:, None] * np.ones((1, W))
    x_idx = np.linspace(0, np.pi, W)[None, :] * np.ones((H, 1))
    ground_truth = (0.5 + 0.3 * np.sin(y_idx) * np.cos(x_idx))
    ground_truth = np.stack([ground_truth] * 3, axis=-1)

    noise = rng.normal(0.0, noise_sigma, ground_truth.shape)
    noisy = np.clip(ground_truth + noise, 0.0, None)

    obj_id = np.zeros((H, W), dtype=np.int32)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    depth = np.ones((H, W), dtype=np.float64)

    return ground_truth, noisy, obj_id, normal, depth


def l1(a, b):
    return float(np.mean(np.abs(a - b)))


def test_mean_reduces_l1_error():
    gt, noisy, oid, n, d = _make_smooth_noisy()
    filtered = bilateral_mean(noisy, oid, n, d, sigma_s=2.5, sigma_n=10.0, sigma_z=10.0, radius=3)

    l1_noisy = l1(noisy, gt)
    l1_filtered = l1(filtered, gt)

    assert l1_filtered < l1_noisy, (
        f"Mean filter did not reduce L1 error: "
        f"noisy={l1_noisy:.4f}  filtered={l1_filtered:.4f}"
    )


def test_median_reduces_l1_error():
    gt, noisy, oid, n, d = _make_smooth_noisy()
    filtered = bilateral_median(noisy, oid, n, d, sigma_n=10.0, sigma_z=10.0, radius=2)

    l1_noisy = l1(noisy, gt)
    l1_filtered = l1(filtered, gt)

    assert l1_filtered < l1_noisy, (
        f"Median filter did not reduce L1 error: "
        f"noisy={l1_noisy:.4f}  filtered={l1_filtered:.4f}"
    )


def test_split_direct_indirect_helps():
    """Filtering direct+indirect separately should match or beat combined."""
    from bilateral import bilateral_filter
    gt, noisy, oid, n, d = _make_smooth_noisy()

    aov_combined = {
        "direct": noisy,
        "indirect": np.zeros_like(noisy),
        "color": noisy,
        "obj_id": oid,
        "normal": n,
        "depth": d,
    }

    f_no_split = bilateral_filter(aov_combined, mode="mean", sigma_s=2.5, sigma_n=10.0,
                                   sigma_z=10.0, radius=3, split_direct_indirect=False)
    f_split = bilateral_filter(aov_combined, mode="mean", sigma_s=2.5, sigma_n=10.0,
                                sigma_z=10.0, radius=3, split_direct_indirect=True)

    # Both should reduce noise; split variant result should be close to non-split
    # (here indirect=0, so they should be identical)
    assert np.allclose(f_no_split, f_split, atol=1e-5)
