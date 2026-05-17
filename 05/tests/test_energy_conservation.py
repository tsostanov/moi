"""test_energy_conservation.py — energy_normalize must make sum_p g_p == sum_p f_p per object."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import bilateral_mean, bilateral_median, energy_normalize


def _make_noisy_scene(H=40, W=40, n_objects=3, seed=42):
    rng = np.random.default_rng(seed)
    obj_id = np.zeros((H, W), dtype=np.int32)
    obj_id[:, W // 3 :] = 1
    obj_id[:, 2 * W // 3 :] = 2

    normal = np.zeros((H, W, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    depth = rng.uniform(1.0, 3.0, (H, W))

    # Smooth base + noise
    base = rng.uniform(0.2, 0.8, (H, W, 3))
    noise = rng.normal(0.0, 0.15, (H, W, 3))
    image = np.clip(base + noise, 0.0, None).astype(np.float64)

    return image, obj_id, normal, depth


@pytest.mark.parametrize("mode", ["mean", "median"])
def test_energy_conservation_per_object(mode):
    img, oid, n, d = _make_noisy_scene()

    if mode == "mean":
        filtered = bilateral_mean(img, oid, n, d, sigma_s=2.0, sigma_n=0.5, sigma_z=0.5, radius=2)
    else:
        filtered = bilateral_median(img, oid, n, d, sigma_n=0.5, sigma_z=0.5, radius=2)

    normalized = energy_normalize(filtered, img, oid)

    for c_oid in range(3):
        mask = oid == c_oid
        for ch in range(3):
            e_in = float(img[mask, ch].sum())
            e_out = float(normalized[mask, ch].sum())
            assert abs(e_in - e_out) < 1e-6 * max(abs(e_in), 1.0), \
                f"obj={c_oid} ch={ch}: energy_in={e_in:.6f} energy_out={e_out:.6f}"


def test_energy_normalize_without_filter():
    """energy_normalize applied to the original should return the original."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 1.0, (20, 20, 3)).astype(np.float64)
    oid = np.zeros((20, 20), dtype=np.int32)

    result = energy_normalize(img, img, oid)
    assert np.allclose(result, img, atol=1e-10)


def test_energy_normalize_zero_output_safe():
    """If filtered values are all zero, energy_normalize must not crash."""
    img = np.ones((10, 10, 3), dtype=np.float64)
    filtered = np.zeros((10, 10, 3), dtype=np.float64)
    oid = np.zeros((10, 10), dtype=np.int32)
    # Should not raise
    result = energy_normalize(filtered, img, oid)
    assert result.shape == filtered.shape
