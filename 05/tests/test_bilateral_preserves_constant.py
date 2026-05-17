"""test_bilateral_preserves_constant.py — constant image must pass through unchanged."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import bilateral_mean, bilateral_median


def _make_constant_aov(H=20, W=20, color=(0.5, 0.3, 0.8)):
    image = np.full((H, W, 3), color, dtype=np.float64)
    obj_id = np.zeros((H, W), dtype=np.int32)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    normal[..., 2] = 1.0  # all normals point in +Z
    depth = np.full((H, W), 1.0, dtype=np.float64)
    return image, obj_id, normal, depth


@pytest.mark.parametrize("color", [(0.5, 0.3, 0.8), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
def test_mean_preserves_constant(color):
    img, oid, n, d = _make_constant_aov(color=color)
    result = bilateral_mean(img, oid, n, d, sigma_s=2.0, sigma_n=0.3, sigma_z=0.1, radius=3)
    assert np.allclose(result, img, atol=1e-9), \
        f"mean filter changed constant image: max diff = {np.abs(result - img).max()}"


@pytest.mark.parametrize("color", [(0.5, 0.3, 0.8), (0.2, 0.4, 0.6)])
def test_median_preserves_constant(color):
    img, oid, n, d = _make_constant_aov(color=color)
    result = bilateral_median(img, oid, n, d, sigma_n=0.3, sigma_z=0.1, radius=2)
    assert np.allclose(result, img, atol=1e-9), \
        f"median filter changed constant image: max diff = {np.abs(result - img).max()}"


def test_mean_non_unit_constant():
    """Constant values larger than 1 should also be preserved."""
    img, oid, n, d = _make_constant_aov(color=(5.0, 3.2, 7.1))
    result = bilateral_mean(img, oid, n, d, sigma_s=1.5, sigma_n=0.5, sigma_z=0.2, radius=2)
    assert np.allclose(result, img, atol=1e-6)
