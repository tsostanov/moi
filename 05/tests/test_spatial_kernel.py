"""test_spatial_kernel.py — unit tests for make_spatial_kernel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import make_spatial_kernel


@pytest.mark.parametrize("sigma_s,radius", [(1.0, 2), (2.0, 3), (0.5, 1)])
def test_kernel_sums_to_one(sigma_s, radius):
    k = make_spatial_kernel(sigma_s, radius)
    assert abs(k.sum() - 1.0) < 1e-12, f"kernel sum = {k.sum()}"


@pytest.mark.parametrize("sigma_s,radius", [(1.0, 2), (2.0, 4)])
def test_kernel_is_symmetric(sigma_s, radius):
    k = make_spatial_kernel(sigma_s, radius)
    assert np.allclose(k, k.T), "kernel must be symmetric (k == k.T)"
    assert np.allclose(k, k[::-1, ::-1]), "kernel must be symmetric under 180° rotation"


@pytest.mark.parametrize("sigma_s,radius", [(1.0, 2), (2.0, 3)])
def test_center_is_maximum(sigma_s, radius):
    k = make_spatial_kernel(sigma_s, radius)
    center = k[radius, radius]
    assert center == k.max(), "center element must be the maximum"


def test_kernel_shape():
    k = make_spatial_kernel(1.5, 3)
    assert k.shape == (7, 7)


def test_all_positive():
    k = make_spatial_kernel(1.0, 2)
    assert np.all(k > 0), "all kernel values must be positive"
