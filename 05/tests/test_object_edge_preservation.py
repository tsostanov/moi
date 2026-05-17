"""test_object_edge_preservation.py — obj_id hard mask must prevent cross-object blending."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import bilateral_mean, bilateral_median


def _make_two_object_scene(H=30, W=30):
    """Left half: obj_id=0 color=(1,0,0); right half: obj_id=1 color=(0,0,1)."""
    image = np.zeros((H, W, 3), dtype=np.float64)
    obj_id = np.zeros((H, W), dtype=np.int32)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    depth = np.ones((H, W), dtype=np.float64)

    mid = W // 2
    image[:, :mid] = [1.0, 0.0, 0.0]  # red
    image[:, mid:] = [0.0, 0.0, 1.0]  # blue
    obj_id[:, :mid] = 0
    obj_id[:, mid:] = 1

    return image, obj_id, normal, depth


def test_mean_no_cross_object_blend():
    img, oid, n, d = _make_two_object_scene(H=30, W=30)
    result = bilateral_mean(img, oid, n, d, sigma_s=3.0, sigma_n=0.5, sigma_z=0.5, radius=4)

    mid = 15
    # Interior pixels far from the border must remain purely red or blue
    left_interior = result[15, 4]    # well inside left half
    right_interior = result[15, 25]  # well inside right half

    assert left_interior[2] < 1e-9, \
        f"Left object got blue contribution: {left_interior[2]:.6f}"
    assert right_interior[0] < 1e-9, \
        f"Right object got red contribution: {right_interior[0]:.6f}"


def test_median_no_cross_object_blend():
    img, oid, n, d = _make_two_object_scene(H=30, W=30)
    result = bilateral_median(img, oid, n, d, sigma_n=0.5, sigma_z=0.5, radius=3)

    left_interior = result[15, 4]
    right_interior = result[15, 25]

    assert left_interior[2] < 1e-9, \
        f"Left object got blue contribution: {left_interior[2]:.6f}"
    assert right_interior[0] < 1e-9, \
        f"Right object got red contribution: {right_interior[0]:.6f}"


def test_mean_border_pixels_unchanged_on_own_side():
    """Pixels at the boundary should still be purely their own object's color."""
    img, oid, n, d = _make_two_object_scene(H=30, W=30)
    result = bilateral_mean(img, oid, n, d, sigma_s=2.0, sigma_n=0.5, sigma_z=0.5, radius=3)

    mid = 15
    # Pixel just left of boundary
    p = result[15, mid - 1]
    assert p[2] < 1e-9, f"Boundary pixel on left side has blue: {p[2]}"
    # Pixel just right of boundary
    p = result[15, mid]
    assert p[0] < 1e-9, f"Boundary pixel on right side has red: {p[0]}"
