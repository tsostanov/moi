"""test_normal_edge_preservation.py — normal edge-stop must suppress cross-plane blending."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bilateral import bilateral_mean


def _make_two_normal_scene(H=30, W=30, sigma_n=0.05):
    """
    Same object (id=0), two halves with orthogonal normals.
    Left: normal=(0,0,1), color=(1,0,0)
    Right: normal=(1,0,0), color=(0,0,1)
    With small sigma_n, cross-plane blending must be suppressed.
    """
    image = np.zeros((H, W, 3), dtype=np.float64)
    obj_id = np.zeros((H, W), dtype=np.int32)  # same object!
    normal = np.zeros((H, W, 3), dtype=np.float64)
    depth = np.ones((H, W), dtype=np.float64)

    mid = W // 2
    image[:, :mid] = [1.0, 0.0, 0.0]
    image[:, mid:] = [0.0, 0.0, 1.0]
    normal[:, :mid] = [0.0, 0.0, 1.0]   # +Z
    normal[:, mid:] = [1.0, 0.0, 0.0]   # +X  (orthogonal → dot=0)

    return image, obj_id, normal, depth


def test_mean_suppresses_orthogonal_normal_blending():
    """With very small sigma_n, normals at 90° should produce near-zero weight."""
    img, oid, n, d = _make_two_normal_scene()
    # sigma_n=0.05 → exp(-(1 - 0)/0.05^2) = exp(-400) ≈ 0 for orthogonal normals
    result = bilateral_mean(img, oid, n, d,
                            sigma_s=2.0, sigma_n=0.05, sigma_z=10.0, radius=4)
    # Interior pixels on the left should remain red (no blue leaked in)
    left_interior = result[15, 4]
    assert left_interior[2] < 1e-5, \
        f"Left got blue from orthogonal normal region: {left_interior[2]:.8f}"


def test_mean_allows_same_normal_blending():
    """With same normal across the image, mean filter should blend smoothly."""
    H, W = 30, 30
    # Step edge, same obj, same normal
    image = np.zeros((H, W, 3), dtype=np.float64)
    image[:, :W//2] = [1.0, 0.0, 0.0]
    image[:, W//2:] = [0.0, 0.0, 1.0]
    obj_id = np.zeros((H, W), dtype=np.int32)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    normal[..., 2] = 1.0  # all same normal
    depth = np.ones((H, W), dtype=np.float64)

    result = bilateral_mean(image, obj_id, normal, depth,
                            sigma_s=2.0, sigma_n=10.0, sigma_z=10.0, radius=3)

    # Border pixel should receive contributions from both sides → mixed color
    border = result[15, W//2 - 1]
    # Not purely red anymore
    assert border[2] > 0.01, "Expected some blue contribution at border from same-normal blending"
