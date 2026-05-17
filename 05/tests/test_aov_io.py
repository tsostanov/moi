"""test_aov_io.py — round-trip write/read of aov.npz preserves all keys, dtypes, shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


def _make_synthetic_aov(H=10, W=10):
    rng = np.random.default_rng(1)
    direct = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
    indirect = rng.uniform(0, 0.5, (H, W, 3)).astype(np.float32)
    depth = rng.uniform(1, 5, (H, W)).astype(np.float32)
    obj_id = rng.integers(0, 4, (H, W)).astype(np.int32)
    normal = rng.standard_normal((H, W, 3)).astype(np.float32)
    # Normalize normals
    norms = np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6, None)
    normal = (normal / norms).astype(np.float32)
    color = (direct + indirect).astype(np.float32)
    return dict(direct=direct, indirect=indirect, depth=depth,
                obj_id=obj_id, normal=normal, color=color)


def test_roundtrip_all_keys_present(tmp_path):
    aov = _make_synthetic_aov()
    out_path = tmp_path / "aov.npz"
    np.savez_compressed(str(out_path), **aov)

    loaded = dict(np.load(str(out_path)))
    for key in ("direct", "indirect", "depth", "obj_id", "normal", "color"):
        assert key in loaded, f"key {key!r} missing after round-trip"


def test_roundtrip_shapes_preserved(tmp_path):
    aov = _make_synthetic_aov(H=12, W=15)
    out_path = tmp_path / "aov_shapes.npz"
    np.savez_compressed(str(out_path), **aov)

    loaded = dict(np.load(str(out_path)))
    for key in ("direct", "indirect", "normal", "color"):
        assert loaded[key].shape == (12, 15, 3), f"{key} shape mismatch: {loaded[key].shape}"
    assert loaded["depth"].shape == (12, 15)
    assert loaded["obj_id"].shape == (12, 15)


def test_roundtrip_dtypes_preserved(tmp_path):
    aov = _make_synthetic_aov()
    out_path = tmp_path / "aov_dtypes.npz"
    np.savez_compressed(str(out_path), **aov)

    loaded = dict(np.load(str(out_path)))
    for key in ("direct", "indirect", "depth", "normal", "color"):
        assert loaded[key].dtype == np.float32, f"{key} dtype mismatch: {loaded[key].dtype}"
    assert loaded["obj_id"].dtype == np.int32, f"obj_id dtype: {loaded['obj_id'].dtype}"


def test_roundtrip_values_exact(tmp_path):
    aov = _make_synthetic_aov()
    out_path = tmp_path / "aov_values.npz"
    np.savez_compressed(str(out_path), **aov)

    loaded = dict(np.load(str(out_path)))
    for key in aov:
        assert np.array_equal(aov[key], loaded[key]), f"values changed for key {key!r}"


def test_color_equals_direct_plus_indirect(tmp_path):
    aov = _make_synthetic_aov()
    assert np.allclose(aov["color"], aov["direct"] + aov["indirect"], atol=1e-6)
