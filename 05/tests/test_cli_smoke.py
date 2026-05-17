"""test_cli_smoke.py — smoke test: run main.py on a mini AOV and check outputs exist."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_LAB5 = Path(__file__).resolve().parent.parent


def _make_mini_aov(tmp_path: Path, H: int = 10, W: int = 10) -> Path:
    rng = np.random.default_rng(42)
    direct = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
    indirect = rng.uniform(0, 0.3, (H, W, 3)).astype(np.float32)
    depth = rng.uniform(1, 5, (H, W)).astype(np.float32)
    obj_id = (np.arange(H * W).reshape(H, W) % 4).astype(np.int32)
    normal = np.zeros((H, W, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    color = (direct + indirect).astype(np.float32)

    aov_path = tmp_path / "mini_aov.npz"
    np.savez_compressed(str(aov_path),
                        direct=direct, indirect=indirect, depth=depth,
                        obj_id=obj_id, normal=normal, color=color)
    return aov_path


def _run_main(tmp_path: Path, extra_args: list[str]) -> subprocess.CompletedProcess:
    aov_path = _make_mini_aov(tmp_path)
    out_path = tmp_path / "filtered.png"
    cmd = [
        sys.executable, str(_LAB5 / "main.py"),
        "--aov", str(aov_path),
        "--output", str(out_path),
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_LAB5))


def test_mean_creates_output_files(tmp_path):
    result = _run_main(tmp_path, ["--mode", "mean", "--sigma-s", "1.0"])
    assert result.returncode == 0, f"main.py failed:\n{result.stderr}"

    png = tmp_path / "filtered.png"
    ppm = tmp_path / "filtered.ppm"
    txt = tmp_path / "filtered.txt"

    assert png.exists() and png.stat().st_size > 0, "filtered.png not created or empty"
    assert ppm.exists() and ppm.stat().st_size > 0, "filtered.ppm not created or empty"
    assert txt.exists() and txt.stat().st_size > 0, "filtered.txt not created or empty"


def test_median_creates_output_files(tmp_path):
    result = _run_main(tmp_path, ["--mode", "median", "--radius", "2"])
    assert result.returncode == 0, f"main.py failed:\n{result.stderr}"

    assert (tmp_path / "filtered.png").exists()
    assert (tmp_path / "filtered.ppm").exists()
    assert (tmp_path / "filtered.txt").exists()


def test_split_direct_indirect_flag(tmp_path):
    result = _run_main(tmp_path, ["--mode", "mean", "--split-direct-indirect"])
    assert result.returncode == 0, f"main.py failed:\n{result.stderr}"
    assert (tmp_path / "filtered.png").exists()


def test_energy_normalize_object(tmp_path):
    result = _run_main(tmp_path, ["--mode", "mean", "--energy-normalize", "object"])
    assert result.returncode == 0, f"main.py failed:\n{result.stderr}"
    assert (tmp_path / "filtered.png").exists()


def test_dump_debug_creates_debug_images(tmp_path):
    result = _run_main(tmp_path, ["--mode", "mean", "--dump-debug"])
    assert result.returncode == 0, f"main.py failed:\n{result.stderr}"

    for name in ("debug_objid.png", "debug_normal.png", "debug_depth.png", "noisy.png", "diff.png"):
        assert (tmp_path / name).exists(), f"debug image {name!r} not created"


def test_txt_contains_expected_sections(tmp_path):
    result = _run_main(tmp_path, ["--mode", "mean"])
    assert result.returncode == 0

    txt = (tmp_path / "filtered.txt").read_text()
    assert "Lab 5 bilateral filter run" in txt
    assert "mode: mean" in txt
    assert "Energy per object" in txt
