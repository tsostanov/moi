"""render_aov.py — path tracer with AOV channel output for Lab 5.

Extends 04/main.py with:
- Per-triangle object IDs for edge-stop bilateral filtering
- AOV channel recording: direct, indirect, depth, obj_id, normal
- Output: aov.npz + preview PNG
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import base path tracer from Lab 4
# ---------------------------------------------------------------------------
_LAB4 = Path(__file__).resolve().parent.parent / "04"
sys.path.insert(0, str(_LAB4))
from main import (  # noqa: E402
    Vec3, Ray, Material, AABB, Triangle, Hit, Scene, Camera,
    intersect_triangle, reflect, sample_cosine_hemisphere,
    offset_ray_origin, estimate_direct_light,
    add_quad, add_box, add_pyramid, add_ceiling_with_light_slot,
    build_display_bytes, write_png, write_ppm,
    BLACK, ONE, EPSILON, GAMMA,
)

# ---------------------------------------------------------------------------
# Object ID constants
# ---------------------------------------------------------------------------
OBJ_FLOOR = 0
OBJ_CEILING = 1
OBJ_BACK_WALL = 2
OBJ_LEFT_WALL = 3
OBJ_RIGHT_WALL = 4
OBJ_LIGHT = 5
OBJ_BOX1 = 6
OBJ_BOX2 = 7


def _add(tris: list[Triangle], ids: list[int], fn, *args, oid: int, **kw) -> None:
    # Все треугольники, которые добавил helper, получают один и тот же obj_id.
    before = len(tris)
    fn(tris, *args, **kw)
    ids.extend([oid] * (len(tris) - before))


def make_scene_aov(args: argparse.Namespace) -> tuple[Scene, list[int]]:
    light_color = Vec3(*args.light_color) * args.light_scale

    if args.material_mode == "diffuse":
        object_specular = Vec3(0.0, 0.0, 0.0)
        box_diffuse = Vec3(0.42, 0.42, 0.46)
        box_specular = Vec3(0.0, 0.0, 0.0)
    elif args.material_mode == "mirror":
        object_specular = Vec3(0.82, 0.82, 0.82)
        box_diffuse = Vec3(0.04, 0.04, 0.04)
        box_specular = Vec3(0.86, 0.86, 0.86)
    else:
        object_specular = Vec3(0.25, 0.25, 0.25)
        box_diffuse = Vec3(0.30, 0.30, 0.34)
        box_specular = Vec3(0.20, 0.20, 0.20)

    materials = [
        Material("white wall", Vec3(0.72, 0.72, 0.72)),
        Material("red wall", Vec3(0.75, 0.12, 0.10)),
        Material("green wall", Vec3(0.12, 0.62, 0.18)),
        Material("mixed object", Vec3(0.45, 0.36, 0.22), object_specular),
        Material("box object", box_diffuse, box_specular),
        Material("area light", BLACK, BLACK, light_color),
        Material("obj material", Vec3(0.34, 0.50, 0.78), Vec3(0.18, 0.18, 0.18)),
    ]
    white, red, green, mixed, box, light, _obj_mat = range(len(materials))

    triangles: list[Triangle] = []
    tri_obj_id: list[int] = []

    _add(triangles, tri_obj_id, add_quad,
         Vec3(-1.0, 0.0, -1.0), Vec3(-1.0, 0.0, 1.0), Vec3(1.0, 0.0, 1.0), Vec3(1.0, 0.0, -1.0), white,
         oid=OBJ_FLOOR)
    _add(triangles, tri_obj_id, add_ceiling_with_light_slot,
         white, 2.0, -1.0, 1.0, -0.35, 0.35,
         oid=OBJ_CEILING)
    _add(triangles, tri_obj_id, add_quad,
         Vec3(-1.0, 0.0, -1.0), Vec3(1.0, 0.0, -1.0), Vec3(1.0, 2.0, -1.0), Vec3(-1.0, 2.0, -1.0), white,
         oid=OBJ_BACK_WALL)
    _add(triangles, tri_obj_id, add_quad,
         Vec3(-1.0, 0.0, 1.0), Vec3(-1.0, 0.0, -1.0), Vec3(-1.0, 2.0, -1.0), Vec3(-1.0, 2.0, 1.0), red,
         oid=OBJ_LEFT_WALL)
    _add(triangles, tri_obj_id, add_quad,
         Vec3(1.0, 0.0, -1.0), Vec3(1.0, 0.0, 1.0), Vec3(1.0, 2.0, 1.0), Vec3(1.0, 2.0, -1.0), green,
         oid=OBJ_RIGHT_WALL)
    _add(triangles, tri_obj_id, add_quad,
         Vec3(-0.35, 2.0, -0.35), Vec3(0.35, 2.0, -0.35), Vec3(0.35, 2.0, 0.35), Vec3(-0.35, 2.0, 0.35), light,
         oid=OBJ_LIGHT)

    if args.scene == "cornell":
        _add(triangles, tri_obj_id, add_box,
             Vec3(0.22, 0.0, -0.48), Vec3(0.72, 0.82, 0.08), box,
             oid=OBJ_BOX1)
        _add(triangles, tri_obj_id, add_pyramid,
             Vec3(-0.42, 0.0, 0.20), 0.35, 0.85, mixed,
             oid=OBJ_BOX2)
    else:
        _add(triangles, tri_obj_id, add_box,
             Vec3(-0.62, 0.0, -0.40), Vec3(-0.12, 0.70, 0.10), mixed,
             oid=OBJ_BOX1)
        _add(triangles, tri_obj_id, add_box,
             Vec3(0.15, 0.0, -0.72), Vec3(0.72, 1.15, -0.15), box,
             oid=OBJ_BOX2)

    scene = Scene(materials, triangles)
    scene.rebuild_lights()
    scene.rebuild_bvh()
    scene.rebuild_native_intersector()
    return scene, tri_obj_id


def trace_path_aov(
    ray: Ray,
    scene: Scene,
    tri_obj_id: list[int],
    rng: random.Random,
    max_depth: int,
    rr_depth: int,
) -> tuple[Vec3, Vec3, Vec3, float, int, Vec3]:
    """Trace path and return (color, direct, indirect, depth, obj_id, normal)."""
    radiance = BLACK
    throughput = ONE
    specular_bounce = True
    first_direct = BLACK
    aov_depth: float = float("inf")
    aov_obj_id: int = -1
    aov_normal = Vec3(0.0, 0.0, 0.0)

    for depth in range(max_depth):
        hit = scene.intersect(ray)
        if hit is None:
            break

        tri = scene.triangles[hit.triangle_id]
        mat = scene.materials[tri.material_id]
        front = tri.normal.dot(ray.direction) < 0.0
        n = tri.normal if front else -tri.normal

        if depth == 0:
            # Для AOV записываем только первое видимое пересечение луча.
            aov_depth = hit.distance
            aov_obj_id = tri_obj_id[hit.triangle_id]
            aov_normal = n

        if not mat.emission.is_black():
            if front and (depth == 0 or specular_bounce):
                c = throughput.mul(mat.emission)
                radiance = radiance + c
                if depth == 0:
                    first_direct = c
            break

        d = estimate_direct_light(hit.point, n, mat, scene, rng)
        dc = throughput.mul(d)
        radiance = radiance + dc
        if depth == 0:
            # Прямой свет первого хита сохраняем отдельно,
            # чтобы потом восстановить indirect как остаток.
            first_direct = dc

        dw = max(0.0, mat.diffuse.luminance())
        sw = max(0.0, mat.specular.luminance())
        ew = dw + sw
        if ew <= 0.0:
            break

        sp = sw / ew
        if rng.random() < sp:
            nd = reflect(ray.direction, n).normalize()
            throughput = throughput.mul(mat.specular) * (1.0 / sp)
            specular_bounce = True
        else:
            dp2 = 1.0 - sp
            nd = sample_cosine_hemisphere(n, rng)
            throughput = throughput.mul(mat.diffuse) * (1.0 / dp2)
            specular_bounce = False

        if depth + 1 >= rr_depth:
            surv = min(0.95, max(0.05, throughput.max_component()))
            if rng.random() > surv:
                break
            throughput = throughput * (1.0 / surv)

        ray = Ray(offset_ray_origin(hit.point, n, nd), nd)

    indirect = Vec3(
        radiance.x - first_direct.x,
        radiance.y - first_direct.y,
        radiance.z - first_direct.z,
    )
    return radiance, first_direct, indirect, aov_depth, aov_obj_id, aov_normal


def render_aov(
    scene: Scene,
    camera: Camera,
    tri_obj_id: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W, S = args.height, args.width, args.samples
    direct = np.zeros((H, W, 3), dtype=np.float32)
    indirect = np.zeros((H, W, 3), dtype=np.float32)
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)
    obj_id_buf = np.full((H, W), -1, dtype=np.int32)
    normal_buf = np.zeros((H, W, 3), dtype=np.float32)

    start = time.monotonic()
    next_report = 0
    for y in range(H):
        rng = random.Random(args.seed + y * 1_000_003)
        for x in range(W):
            d_acc = BLACK
            ind_acc = BLACK
            geom_set = False
            for _ in range(S):
                ray = camera.make_ray(x, y, W, H, rng)
                _col, d, ind, dep, oid, norm = trace_path_aov(
                    ray, scene, tri_obj_id, rng, args.max_depth, args.rr_depth
                )
                d_acc = d_acc + d
                ind_acc = ind_acc + ind
                if not geom_set and dep < float("inf"):
                    # Геометрические AOV достаточно взять из первого
                    # успешного сэмпла данного пикселя.
                    depth_buf[y, x] = dep
                    obj_id_buf[y, x] = oid
                    normal_buf[y, x] = [norm.x, norm.y, norm.z]
                    geom_set = True

            inv_s = 1.0 / S
            direct[y, x] = [d_acc.x * inv_s, d_acc.y * inv_s, d_acc.z * inv_s]
            indirect[y, x] = [ind_acc.x * inv_s, ind_acc.y * inv_s, ind_acc.z * inv_s]

        if y >= next_report or y == H - 1:
            elapsed = time.monotonic() - start
            print(f"render {100.0 * (y + 1) / H:5.1f}%  elapsed {elapsed:6.1f}s", flush=True)
            next_report = y + max(1, H // 10)

    return direct, indirect, depth_buf, obj_id_buf, normal_buf


def save_aov_preview(path: Path, color: np.ndarray, width: int, height: int) -> None:
    max_val = float(color.max())
    norm_val = max(max_val, 1e-12)
    pixels = [
        Vec3(float(color[y, x, 0]), float(color[y, x, 1]), float(color[y, x, 2]))
        for y in range(height)
        for x in range(width)
    ]
    rgb = build_display_bytes(pixels, norm_val)
    write_png(path, rgb, width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 5 AOV path tracer")
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--rr-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--scene", choices=("cornell", "mirror-test"), default="cornell")
    parser.add_argument("--material-mode", choices=("balanced", "diffuse", "mirror"), default="balanced")
    parser.add_argument(
        "--camera", nargs=3, type=float, default=(-0.28, 1.08, 3.45), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--look-at", nargs=3, type=float, default=(0.08, 0.90, -0.08), metavar=("X", "Y", "Z")
    )
    parser.add_argument("--fov", type=float, default=42.0)
    parser.add_argument(
        "--light-color", nargs=3, type=float, default=(12.0, 10.4, 8.6), metavar=("R", "G", "B")
    )
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "aov.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        scene, tri_obj_id = make_scene_aov(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    camera = Camera.look_at(
        origin=Vec3(*args.camera),
        target=Vec3(*args.look_at),
        fov_degrees=args.fov,
        aspect=args.width / args.height,
    )
    print(
        f"scene=cornell triangles={len(scene.triangles)} "
        f"resolution={args.width}x{args.height} spp={args.samples} max_depth={args.max_depth}",
        flush=True,
    )

    direct, indirect, depth, obj_id, normal = render_aov(scene, camera, tri_obj_id, args)
    color = direct + indirect

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Один NPZ хранит и цвет, и все вспомогательные буферы для фильтра.
    np.savez_compressed(
        str(out),
        direct=direct,
        indirect=indirect,
        depth=depth,
        obj_id=obj_id,
        normal=normal,
        color=color,
    )
    print(f"saved {out}")

    preview = out.with_suffix(".png")
    save_aov_preview(preview, color, args.width, args.height)
    print(f"saved {preview}")


if __name__ == "__main__":
    main()
