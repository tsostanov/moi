from __future__ import annotations

import argparse
import concurrent.futures
import math
import random
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path


EPSILON = 1e-6
GAMMA = 2.2


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vec3":
        length = self.length()
        if length < EPSILON:
            raise ValueError("Cannot normalize a zero vector.")
        return self / length

    def mul(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def luminance(self) -> float:
        return 0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z

    def is_black(self) -> bool:
        return self.max_component() <= 0.0


BLACK = Vec3(0.0, 0.0, 0.0)
ONE = Vec3(1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: Vec3
    specular: Vec3 = BLACK
    emission: Vec3 = BLACK

    def __post_init__(self) -> None:
        for channel_name, value in (
            ("R", self.diffuse.x + self.specular.x),
            ("G", self.diffuse.y + self.specular.y),
            ("B", self.diffuse.z + self.specular.z),
        ):
            if value > 1.0 + 1e-9:
                raise ValueError(
                    f"Material {self.name!r} is not energy conserving in {channel_name}: "
                    f"diffuse + specular = {value:.3f}."
                )


@dataclass
class Triangle:
    v0: Vec3
    v1: Vec3
    v2: Vec3
    material_id: int
    normal: Vec3 = field(init=False)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        cross = (self.v1 - self.v0).cross(self.v2 - self.v0)
        self.area = 0.5 * cross.length()
        if self.area < EPSILON:
            raise ValueError("Degenerate triangle in scene.")
        self.normal = cross.normalize()

    def sample_point(self, rng: random.Random) -> Vec3:
        u1 = rng.random()
        u2 = rng.random()
        s = math.sqrt(u1)
        b0 = 1.0 - s
        b1 = s * (1.0 - u2)
        b2 = s * u2
        return self.v0 * b0 + self.v1 * b1 + self.v2 * b2


@dataclass(frozen=True)
class Hit:
    distance: float
    point: Vec3
    triangle_id: int


@dataclass
class Scene:
    materials: list[Material]
    triangles: list[Triangle]
    light_ids: list[int] = field(default_factory=list)
    light_cdf: list[float] = field(default_factory=list)
    light_power: float = 0.0

    def rebuild_lights(self) -> None:
        weights: list[float] = []
        self.light_ids.clear()
        for triangle_id, triangle in enumerate(self.triangles):
            emission = self.materials[triangle.material_id].emission
            if emission.max_component() <= 0.0:
                continue
            # For a Lambertian triangle source, total power is proportional to pi * area * radiance.
            power = math.pi * triangle.area * emission.luminance()
            if power > 0.0:
                self.light_ids.append(triangle_id)
                weights.append(power)

        self.light_power = sum(weights)
        self.light_cdf.clear()
        if self.light_power <= 0.0:
            return

        running = 0.0
        for weight in weights:
            running += weight / self.light_power
            self.light_cdf.append(running)
        self.light_cdf[-1] = 1.0

    def intersect(self, ray: Ray, max_distance: float = float("inf")) -> Hit | None:
        best_distance = max_distance
        best_triangle_id = -1

        for triangle_id, triangle in enumerate(self.triangles):
            distance = intersect_triangle(ray, triangle, best_distance)
            if distance is not None and distance < best_distance:
                best_distance = distance
                best_triangle_id = triangle_id

        if best_triangle_id < 0:
            return None
        return Hit(
            distance=best_distance,
            point=ray.origin + ray.direction * best_distance,
            triangle_id=best_triangle_id,
        )

    def is_occluded(self, ray: Ray, max_distance: float, ignored_triangle_id: int) -> bool:
        for triangle_id, triangle in enumerate(self.triangles):
            if triangle_id == ignored_triangle_id:
                continue
            if intersect_triangle(ray, triangle, max_distance) is not None:
                return True
        return False

    def sample_light(self, rng: random.Random) -> tuple[int, float]:
        if not self.light_ids:
            raise ValueError("Scene has no emissive triangles.")
        u = rng.random()
        for index, cdf_value in enumerate(self.light_cdf):
            if u <= cdf_value:
                triangle_id = self.light_ids[index]
                prev_cdf = 0.0 if index == 0 else self.light_cdf[index - 1]
                select_pdf = cdf_value - prev_cdf
                return triangle_id, select_pdf
        triangle_id = self.light_ids[-1]
        prev_cdf = 0.0 if len(self.light_cdf) == 1 else self.light_cdf[-2]
        return triangle_id, 1.0 - prev_cdf


@dataclass(frozen=True)
class Camera:
    origin: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    vertical_scale: float
    aspect: float

    @classmethod
    def look_at(cls, origin: Vec3, target: Vec3, fov_degrees: float, aspect: float) -> "Camera":
        forward = (target - origin).normalize()
        world_up = Vec3(0.0, 1.0, 0.0)
        right = forward.cross(world_up).normalize()
        up = right.cross(forward).normalize()
        vertical_scale = math.tan(math.radians(fov_degrees) * 0.5)
        return cls(origin, forward, right, up, vertical_scale, aspect)

    def make_ray(self, pixel_x: int, pixel_y: int, width: int, height: int, rng: random.Random) -> Ray:
        jitter_x = rng.random()
        jitter_y = rng.random()
        ndc_x = ((pixel_x + jitter_x) / width) * 2.0 - 1.0
        ndc_y = 1.0 - ((pixel_y + jitter_y) / height) * 2.0
        sensor_x = ndc_x * self.aspect * self.vertical_scale
        sensor_y = ndc_y * self.vertical_scale
        direction = (self.forward + self.right * sensor_x + self.up * sensor_y).normalize()
        return Ray(self.origin, direction)


def intersect_triangle(ray: Ray, triangle: Triangle, max_distance: float) -> float | None:
    edge1 = triangle.v1 - triangle.v0
    edge2 = triangle.v2 - triangle.v0
    pvec = ray.direction.cross(edge2)
    determinant = edge1.dot(pvec)
    if abs(determinant) < EPSILON:
        return None

    inv_det = 1.0 / determinant
    tvec = ray.origin - triangle.v0
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None

    qvec = tvec.cross(edge1)
    v = ray.direction.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None

    distance = edge2.dot(qvec) * inv_det
    if distance <= EPSILON or distance >= max_distance:
        return None
    return distance


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return direction - normal * (2.0 * direction.dot(normal))


def build_basis(normal: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    helper = Vec3(0.0, 1.0, 0.0) if abs(normal.y) < 0.999 else Vec3(1.0, 0.0, 0.0)
    tangent = helper.cross(normal).normalize()
    bitangent = normal.cross(tangent).normalize()
    return tangent, bitangent, normal


def sample_cosine_hemisphere(normal: Vec3, rng: random.Random) -> Vec3:
    tangent, bitangent, n = build_basis(normal)
    u1 = rng.random()
    u2 = rng.random()
    radius = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = radius * math.cos(phi)
    z = radius * math.sin(phi)
    y = math.sqrt(max(0.0, 1.0 - u1))
    return (tangent * x + bitangent * z + n * y).normalize()


def offset_ray_origin(point: Vec3, normal: Vec3, direction: Vec3) -> Vec3:
    sign = 1.0 if normal.dot(direction) >= 0.0 else -1.0
    return point + normal * (EPSILON * 8.0 * sign)


def estimate_direct_light(point: Vec3, normal: Vec3, material: Material, scene: Scene, rng: random.Random) -> Vec3:
    if material.diffuse.is_black() or not scene.light_ids:
        return BLACK

    light_id, select_pdf = scene.sample_light(rng)
    light_triangle = scene.triangles[light_id]
    light_material = scene.materials[light_triangle.material_id]
    light_point = light_triangle.sample_point(rng)

    to_light = light_point - point
    distance_squared = to_light.dot(to_light)
    if distance_squared < EPSILON:
        return BLACK

    distance = math.sqrt(distance_squared)
    wi = to_light / distance
    surface_cosine = max(0.0, normal.dot(wi))
    light_cosine = max(0.0, light_triangle.normal.dot(-wi))
    if surface_cosine <= 0.0 or light_cosine <= 0.0:
        return BLACK

    shadow_ray = Ray(offset_ray_origin(point, normal, wi), wi)
    if scene.is_occluded(shadow_ray, distance - EPSILON * 16.0, light_id):
        return BLACK

    pdf_area = select_pdf / light_triangle.area
    brdf = material.diffuse * (1.0 / math.pi)
    geometry = surface_cosine * light_cosine / distance_squared
    return brdf.mul(light_material.emission) * (geometry / pdf_area)


def trace_path(ray: Ray, scene: Scene, rng: random.Random, max_depth: int, rr_depth: int) -> Vec3:
    radiance = BLACK
    throughput = ONE
    specular_bounce = True

    for depth in range(max_depth):
        hit = scene.intersect(ray)
        if hit is None:
            break

        triangle = scene.triangles[hit.triangle_id]
        material = scene.materials[triangle.material_id]
        front_face = triangle.normal.dot(ray.direction) < 0.0
        normal = triangle.normal if front_face else -triangle.normal

        if not material.emission.is_black():
            if front_face and (depth == 0 or specular_bounce):
                radiance = radiance + throughput.mul(material.emission)
            break

        direct = estimate_direct_light(hit.point, normal, material, scene, rng)
        radiance = radiance + throughput.mul(direct)

        diffuse_weight = max(0.0, material.diffuse.luminance())
        specular_weight = max(0.0, material.specular.luminance())
        event_weight = diffuse_weight + specular_weight
        if event_weight <= 0.0:
            break

        specular_probability = specular_weight / event_weight
        if rng.random() < specular_probability:
            if specular_probability <= 0.0:
                break
            next_direction = reflect(ray.direction, normal).normalize()
            throughput = throughput.mul(material.specular) * (1.0 / specular_probability)
            specular_bounce = True
        else:
            diffuse_probability = 1.0 - specular_probability
            if diffuse_probability <= 0.0:
                break
            next_direction = sample_cosine_hemisphere(normal, rng)
            throughput = throughput.mul(material.diffuse) * (1.0 / diffuse_probability)
            specular_bounce = False

        if depth + 1 >= rr_depth:
            survival_probability = min(0.95, max(0.05, throughput.max_component()))
            if rng.random() > survival_probability:
                break
            throughput = throughput * (1.0 / survival_probability)

        ray = Ray(offset_ray_origin(hit.point, normal, next_direction), next_direction)

    return radiance


def add_quad(
    triangles: list[Triangle],
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    material_id: int,
) -> None:
    triangles.append(Triangle(v0, v1, v2, material_id))
    triangles.append(Triangle(v0, v2, v3, material_id))


def add_box(triangles: list[Triangle], p_min: Vec3, p_max: Vec3, material_id: int) -> None:
    xmin, ymin, zmin = p_min.x, p_min.y, p_min.z
    xmax, ymax, zmax = p_max.x, p_max.y, p_max.z

    add_quad(triangles, Vec3(xmin, ymin, zmax), Vec3(xmax, ymin, zmax), Vec3(xmax, ymax, zmax), Vec3(xmin, ymax, zmax), material_id)
    add_quad(triangles, Vec3(xmax, ymin, zmin), Vec3(xmin, ymin, zmin), Vec3(xmin, ymax, zmin), Vec3(xmax, ymax, zmin), material_id)
    add_quad(triangles, Vec3(xmin, ymin, zmin), Vec3(xmin, ymin, zmax), Vec3(xmin, ymax, zmax), Vec3(xmin, ymax, zmin), material_id)
    add_quad(triangles, Vec3(xmax, ymin, zmax), Vec3(xmax, ymin, zmin), Vec3(xmax, ymax, zmin), Vec3(xmax, ymax, zmax), material_id)
    add_quad(triangles, Vec3(xmin, ymax, zmax), Vec3(xmax, ymax, zmax), Vec3(xmax, ymax, zmin), Vec3(xmin, ymax, zmin), material_id)
    add_quad(triangles, Vec3(xmin, ymin, zmin), Vec3(xmax, ymin, zmin), Vec3(xmax, ymin, zmax), Vec3(xmin, ymin, zmax), material_id)


def add_pyramid(triangles: list[Triangle], center: Vec3, radius: float, height: float, material_id: int) -> None:
    y0 = center.y
    apex = Vec3(center.x, y0 + height, center.z)
    p0 = Vec3(center.x - radius, y0, center.z - radius)
    p1 = Vec3(center.x + radius, y0, center.z - radius)
    p2 = Vec3(center.x + radius, y0, center.z + radius)
    p3 = Vec3(center.x - radius, y0, center.z + radius)
    add_quad(triangles, p0, p3, p2, p1, material_id)
    triangles.append(Triangle(p0, p1, apex, material_id))
    triangles.append(Triangle(p1, p2, apex, material_id))
    triangles.append(Triangle(p2, p3, apex, material_id))
    triangles.append(Triangle(p3, p0, apex, material_id))


def load_obj_triangles(path: Path, material_id: int, scale: float, offset: Vec3) -> list[Triangle]:
    vertices: list[Vec3] = []
    triangles: list[Triangle] = []

    def parse_vertex_index(token: str) -> int:
        value = int(token.split("/")[0])
        if value < 0:
            return len(vertices) + value
        return value - 1

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(
                Vec3(float(parts[1]), float(parts[2]), float(parts[3])) * scale + offset
            )
        elif parts[0] == "f" and len(parts) >= 4:
            indices = [parse_vertex_index(token) for token in parts[1:]]
            for local_index in range(1, len(indices) - 1):
                triangles.append(
                    Triangle(
                        vertices[indices[0]],
                        vertices[indices[local_index]],
                        vertices[indices[local_index + 1]],
                        material_id,
                    )
                )

    if not triangles:
        raise ValueError(f"OBJ file {path} does not contain polygon faces.")
    return triangles


def make_scene(args: argparse.Namespace) -> Scene:
    light_color = Vec3(*args.light_color) * args.light_scale

    if args.material_mode == "diffuse":
        object_specular = Vec3(0.0, 0.0, 0.0)
    elif args.material_mode == "mirror":
        object_specular = Vec3(0.82, 0.82, 0.82)
    else:
        object_specular = Vec3(0.25, 0.25, 0.25)

    materials = [
        Material("white wall", Vec3(0.72, 0.72, 0.72)),
        Material("red wall", Vec3(0.75, 0.12, 0.10)),
        Material("green wall", Vec3(0.12, 0.62, 0.18)),
        Material("mixed object", Vec3(0.45, 0.36, 0.22), object_specular),
        Material("mirror object", Vec3(0.04, 0.04, 0.04), Vec3(0.86, 0.86, 0.86)),
        Material("area light", BLACK, BLACK, light_color),
        Material("obj material", Vec3(0.34, 0.50, 0.78), Vec3(0.18, 0.18, 0.18)),
    ]

    white, red, green, mixed, mirror, light, obj_material = range(len(materials))
    triangles: list[Triangle] = []

    add_quad(triangles, Vec3(-1.0, 0.0, -1.0), Vec3(-1.0, 0.0, 1.0), Vec3(1.0, 0.0, 1.0), Vec3(1.0, 0.0, -1.0), white)
    add_quad(triangles, Vec3(-1.0, 2.0, -1.0), Vec3(1.0, 2.0, -1.0), Vec3(1.0, 2.0, 1.0), Vec3(-1.0, 2.0, 1.0), white)
    add_quad(triangles, Vec3(-1.0, 0.0, -1.0), Vec3(1.0, 0.0, -1.0), Vec3(1.0, 2.0, -1.0), Vec3(-1.0, 2.0, -1.0), white)
    add_quad(triangles, Vec3(-1.0, 0.0, 1.0), Vec3(-1.0, 0.0, -1.0), Vec3(-1.0, 2.0, -1.0), Vec3(-1.0, 2.0, 1.0), red)
    add_quad(triangles, Vec3(1.0, 0.0, -1.0), Vec3(1.0, 0.0, 1.0), Vec3(1.0, 2.0, 1.0), Vec3(1.0, 2.0, -1.0), green)

    add_quad(triangles, Vec3(-0.35, 1.985, -0.35), Vec3(0.35, 1.985, -0.35), Vec3(0.35, 1.985, 0.35), Vec3(-0.35, 1.985, 0.35), light)

    if args.scene == "cornell":
        add_box(triangles, Vec3(0.25, 0.0, -0.55), Vec3(0.75, 0.85, -0.05), mirror)
        add_pyramid(triangles, Vec3(-0.42, 0.0, 0.20), 0.35, 0.85, mixed)
    else:
        add_box(triangles, Vec3(-0.62, 0.0, -0.40), Vec3(-0.12, 0.70, 0.10), mixed)
        add_box(triangles, Vec3(0.15, 0.0, -0.72), Vec3(0.72, 1.15, -0.15), mirror)

    if args.obj:
        triangles.extend(load_obj_triangles(Path(args.obj), obj_material, args.obj_scale, Vec3(*args.obj_offset)))

    scene = Scene(materials, triangles)
    scene.rebuild_lights()
    return scene


def render_rows(y_start: int, y_end: int, scene: Scene, camera: Camera, args: argparse.Namespace) -> tuple[int, list[Vec3]]:
    rng = random.Random(args.seed + y_start * 1_000_003)
    rows: list[Vec3] = []
    for y in range(y_start, y_end):
        for x in range(args.width):
            pixel = BLACK
            for _ in range(args.samples):
                ray = camera.make_ray(x, y, args.width, args.height, rng)
                pixel = pixel + trace_path(ray, scene, rng, args.max_depth, args.rr_depth)
            rows.append(pixel * (1.0 / args.samples))
    return y_start, rows


def render(scene: Scene, camera: Camera, args: argparse.Namespace) -> list[Vec3]:
    framebuffer = [BLACK for _ in range(args.width * args.height)]
    start_time = time.monotonic()

    if args.workers <= 1:
        next_progress_row = 0
        for y_start in range(args.height):
            _, row = render_rows(y_start, y_start + 1, scene, camera, args)
            framebuffer[y_start * args.width : (y_start + 1) * args.width] = row
            if y_start >= next_progress_row or y_start == args.height - 1:
                elapsed = time.monotonic() - start_time
                done = (y_start + 1) / args.height
                print(f"render {100.0 * done:5.1f}%  elapsed {elapsed:6.1f}s", flush=True)
                next_progress_row += max(1, args.height // 10)
        return framebuffer

    chunk_size = max(1, args.height // (args.workers * 8))
    tasks = [
        (y_start, min(args.height, y_start + chunk_size))
        for y_start in range(0, args.height, chunk_size)
    ]
    completed_rows = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(render_rows, y_start, y_end, scene, camera, args)
                for y_start, y_end in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                y_start, rows = future.result()
                row_count = len(rows) // args.width
                framebuffer[y_start * args.width : (y_start + row_count) * args.width] = rows
                completed_rows += row_count
                elapsed = time.monotonic() - start_time
                done = completed_rows / args.height
                print(f"render {100.0 * done:5.1f}%  elapsed {elapsed:6.1f}s", flush=True)
    except OSError as error:
        print(f"warning: parallel render is unavailable ({error}); falling back to one worker.")
        args.workers = 1
        return render(scene, camera, args)

    return framebuffer


def build_display_bytes(pixels: list[Vec3], normalization: float) -> bytearray:
    data = bytearray()
    for pixel in pixels:
        relative = pixel * (1.0 / normalization)
        for value in (relative.x, relative.y, relative.z):
            clamped = max(0.0, min(1.0, value))
            corrected = clamped ** (1.0 / GAMMA)
            data.append(int(round(corrected * 255.0)))
    return data


def write_png(path: Path, rgb_data: bytearray, width: int, height: int) -> None:
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    scanlines = bytearray()
    row_size = width * 3
    for y in range(height):
        scanlines.append(0)
        start = y * row_size
        scanlines.extend(rgb_data[start : start + row_size])

    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with path.open("wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")
        file.write(png_chunk(b"IHDR", ihdr))
        file.write(png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6)))
        file.write(png_chunk(b"IEND", b""))


def write_ppm(path: Path, rgb_data: bytearray, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(rgb_data)


def write_images(path: Path, pixels: list[Vec3], width: int, height: int, white_point: float | None) -> float:
    max_value = max((pixel.max_component() for pixel in pixels), default=0.0)
    normalization = white_point if white_point and white_point > 0.0 else max(max_value, 1e-12)
    rgb_data = build_display_bytes(pixels, normalization)
    write_ppm(path, rgb_data, width, height)
    write_png(path.with_suffix(".png"), rgb_data, width, height)
    return normalization


def write_stats(path: Path, args: argparse.Namespace, scene: Scene, normalization: float, elapsed: float) -> None:
    material_lines = []
    for material in scene.materials:
        material_lines.append(
            f"- {material.name}: diffuse={material.diffuse}, specular={material.specular}, emission={material.emission}"
        )

    text = "\n".join(
        [
            "Lab 4 path tracing run",
            f"image: {args.width}x{args.height}",
            f"samples_per_pixel: {args.samples}",
            f"max_depth: {args.max_depth}",
            f"russian_roulette_from_depth: {args.rr_depth}",
            f"seed: {args.seed}",
            f"triangles: {len(scene.triangles)}",
            f"emissive_triangles: {len(scene.light_ids)}",
            f"normalization_white_point: {normalization:.8g}",
            f"elapsed_seconds: {elapsed:.3f}",
            "",
            "Materials:",
            *material_lines,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 4: triangular-mesh path tracer with diffuse/specular RGB materials and area lights."
    )
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1, help="Rays per pixel.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--rr-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--scene", choices=("cornell", "mirror-test"), default="cornell")
    parser.add_argument("--material-mode", choices=("balanced", "diffuse", "mirror"), default="balanced")
    parser.add_argument("--camera", nargs=3, type=float, default=(0.0, 1.05, 3.35), metavar=("X", "Y", "Z"))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.92, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--fov", type=float, default=42.0)
    parser.add_argument("--light-color", nargs=3, type=float, default=(12.0, 10.4, 8.6), metavar=("R", "G", "B"))
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument("--white-point", type=float, default=None, help="Manual normalization value. Default: image maximum.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes. Use 1 for deterministic row order.")
    parser.add_argument("--obj", default="", help="Optional OBJ mesh inserted into the room.")
    parser.add_argument("--obj-scale", type=float, default=0.45)
    parser.add_argument("--obj-offset", nargs=3, type=float, default=(0.0, 0.0, 0.15), metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "render.ppm")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if args.width > 1000 or args.height > 1000:
        raise ValueError("Lab requirement: image size must not exceed 1000x1000.")
    if args.width < 500 or args.height < 500:
        print("warning: final lab image should be at least 500x500; this size is useful only for quick tests.")
    if args.samples <= 0:
        raise ValueError("Samples per pixel must be positive.")
    if args.max_depth <= 0:
        raise ValueError("Max depth must be positive.")
    if args.workers <= 0:
        raise ValueError("Workers must be positive.")
    return args


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    scene = make_scene(args)
    camera = Camera.look_at(
        origin=Vec3(*args.camera),
        target=Vec3(*args.look_at),
        fov_degrees=args.fov,
        aspect=args.width / args.height,
    )

    print(
        f"scene={args.scene}, triangles={len(scene.triangles)}, lights={len(scene.light_ids)}, "
        f"resolution={args.width}x{args.height}, spp={args.samples}"
    )
    framebuffer = render(scene, camera, args)
    elapsed = time.monotonic() - start_time
    normalization = write_images(args.output, framebuffer, args.width, args.height, args.white_point)
    write_stats(args.output.with_suffix(".txt"), args, scene, normalization, elapsed)
    print(f"saved {args.output}")
    print(f"saved {args.output.with_suffix('.png')}")
    print(f"saved {args.output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
