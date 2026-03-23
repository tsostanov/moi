from __future__ import annotations

import math
import random
from dataclasses import dataclass


SAMPLE_COUNT = 100000
BASE_SEED = 20260323
EPSILON = 1e-9


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

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
            raise ValueError("Zero-length vector cannot be normalized.")
        return self * (1.0 / length)


def seed_from_key(key: str) -> int:
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(key))
    return BASE_SEED + checksum


def format_vec(value: Vec3) -> str:
    return f"({value.x:.6f}, {value.y:.6f}, {value.z:.6f})"


def format_table(headers: list[str], rows: list[list[str]], alignments: str = "") -> str:
    if not alignments:
        alignments = "r" * len(headers)

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_cell(index: int, cell: str) -> str:
        if alignments[index] == "l":
            return f" {cell:<{widths[index]}} "
        return f" {cell:>{widths[index]}} "

    def format_row(row: list[str]) -> str:
        return "|" + "|".join(format_cell(index, cell) for index, cell in enumerate(row)) + "|"

    lines = [border(), format_row(headers), border()]
    for row in rows:
        lines.append(format_row(row))
    lines.append(border())
    return "\n".join(lines)


def build_count_table(bin_name: str, counts: list[int], expected: float) -> str:
    rows = []
    for index, count in enumerate(counts, start=1):
        diff = count - expected
        rel = 100.0 * diff / expected
        rows.append(
            [
                str(index),
                str(count),
                f"{expected:.1f}",
                f"{diff:+.1f}",
                f"{rel:+.2f}%",
            ]
        )
    return format_table(
        [bin_name, "count", "expected", "diff", "diff %"],
        rows,
    )


def build_basis(normal: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    n = normal.normalize()
    helper = Vec3(0.0, 0.0, 1.0) if abs(n.z) < 0.999 else Vec3(0.0, 1.0, 0.0)
    tangent = helper.cross(n).normalize()
    bitangent = n.cross(tangent).normalize()
    return tangent, bitangent, n


def sample_triangle_point(v1: Vec3, v2: Vec3, v3: Vec3, rng: random.Random) -> tuple[Vec3, tuple[float, float, float]]:
    u1 = rng.random()
    u2 = rng.random()
    s = math.sqrt(u1)
    b1 = 1.0 - s
    b2 = s * (1.0 - u2)
    b3 = s * u2
    point = v1 * b1 + v2 * b2 + v3 * b3
    return point, (b1, b2, b3)


def sample_disk_point(
    center: Vec3,
    basis: tuple[Vec3, Vec3, Vec3],
    radius: float,
    rng: random.Random,
) -> tuple[Vec3, float, float]:
    tangent, bitangent, _ = basis
    u1 = rng.random()
    u2 = rng.random()
    local_radius = radius * math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = local_radius * math.cos(phi)
    y = local_radius * math.sin(phi)
    point = center + tangent * x + bitangent * y
    return point, u1, phi


def sample_uniform_sphere_direction(rng: random.Random) -> tuple[Vec3, float, float]:
    z = 1.0 - 2.0 * rng.random()
    phi = 2.0 * math.pi * rng.random()
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    direction = Vec3(radial * math.cos(phi), radial * math.sin(phi), z)
    return direction, z, phi


def sample_cosine_direction(basis: tuple[Vec3, Vec3, Vec3], rng: random.Random) -> tuple[Vec3, float, float]:
    tangent, bitangent, n = basis
    u1 = rng.random()
    u2 = rng.random()
    radial = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = radial * math.cos(phi)
    y = radial * math.sin(phi)
    z = math.sqrt(1.0 - u1)
    direction = (tangent * x + bitangent * y + n * z).normalize()
    return direction, z, phi


def phi_bin_index(phi: float, bins: int) -> int:
    return min(int((phi / (2.0 * math.pi)) * bins), bins - 1)


def build_metrics_table(rows: list[tuple[str, str]]) -> str:
    return format_table(["metric", "value"], [[name, value] for name, value in rows], alignments="lr")


def analyze_triangle() -> str:
    rng = random.Random(seed_from_key("triangle"))
    v1 = Vec3(0.0, 0.0, 0.0)
    v2 = Vec3(3.0, 1.0, 2.0)
    v3 = Vec3(1.0, 4.0, 1.0)

    sum_point = Vec3(0.0, 0.0, 0.0)
    sum_b1 = 0.0
    sum_b2 = 0.0
    sum_b3 = 0.0
    max_bary_sum_error = 0.0
    invalid_count = 0
    region_counts = [0, 0, 0, 0]

    for _ in range(SAMPLE_COUNT):
        point, (b1, b2, b3) = sample_triangle_point(v1, v2, v3, rng)
        sum_point = sum_point + point
        sum_b1 += b1
        sum_b2 += b2
        sum_b3 += b3
        max_bary_sum_error = max(max_bary_sum_error, abs(b1 + b2 + b3 - 1.0))
        if min(b1, b2, b3) < -EPSILON:
            invalid_count += 1

        if b1 >= 0.5:
            region_counts[0] += 1
        elif b2 >= 0.5:
            region_counts[1] += 1
        elif b3 >= 0.5:
            region_counts[2] += 1
        else:
            region_counts[3] += 1

    centroid = (v1 + v2 + v3) * (1.0 / 3.0)
    sample_centroid = sum_point * (1.0 / SAMPLE_COUNT)
    centroid_error = (sample_centroid - centroid).length()

    lines = [
        "1. Uniform points inside triangle",
        f"V1 = {format_vec(v1)}, V2 = {format_vec(v2)}, V3 = {format_vec(v3)}",
        build_metrics_table(
            [
                ("generated points", str(SAMPLE_COUNT)),
                ("invalid points", str(invalid_count)),
                ("max |b1+b2+b3-1|", f"{max_bary_sum_error:.12f}"),
                ("mean barycentric", f"({sum_b1 / SAMPLE_COUNT:.6f}, {sum_b2 / SAMPLE_COUNT:.6f}, {sum_b3 / SAMPLE_COUNT:.6f})"),
                ("theoretical centroid", format_vec(centroid)),
                ("sample centroid", format_vec(sample_centroid)),
                ("centroid error", f"{centroid_error:.6f}"),
            ]
        ),
        "Equal-area subtriangles: expected count = N / 4",
        build_count_table("subtriangle", region_counts, SAMPLE_COUNT / 4.0),
        "",
    ]
    return "\n".join(lines)


def analyze_disk() -> str:
    rng = random.Random(seed_from_key("disk"))
    center = Vec3(1.0, -2.0, 3.0)
    normal = Vec3(1.0, 2.0, 3.0).normalize()
    radius = 2.5
    basis = build_basis(normal)
    tangent, bitangent, n = basis

    sum_point = Vec3(0.0, 0.0, 0.0)
    invalid_count = 0
    max_plane_distance = 0.0
    max_radius_ratio = 0.0
    ring_counts = [0] * 5
    sector_counts = [0] * 8

    for _ in range(SAMPLE_COUNT):
        point, area_u, phi = sample_disk_point(center, basis, radius, rng)
        delta = point - center
        local_x = delta.dot(tangent)
        local_y = delta.dot(bitangent)
        plane_distance = abs(delta.dot(n))
        radius_ratio = math.sqrt(local_x * local_x + local_y * local_y) / radius

        sum_point = sum_point + point
        max_plane_distance = max(max_plane_distance, plane_distance)
        max_radius_ratio = max(max_radius_ratio, radius_ratio)

        if plane_distance > 1e-7 or radius_ratio > 1.0 + 1e-7:
            invalid_count += 1

        ring_index = min(int(area_u * len(ring_counts)), len(ring_counts) - 1)
        sector_index = phi_bin_index(phi, len(sector_counts))
        ring_counts[ring_index] += 1
        sector_counts[sector_index] += 1

    sample_centroid = sum_point * (1.0 / SAMPLE_COUNT)
    centroid_error = (sample_centroid - center).length()

    lines = [
        "2. Uniform points inside disk",
        f"C = {format_vec(center)}, N = {format_vec(n)}, Rc = {radius:.6f}",
        build_metrics_table(
            [
                ("generated points", str(SAMPLE_COUNT)),
                ("invalid points", str(invalid_count)),
                ("max plane distance", f"{max_plane_distance:.12f}"),
                ("max radius / Rc", f"{max_radius_ratio:.6f}"),
                ("theoretical centroid", format_vec(center)),
                ("sample centroid", format_vec(sample_centroid)),
                ("centroid error", f"{centroid_error:.6f}"),
            ]
        ),
        "Equal-area rings: expected count = N / 5",
        build_count_table("ring", ring_counts, SAMPLE_COUNT / 5.0),
        "Azimuth sectors: expected count = N / 8",
        build_count_table("sector", sector_counts, SAMPLE_COUNT / 8.0),
        "",
    ]
    return "\n".join(lines)


def analyze_uniform_sphere() -> str:
    rng = random.Random(seed_from_key("sphere"))
    sum_dir = Vec3(0.0, 0.0, 0.0)
    invalid_count = 0
    max_norm_error = 0.0
    z_counts = [0] * 10
    sector_counts = [0] * 8

    for _ in range(SAMPLE_COUNT):
        direction, z, phi = sample_uniform_sphere_direction(rng)
        norm_error = abs(direction.length() - 1.0)
        max_norm_error = max(max_norm_error, norm_error)
        sum_dir = sum_dir + direction

        if norm_error > 1e-7:
            invalid_count += 1

        z_index = min(int(((z + 1.0) * 0.5) * len(z_counts)), len(z_counts) - 1)
        sector_index = phi_bin_index(phi, len(sector_counts))
        z_counts[z_index] += 1
        sector_counts[sector_index] += 1

    mean_dir = sum_dir * (1.0 / SAMPLE_COUNT)

    lines = [
        "3. Uniform directions on unit sphere",
        build_metrics_table(
            [
                ("generated directions", str(SAMPLE_COUNT)),
                ("invalid directions", str(invalid_count)),
                ("max ||w|-1|", f"{max_norm_error:.12f}"),
                ("sample mean direction", format_vec(mean_dir)),
                ("|mean direction|", f"{mean_dir.length():.6f}"),
            ]
        ),
        "z = cos(theta) bins on [-1, 1]: expected count = N / 10",
        build_count_table("z-bin", z_counts, SAMPLE_COUNT / 10.0),
        "Azimuth sectors: expected count = N / 8",
        build_count_table("sector", sector_counts, SAMPLE_COUNT / 8.0),
        "",
    ]
    return "\n".join(lines)


def analyze_cosine_directions() -> str:
    rng = random.Random(seed_from_key("cosine"))
    normal = Vec3(0.0, 0.0, 1.0)
    basis = build_basis(normal)

    invalid_count = 0
    max_norm_error = 0.0
    min_cosine = 1.0
    cosine_sum = 0.0
    u_counts = [0] * 10
    sector_counts = [0] * 8

    for _ in range(SAMPLE_COUNT):
        direction, cosine, phi = sample_cosine_direction(basis, rng)
        norm_error = abs(direction.length() - 1.0)
        max_norm_error = max(max_norm_error, norm_error)
        min_cosine = min(min_cosine, direction.dot(normal))
        cosine_sum += cosine

        if norm_error > 1e-7 or direction.dot(normal) < -1e-7:
            invalid_count += 1

        u = cosine * cosine
        u_index = min(int(u * len(u_counts)), len(u_counts) - 1)
        sector_index = phi_bin_index(phi, len(sector_counts))
        u_counts[u_index] += 1
        sector_counts[sector_index] += 1

    mean_cosine = cosine_sum / SAMPLE_COUNT

    lines = [
        "4. Cosine-weighted directions around N",
        f"N = {format_vec(normal)}",
        build_metrics_table(
            [
                ("generated directions", str(SAMPLE_COUNT)),
                ("invalid directions", str(invalid_count)),
                ("max ||w|-1|", f"{max_norm_error:.12f}"),
                ("min dot(N, w)", f"{min_cosine:.6f}"),
                ("sample mean cos(theta)", f"{mean_cosine:.6f}"),
                ("theoretical mean cos(theta)", f"{2.0 / 3.0:.6f}"),
            ]
        ),
        "u = cos^2(theta) bins: expected count = N / 10",
        build_count_table("u-bin", u_counts, SAMPLE_COUNT / 10.0),
        "Azimuth sectors: expected count = N / 8",
        build_count_table("sector", sector_counts, SAMPLE_COUNT / 8.0),
        "",
    ]
    return "\n".join(lines)


def build_report() -> str:
    sections = [
        "LAB 3: RANDOM DISTRIBUTIONS IN GEOMETRIC DOMAINS",
        f"samples per task = {SAMPLE_COUNT}, base seed = {BASE_SEED}",
        "validation = geometric constraints + equal-probability bin counts",
        "",
        analyze_triangle(),
        analyze_disk(),
        analyze_uniform_sphere(),
        analyze_cosine_directions(),
    ]
    return "\n".join(sections)


def main() -> None:
    print(build_report())


if __name__ == "__main__":
    main()
