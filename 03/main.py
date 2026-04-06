from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_COUNT = 100000
PLOT_SAMPLE_COUNT = 20000
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
            raise ValueError("Нельзя нормализовать нулевой вектор.")
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
        [bin_name, "число", "ожид.", "разн.", "разн. %"],
        rows,
    )


def chi_square_stat(counts: list[int], expected: float) -> float:
    return sum(((count - expected) ** 2) / expected for count in counts)


def max_relative_deviation_percent(counts: list[int], expected: float) -> float:
    return max(abs(count - expected) / expected for count in counts) * 100.0


def rms_relative_deviation_percent(counts: list[int], expected: float) -> float:
    mean_square = sum(((count - expected) / expected) ** 2 for count in counts) / len(counts)
    return math.sqrt(mean_square) * 100.0


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
    return format_table(["метрика", "значение"], [[name, value] for name, value in rows], alignments="lr")


def triangle_parameters() -> tuple[Vec3, Vec3, Vec3]:
    return (
        Vec3(0.0, 0.0, 0.0),
        Vec3(3.0, 1.0, 2.0),
        Vec3(1.0, 4.0, 1.0),
    )


def disk_parameters() -> tuple[Vec3, Vec3, float]:
    return (
        Vec3(1.0, -2.0, 3.0),
        Vec3(1.0, 2.0, 3.0).normalize(),
        2.5,
    )


def cosine_parameters() -> Vec3:
    return Vec3(0.0, 0.0, 1.0)


def build_triangle_plane_basis(v1: Vec3, v2: Vec3, v3: Vec3) -> tuple[Vec3, Vec3]:
    edge_u = (v2 - v1).normalize()
    normal = (v2 - v1).cross(v3 - v1).normalize()
    edge_v = normal.cross(edge_u).normalize()
    return edge_u, edge_v


def to_plane_coords(origin: Vec3, axis_u: Vec3, axis_v: Vec3, point: Vec3) -> tuple[float, float]:
    delta = point - origin
    return delta.dot(axis_u), delta.dot(axis_v)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.7)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    output_path = Path(__file__).resolve().parent / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def configure_3d_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)


def create_triangle_plot() -> Path:
    v1, v2, v3 = triangle_parameters()
    axis_u, axis_v = build_triangle_plane_basis(v1, v2, v3)
    rng = random.Random(seed_from_key("triangle-plot"))

    points_uv: list[tuple[float, float]] = []
    region_counts = [0, 0, 0, 0]
    vertices_uv = [to_plane_coords(v1, axis_u, axis_v, vertex) for vertex in (v1, v2, v3)]

    for _ in range(PLOT_SAMPLE_COUNT):
        point, (b1, b2, b3) = sample_triangle_point(v1, v2, v3, rng)
        points_uv.append(to_plane_coords(v1, axis_u, axis_v, point))

        if b1 >= 0.5:
            region_counts[0] += 1
        elif b2 >= 0.5:
            region_counts[1] += 1
        elif b3 >= 0.5:
            region_counts[2] += 1
        else:
            region_counts[3] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    scatter_ax, bar_ax = axes

    xs = [point[0] for point in points_uv]
    ys = [point[1] for point in points_uv]
    polygon_x = [vertices_uv[0][0], vertices_uv[1][0], vertices_uv[2][0], vertices_uv[0][0]]
    polygon_y = [vertices_uv[0][1], vertices_uv[1][1], vertices_uv[2][1], vertices_uv[0][1]]

    scatter_ax.scatter(xs, ys, s=1, alpha=0.35, color="#176087", rasterized=True)
    scatter_ax.plot(polygon_x, polygon_y, color="#102a43", linewidth=2.0)
    for index, vertex in enumerate(vertices_uv, start=1):
        scatter_ax.scatter([vertex[0]], [vertex[1]], s=45, color="#d64545", zorder=3)
        scatter_ax.text(vertex[0], vertex[1], f"  V{index}", fontsize=10, va="bottom")
    scatter_ax.set_title("Точки внутри треугольника")
    scatter_ax.set_xlabel("локальная координата u")
    scatter_ax.set_ylabel("локальная координата v")
    scatter_ax.set_aspect("equal", adjustable="box")
    style_axes(scatter_ax)

    expected = PLOT_SAMPLE_COUNT / 4.0
    labels = ["О1", "О2", "О3", "О4"]
    bar_ax.bar(labels, region_counts, color="#4f9d69")
    bar_ax.axhline(expected, color="#8b1e3f", linestyle="--", linewidth=1.5, label=f"ожид. = {expected:.0f}")
    bar_ax.set_title("Равновеликие подтреугольники")
    bar_ax.set_ylabel("число точек")
    bar_ax.legend()
    style_axes(bar_ax)

    fig.suptitle(f"Распределение в треугольнике ({PLOT_SAMPLE_COUNT} точек)")
    return save_figure(fig, "triangle_distribution.png")


def create_disk_plot() -> Path:
    center, normal, radius = disk_parameters()
    basis = build_basis(normal)
    tangent, bitangent, _ = basis
    rng = random.Random(seed_from_key("disk-plot"))

    xs: list[float] = []
    ys: list[float] = []
    area_values: list[float] = []

    for _ in range(PLOT_SAMPLE_COUNT):
        point, area_u, _ = sample_disk_point(center, basis, radius, rng)
        delta = point - center
        xs.append(delta.dot(tangent))
        ys.append(delta.dot(bitangent))
        area_values.append(area_u)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    scatter_ax, hist_ax = axes

    scatter_ax.scatter(xs, ys, s=1, alpha=0.35, color="#157a6e", rasterized=True)
    circle = plt.Circle((0.0, 0.0), radius, fill=False, linewidth=2.0, color="#102a43")
    scatter_ax.add_patch(circle)
    scatter_ax.set_title("Точки внутри круга")
    scatter_ax.set_xlabel("координата вдоль касательной")
    scatter_ax.set_ylabel("координата вдоль бинормали")
    scatter_ax.set_aspect("equal", adjustable="box")
    scatter_ax.set_xlim(-radius * 1.08, radius * 1.08)
    scatter_ax.set_ylim(-radius * 1.08, radius * 1.08)
    style_axes(scatter_ax)

    hist_ax.hist(area_values, bins=20, range=(0.0, 1.0), density=True, color="#f4b942", edgecolor="#6b4f00")
    hist_ax.axhline(1.0, color="#8b1e3f", linestyle="--", linewidth=1.5, label="равномерная плотность")
    hist_ax.set_title("Распределение u = (r / Rc)^2")
    hist_ax.set_xlabel("u")
    hist_ax.set_ylabel("плотность")
    hist_ax.legend()
    style_axes(hist_ax)

    fig.suptitle(f"Распределение в круге ({PLOT_SAMPLE_COUNT} точек)")
    return save_figure(fig, "disk_distribution.png")


def create_uniform_sphere_plot() -> Path:
    rng = random.Random(seed_from_key("sphere-plot"))
    points: list[Vec3] = []
    z_values: list[float] = []

    for _ in range(PLOT_SAMPLE_COUNT):
        direction, z, _ = sample_uniform_sphere_direction(rng)
        points.append(direction)
        z_values.append(z)

    fig = plt.figure(figsize=(12, 5.2))
    scatter_ax = fig.add_subplot(1, 2, 1, projection="3d")
    hist_ax = fig.add_subplot(1, 2, 2)

    scatter_ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        [point.z for point in points],
        c=z_values,
        cmap="viridis",
        s=2,
        alpha=0.45,
        rasterized=True,
    )
    configure_3d_axes(scatter_ax, "Равномерные направления на сфере")

    hist_ax.hist(z_values, bins=24, range=(-1.0, 1.0), density=True, color="#3b82b0", edgecolor="#16324f")
    hist_ax.axhline(0.5, color="#8b1e3f", linestyle="--", linewidth=1.5, label="теория: p(z)=1/2")
    hist_ax.set_title("Распределение z = cos(theta)")
    hist_ax.set_xlabel("z")
    hist_ax.set_ylabel("плотность")
    hist_ax.legend()
    style_axes(hist_ax)

    fig.suptitle(f"Равномерное распределение на сфере ({PLOT_SAMPLE_COUNT} направлений)")
    return save_figure(fig, "uniform_sphere_distribution.png")


def create_cosine_plot() -> Path:
    normal = cosine_parameters()
    basis = build_basis(normal)
    rng = random.Random(seed_from_key("cosine-plot"))
    points: list[Vec3] = []
    cosines: list[float] = []

    for _ in range(PLOT_SAMPLE_COUNT):
        direction, cosine, _ = sample_cosine_direction(basis, rng)
        points.append(direction)
        cosines.append(cosine)

    fig = plt.figure(figsize=(12, 5.2))
    scatter_ax = fig.add_subplot(1, 2, 1, projection="3d")
    hist_ax = fig.add_subplot(1, 2, 2)

    scatter_ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        [point.z for point in points],
        c=cosines,
        cmap="plasma",
        s=2,
        alpha=0.45,
        rasterized=True,
    )
    scatter_ax.quiver(0.0, 0.0, 0.0, normal.x, normal.y, normal.z, length=1.1, color="#111111", linewidth=2.0)
    configure_3d_axes(scatter_ax, "Косинусное распределение на полусфере")
    scatter_ax.set_zlim(0.0, 1.05)

    hist_ax.hist(cosines, bins=24, range=(0.0, 1.0), density=True, color="#f08c2e", edgecolor="#6b3a00")
    xs = [index / 200.0 for index in range(201)]
    hist_ax.plot(xs, [2.0 * x for x in xs], color="#8b1e3f", linewidth=2.0, label="теория: p(mu)=2mu")
    hist_ax.set_title("Распределение mu = cos(theta)")
    hist_ax.set_xlabel("mu")
    hist_ax.set_ylabel("плотность")
    hist_ax.legend()
    style_axes(hist_ax)

    fig.suptitle(f"Косинусное распределение на полусфере ({PLOT_SAMPLE_COUNT} направлений)")
    return save_figure(fig, "cosine_distribution.png")


def create_all_plots() -> list[Path]:
    return [
        create_triangle_plot(),
        create_disk_plot(),
        create_uniform_sphere_plot(),
        create_cosine_plot(),
    ]


def analyze_triangle() -> str:
    rng = random.Random(seed_from_key("triangle"))
    v1, v2, v3 = triangle_parameters()

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
    expected_subtriangle = SAMPLE_COUNT / 4.0

    lines = [
        "1. Равномерное распределение точек внутри треугольника",
        f"V1 = {format_vec(v1)}, V2 = {format_vec(v2)}, V3 = {format_vec(v3)}",
        "формирование: u1,u2 ~ U[0,1], s = sqrt(u1),",
        "b1 = 1 - s, b2 = s(1-u2), b3 = s*u2, P = b1*V1 + b2*V2 + b3*V3",
        "идея доказательства: b1,b2,b3 >= 0 и b1+b2+b3 = 1, значит все точки лежат внутри треугольника; якобиан преобразования постоянен по площади.",
        build_metrics_table(
            [
                ("число сгенерированных точек", str(SAMPLE_COUNT)),
                ("число неверных точек", str(invalid_count)),
                ("проверка принадлежности", "пройдена" if invalid_count == 0 else "не пройдена"),
                ("max |b1+b2+b3-1|", f"{max_bary_sum_error:.12f}"),
                ("средние барицентрические", f"({sum_b1 / SAMPLE_COUNT:.6f}, {sum_b2 / SAMPLE_COUNT:.6f}, {sum_b3 / SAMPLE_COUNT:.6f})"),
                ("теоретический центр масс", format_vec(centroid)),
                ("выборочный центр масс", format_vec(sample_centroid)),
                ("ошибка центра масс", f"{centroid_error:.6f}"),
                ("chi^2 по подтреугольникам", f"{chi_square_stat(region_counts, expected_subtriangle):.6f}"),
                ("макс. отклонение", f"{max_relative_deviation_percent(region_counts, expected_subtriangle):.3f}%"),
                ("с.к.о. отклонение", f"{rms_relative_deviation_percent(region_counts, expected_subtriangle):.3f}%"),
            ]
        ),
        "Равновеликие подтреугольники: ожидаемое число = N / 4",
        build_count_table("подтр.", region_counts, expected_subtriangle),
        "",
    ]
    return "\n".join(lines)


def analyze_disk() -> str:
    rng = random.Random(seed_from_key("disk"))
    center, normal, radius = disk_parameters()
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
    expected_ring = SAMPLE_COUNT / len(ring_counts)
    expected_sector = SAMPLE_COUNT / len(sector_counts)

    lines = [
        "2. Равномерное распределение точек внутри круга",
        f"C = {format_vec(center)}, N = {format_vec(n)}, Rc = {radius:.6f}",
        "формирование: u1,u2 ~ U[0,1], r = Rc*sqrt(u1), phi = 2*pi*u2,",
        "P = C + T*(r*cos(phi)) + B*(r*sin(phi)) в ортонормированном базисе (T,B,N)",
        "идея доказательства: u1 = r^2 / Rc^2 распределена равномерно, поэтому равновеликие кольца должны содержать одинаковое число точек.",
        build_metrics_table(
            [
                ("число сгенерированных точек", str(SAMPLE_COUNT)),
                ("число неверных точек", str(invalid_count)),
                ("проверка принадлежности", "пройдена" if invalid_count == 0 else "не пройдена"),
                ("макс. расстояние до плоскости", f"{max_plane_distance:.12f}"),
                ("макс. r / Rc", f"{max_radius_ratio:.6f}"),
                ("теоретический центр масс", format_vec(center)),
                ("выборочный центр масс", format_vec(sample_centroid)),
                ("ошибка центра масс", f"{centroid_error:.6f}"),
                ("chi^2 по кольцам", f"{chi_square_stat(ring_counts, expected_ring):.6f}"),
                ("макс. отклонение по кольцам", f"{max_relative_deviation_percent(ring_counts, expected_ring):.3f}%"),
                ("chi^2 по секторам", f"{chi_square_stat(sector_counts, expected_sector):.6f}"),
                ("макс. отклонение по секторам", f"{max_relative_deviation_percent(sector_counts, expected_sector):.3f}%"),
            ]
        ),
        "Равновеликие кольца: ожидаемое число = N / 5",
        build_count_table("кольцо", ring_counts, expected_ring),
        "Азимутальные секторы: ожидаемое число = N / 8",
        build_count_table("сектор", sector_counts, expected_sector),
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
    expected_z = SAMPLE_COUNT / len(z_counts)
    expected_sector = SAMPLE_COUNT / len(sector_counts)

    lines = [
        "3. Равномерное распределение направлений на единичной сфере",
        "формирование: u1,u2 ~ U[0,1], z = 1 - 2*u1, phi = 2*pi*u2,",
        "w = (sqrt(1-z^2)*cos(phi), sqrt(1-z^2)*sin(phi), z)",
        "идея доказательства: z = cos(theta) равномерна на [-1,1], а phi равномерна на [0,2*pi), что даёт постоянную плотность по телесному углу.",
        build_metrics_table(
            [
                ("число сгенерированных направлений", str(SAMPLE_COUNT)),
                ("число неверных направлений", str(invalid_count)),
                ("проверка нормировки", "пройдена" if invalid_count == 0 else "не пройдена"),
                ("max ||w|-1|", f"{max_norm_error:.12f}"),
                ("среднее направление выборки", format_vec(mean_dir)),
                ("|среднее направление|", f"{mean_dir.length():.6f}"),
                ("chi^2 по z-интервалам", f"{chi_square_stat(z_counts, expected_z):.6f}"),
                ("макс. отклонение по z", f"{max_relative_deviation_percent(z_counts, expected_z):.3f}%"),
                ("chi^2 по секторам", f"{chi_square_stat(sector_counts, expected_sector):.6f}"),
                ("макс. отклонение по секторам", f"{max_relative_deviation_percent(sector_counts, expected_sector):.3f}%"),
            ]
        ),
        "Интервалы z = cos(theta) на [-1, 1]: ожидаемое число = N / 10",
        build_count_table("z-инт.", z_counts, expected_z),
        "Азимутальные секторы: ожидаемое число = N / 8",
        build_count_table("сектор", sector_counts, expected_sector),
        "",
    ]
    return "\n".join(lines)


def analyze_cosine_directions() -> str:
    rng = random.Random(seed_from_key("cosine"))
    normal = cosine_parameters()
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
    expected_u = SAMPLE_COUNT / len(u_counts)
    expected_sector = SAMPLE_COUNT / len(sector_counts)

    lines = [
        "4. Косинусное распределение направлений относительно N",
        f"N = {format_vec(normal)}",
        "формирование: сначала выбирается точка единичного круга x = sqrt(u1)*cos(phi), y = sqrt(u1)*sin(phi),",
        "z = sqrt(1-u1), w = T*x + B*y + N*z",
        "идея доказательства: получается плотность pdf(w) = cos(theta)/pi на полусфере; эквивалентно, u = cos^2(theta) распределена равномерно, а phi равномерна.",
        build_metrics_table(
            [
                ("число сгенерированных направлений", str(SAMPLE_COUNT)),
                ("число неверных направлений", str(invalid_count)),
                ("проверка полусферы", "пройдена" if invalid_count == 0 else "не пройдена"),
                ("max ||w|-1|", f"{max_norm_error:.12f}"),
                ("min dot(N, w)", f"{min_cosine:.6f}"),
                ("выборочное среднее cos(theta)", f"{mean_cosine:.6f}"),
                ("теоретическое среднее cos(theta)", f"{2.0 / 3.0:.6f}"),
                ("chi^2 по u-интервалам", f"{chi_square_stat(u_counts, expected_u):.6f}"),
                ("макс. отклонение по u", f"{max_relative_deviation_percent(u_counts, expected_u):.3f}%"),
                ("chi^2 по секторам", f"{chi_square_stat(sector_counts, expected_sector):.6f}"),
                ("макс. отклонение по секторам", f"{max_relative_deviation_percent(sector_counts, expected_sector):.3f}%"),
            ]
        ),
        "Интервалы u = cos^2(theta): ожидаемое число = N / 10",
        build_count_table("u-инт.", u_counts, expected_u),
        "Азимутальные секторы: ожидаемое число = N / 8",
        build_count_table("сектор", sector_counts, expected_sector),
        "",
    ]
    return "\n".join(lines)


def build_report() -> str:
    sections = [
        f"число выборок в каждом пункте = {SAMPLE_COUNT}, базовое зерно = {BASE_SEED}",
        "проверка = аналитические формулы + геометрические ограничения + равновероятные интервалы + статистика chi^2",
        "",
        analyze_triangle(),
        analyze_disk(),
        analyze_uniform_sphere(),
        analyze_cosine_directions(),
    ]
    return "\n".join(sections)


def main() -> None:
    print(build_report())
    saved_plots = create_all_plots()
    print("сохранённые рисунки:")
    for path in saved_plots:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
