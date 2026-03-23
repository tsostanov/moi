from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Callable


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

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> "Vec3":
        length = self.length()
        if length < EPSILON:
            raise ValueError("Невозможно нормализовать нулевой вектор.")
        return self * (1.0 / length)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class Light:
    position: Vec3
    axis: Vec3
    intensity_rgb: Vec3


@dataclass(frozen=True)
class Material:
    color_rgb: Vec3
    kd: float
    ks: float
    shininess: float


@dataclass(frozen=True)
class LightContribution:
    index: int
    distance: float
    theta_cos: float
    alpha_cos: float
    illuminance_rgb: Vec3
    brdf: float
    brightness_rgb: Vec3


@dataclass(frozen=True)
class PointCalculation:
    point: Vec3
    illuminance_total: Vec3
    brightness_total: Vec3
    contributions: list[LightContribution]


@dataclass(frozen=True)
class PreviewPoint:
    index: int
    point: Vec3
    brightness_rgb: Vec3


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def component_mul(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z)


def format_vec(vector: Vec3) -> str:
    return f"({vector.x:.4f}, {vector.y:.4f}, {vector.z:.4f})"


def format_short_vec(vector: Vec3) -> str:
    return f"({vector.x:.3g}, {vector.y:.3g}, {vector.z:.3g})"


def format_legend_vec(vector: Vec3) -> str:
    return f"{vector.x:.4f}/{vector.y:.4f}/{vector.z:.4f}"


def vec_to_hex(vector: Vec3) -> str:
    red = int(round(clamp01(vector.x) * 255))
    green = int(round(clamp01(vector.y) * 255))
    blue = int(round(clamp01(vector.z) * 255))
    return f"#{red:02x}{green:02x}{blue:02x}"


def parse_scalar(text: str, field_name: str) -> float:
    normalized = text.strip().replace(",", ".")
    if not normalized:
        raise ValueError(f"Поле '{field_name}' не заполнено.")
    try:
        return float(normalized)
    except ValueError as error:
        raise ValueError(f"Поле '{field_name}' содержит некорректное число: {text!r}") from error


def parse_lines(text: str, expected_count: int, title: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != expected_count:
            raise ValueError(
                f"{title}: строка {line_no} должна содержать {expected_count} чисел через пробел."
            )
        try:
            rows.append([float(part.replace(",", ".")) for part in parts])
        except ValueError as error:
            raise ValueError(f"{title}: строка {line_no} содержит некорректное число.") from error
    return rows


def local_to_global(p0: Vec3, p1: Vec3, p2: Vec3, x_local: float, y_local: float) -> Vec3:
    return p0 + (p1 - p0) * x_local + (p2 - p0) * y_local


def validate_local_point(x_local: float, y_local: float) -> None:
    third_weight = 1.0 - x_local - y_local
    if x_local < -EPSILON or y_local < -EPSILON or third_weight < -EPSILON:
        raise ValueError(
            "Локальная точка должна лежать внутри треугольника: x >= 0, y >= 0, x + y <= 1."
        )


def compute_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3:
    edge1 = p1 - p0
    edge2 = p2 - p0
    return edge1.cross(edge2).normalize()


def orient_normal_towards_scene(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    normal: Vec3,
    observer: Vec3,
    lights: list[Light],
) -> tuple[Vec3, bool]:
    centroid = (p0 + p1 + p2) * (1.0 / 3.0)
    alignment_score = normal.dot(observer - centroid)
    for light in lights:
        alignment_score += normal.dot(light.position - centroid)

    if alignment_score < 0.0:
        return normal * -1.0, True
    return normal, False


def choose_projection(normal: Vec3) -> tuple[str, Callable[[Vec3], tuple[float, float]]]:
    axis_weights = {
        "XY": abs(normal.z),
        "XZ": abs(normal.y),
        "YZ": abs(normal.x),
    }
    projection_name = max(axis_weights, key=axis_weights.get)
    if projection_name == "XY":
        return projection_name, lambda point: (point.x, point.y)
    if projection_name == "XZ":
        return projection_name, lambda point: (point.x, point.z)
    return projection_name, lambda point: (point.y, point.z)


def compute_point_lighting(
    point: Vec3,
    normal: Vec3,
    observer: Vec3,
    lights: list[Light],
    material: Material,
) -> PointCalculation:
    try:
        view_dir = (observer - point).normalize()
    except ValueError as error:
        raise ValueError("Точка наблюдателя совпадает с точкой на поверхности.") from error

    illuminance_total = Vec3(0.0, 0.0, 0.0)
    brightness_total = Vec3(0.0, 0.0, 0.0)
    contributions: list[LightContribution] = []

    for index, light in enumerate(lights, start=1):
        to_light = light.position - point
        distance = to_light.length()
        if distance < EPSILON:
            raise ValueError(f"Точка поверхности совпадает с источником света L{index}.")

        light_dir = to_light * (1.0 / distance)
        emission_dir = (point - light.position).normalize()
        axis_dir = light.axis.normalize()

        theta_cos = clamp01(axis_dir.dot(emission_dir))
        alpha_cos = clamp01(normal.dot(light_dir))

        intensity_rgb = light.intensity_rgb * theta_cos
        illuminance_rgb = intensity_rgb * (alpha_cos / (distance * distance))

        half_vector_raw = light_dir + view_dir
        if half_vector_raw.length() < EPSILON:
            half_vector = normal
        else:
            half_vector = half_vector_raw.normalize()

        specular = max(0.0, normal.dot(half_vector)) ** material.shininess
        brdf = material.kd + material.ks * specular
        brightness_rgb = component_mul(illuminance_rgb, material.color_rgb) * brdf

        illuminance_total = illuminance_total + illuminance_rgb
        brightness_total = brightness_total + brightness_rgb
        contributions.append(
            LightContribution(
                index=index,
                distance=distance,
                theta_cos=theta_cos,
                alpha_cos=alpha_cos,
                illuminance_rgb=illuminance_rgb,
                brdf=brdf,
                brightness_rgb=brightness_rgb,
            )
        )

    return PointCalculation(
        point=point,
        illuminance_total=illuminance_total,
        brightness_total=brightness_total,
        contributions=contributions,
    )


class VectorEntry(ttk.Frame):
    def __init__(self, master: tk.Misc, labels: tuple[str, str, str], defaults: tuple[str, str, str]) -> None:
        super().__init__(master)
        self.entries: list[ttk.Entry] = []
        for column, (label, default) in enumerate(zip(labels, defaults)):
            ttk.Label(self, text=label).grid(row=0, column=column * 2, padx=(0, 4), pady=2, sticky="w")
            entry = ttk.Entry(self, width=10)
            entry.insert(0, default)
            entry.grid(row=0, column=column * 2 + 1, padx=(0, 10), pady=2, sticky="ew")
            self.entries.append(entry)

    def get_vec3(self, prefix: str) -> Vec3:
        values = [
            parse_scalar(entry.get(), f"{prefix} {axis_name}")
            for entry, axis_name in zip(self.entries, ("x", "y", "z"))
        ]
        return Vec3(*values)

    def set_values(self, values: tuple[str, str, str]) -> None:
        for entry, value in zip(self.entries, values):
            entry.delete(0, tk.END)
            entry.insert(0, value)


class LightInput(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        position_defaults: tuple[str, str, str],
        axis_defaults: tuple[str, str, str],
        intensity_defaults: tuple[str, str, str],
    ) -> None:
        super().__init__(master, text=title, padding=10)
        self.position = VectorEntry(self, ("Px", "Py", "Pz"), position_defaults)
        self.axis = VectorEntry(self, ("Ox", "Oy", "Oz"), axis_defaults)
        self.intensity = VectorEntry(self, ("Ir", "Ig", "Ib"), intensity_defaults)

        ttk.Label(self, text="Положение").grid(row=0, column=0, sticky="w")
        self.position.grid(row=1, column=0, sticky="ew")
        ttk.Label(self, text="Ось источника").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.axis.grid(row=3, column=0, sticky="ew")
        ttk.Label(self, text="I0 RGB").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.intensity.grid(row=5, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

    def get_light(self) -> Light:
        position = self.position.get_vec3(f"{self['text']} положение")
        axis = self.axis.get_vec3(f"{self['text']} ось")
        intensity_rgb = self.intensity.get_vec3(f"{self['text']} интенсивность")
        return Light(position=position, axis=axis, intensity_rgb=intensity_rgb)

    def set_title(self, title: str) -> None:
        self.configure(text=title)


class App:
    DEFAULT_LIGHTS = [
        (("-2", "6", "5"), ("0.5", "-1", "-1"), ("1.0", "0.9", "0.8")),
        (("4", "4", "6"), ("-0.3", "-0.8", "-1"), ("0.6", "0.8", "1.0")),
        (("0", "7", "4"), ("0", "-1", "-0.5"), ("0.8", "0.7", "0.7")),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Расчет яркости точки на поверхности")
        self.root.geometry("1320x900")
        self.root.minsize(1160, 780)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        container = ttk.Frame(root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        left = ttk.Frame(container)
        right = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_triangle_section(left)
        self._build_material_section(left)
        self._build_lights_section(left)
        self._build_points_section(left)
        self._build_actions(left)
        self._build_preview(right)
        self._build_results(right)

    def _build_triangle_section(self, master: ttk.Frame) -> None:
        frame = ttk.LabelFrame(master, text="Треугольник", padding=10)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text="P0").grid(row=0, column=0, sticky="w")
        self.p0_entry = VectorEntry(frame, ("x", "y", "z"), ("0", "0", "0"))
        self.p0_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(frame, text="P1").grid(row=1, column=0, sticky="w")
        self.p1_entry = VectorEntry(frame, ("x", "y", "z"), ("3", "1", "0"))
        self.p1_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(frame, text="P2").grid(row=2, column=0, sticky="w")
        self.p2_entry = VectorEntry(frame, ("x", "y", "z"), ("1", "-2", "0"))
        self.p2_entry.grid(row=2, column=1, sticky="ew", pady=2)

        frame.columnconfigure(1, weight=1)

    def _build_material_section(self, master: ttk.Frame) -> None:
        frame = ttk.LabelFrame(master, text="Наблюдение и материал", padding=10)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text="Наблюдатель V").grid(row=0, column=0, sticky="w")
        self.observer_entry = VectorEntry(frame, ("x", "y", "z"), ("5", "6", "4"))
        self.observer_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(frame, text="Цвет поверхности RGB (0..1)").grid(row=1, column=0, sticky="w")
        self.color_entry = VectorEntry(frame, ("r", "g", "b"), ("0.8", "0.7", "0.6"))
        self.color_entry.grid(row=1, column=1, sticky="ew", pady=2)

        scalar_frame = ttk.Frame(frame)
        scalar_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(scalar_frame, text="kd").grid(row=0, column=0, sticky="w")
        self.kd_entry = ttk.Entry(scalar_frame, width=10)
        self.kd_entry.insert(0, "0.7")
        self.kd_entry.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(scalar_frame, text="ks").grid(row=0, column=2, sticky="w")
        self.ks_entry = ttk.Entry(scalar_frame, width=10)
        self.ks_entry.insert(0, "0.3")
        self.ks_entry.grid(row=0, column=3, padx=(4, 12))

        ttk.Label(scalar_frame, text="Блеск n").grid(row=0, column=4, sticky="w")
        self.shininess_entry = ttk.Entry(scalar_frame, width=10)
        self.shininess_entry.insert(0, "12")
        self.shininess_entry.grid(row=0, column=5, padx=(4, 0))

        ttk.Label(
            frame,
            text="Формат точек: вводите локальные строки 'x y', глобальные координаты строятся автоматически.",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        frame.columnconfigure(1, weight=1)

    def _build_lights_section(self, master: ttk.Frame) -> None:
        wrapper = ttk.LabelFrame(master, text="Источники света", padding=10)
        wrapper.pack(fill="x", pady=(0, 10))

        toolbar = ttk.Frame(wrapper)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="Добавить источник", command=self._add_light_input).pack(side="left")
        self.remove_light_button = ttk.Button(
            toolbar,
            text="Удалить последний",
            command=self._remove_light_input,
        )
        self.remove_light_button.pack(side="left", padx=(8, 0))

        lights_viewport = ttk.Frame(wrapper)
        lights_viewport.grid(row=1, column=0, columnspan=2, sticky="ew")
        lights_viewport.columnconfigure(0, weight=1)
        lights_viewport.rowconfigure(0, weight=1)

        self.lights_canvas = tk.Canvas(lights_viewport, height=220, highlightthickness=0, borderwidth=0)
        self.lights_canvas.grid(row=0, column=0, sticky="ew")

        lights_scrollbar = ttk.Scrollbar(lights_viewport, orient="vertical", command=self.lights_canvas.yview)
        lights_scrollbar.grid(row=0, column=1, sticky="ns")
        self.lights_canvas.configure(yscrollcommand=lights_scrollbar.set)

        self.lights_container = ttk.Frame(self.lights_canvas)
        self.lights_canvas_window = self.lights_canvas.create_window((0, 0), window=self.lights_container, anchor="nw")
        self.lights_container.bind("<Configure>", self._update_lights_scrollregion)
        self.lights_canvas.bind("<Configure>", self._resize_lights_canvas_window)
        self.lights_canvas.bind("<Enter>", self._bind_lights_mousewheel)
        self.lights_canvas.bind("<Leave>", self._unbind_lights_mousewheel)
        self.light_inputs: list[LightInput] = []

        for position_defaults, axis_defaults, intensity_defaults in self.DEFAULT_LIGHTS:
            self._add_light_input(position_defaults, axis_defaults, intensity_defaults)

        self._update_light_controls()
        wrapper.columnconfigure(0, weight=1)
        wrapper.columnconfigure(1, weight=1)

    def _add_light_input(
        self,
        position_defaults: tuple[str, str, str] | None = None,
        axis_defaults: tuple[str, str, str] | None = None,
        intensity_defaults: tuple[str, str, str] | None = None,
    ) -> None:
        light_index = len(self.light_inputs)
        if position_defaults is None or axis_defaults is None or intensity_defaults is None:
            if light_index < len(self.DEFAULT_LIGHTS):
                position_defaults, axis_defaults, intensity_defaults = self.DEFAULT_LIGHTS[light_index]
            else:
                position_defaults = ("0", str(6 + light_index), str(4 + light_index))
                axis_defaults = ("0", "-1", "-1")
                intensity_defaults = ("1.0", "1.0", "1.0")

        light_input = LightInput(
            self.lights_container,
            f"L{light_index + 1}",
            position_defaults,
            axis_defaults,
            intensity_defaults,
        )
        self.light_inputs.append(light_input)
        self._relayout_light_inputs()
        self._update_light_controls()

    def _remove_light_input(self) -> None:
        if not self.light_inputs:
            return
        light_input = self.light_inputs.pop()
        light_input.destroy()
        self._relayout_light_inputs()
        self._update_light_controls()

    def _relayout_light_inputs(self) -> None:
        self.lights_container.columnconfigure(0, weight=1)
        self.lights_container.columnconfigure(1, weight=0)

        for index, light_input in enumerate(self.light_inputs):
            light_input.set_title(f"L{index + 1}")
            light_input.grid(
                row=index,
                column=0,
                padx=5,
                pady=5,
                sticky="ew",
            )

        self._update_lights_scrollregion()

    def _update_light_controls(self) -> None:
        if self.light_inputs:
            self.remove_light_button.state(["!disabled"])
        else:
            self.remove_light_button.state(["disabled"])

    def _update_lights_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.lights_canvas.configure(scrollregion=self.lights_canvas.bbox("all"))

    def _resize_lights_canvas_window(self, event: tk.Event) -> None:
        self.lights_canvas.itemconfigure(self.lights_canvas_window, width=event.width)

    def _bind_lights_mousewheel(self, _event: tk.Event) -> None:
        self.lights_canvas.bind_all("<MouseWheel>", self._on_lights_mousewheel)

    def _unbind_lights_mousewheel(self, _event: tk.Event) -> None:
        self.lights_canvas.unbind_all("<MouseWheel>")

    def _on_lights_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self.lights_canvas.yview_scroll(-int(delta / 120), "units")

    def _build_points_section(self, master: ttk.Frame) -> None:
        frame = ttk.LabelFrame(master, text="Точки для расчета", padding=10)
        frame.pack(fill="both", expand=False, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Локальные координаты (x y)").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Глобальные координаты тех же точек").grid(row=0, column=1, sticky="w")

        self.local_points_text = tk.Text(frame, width=28, height=8, wrap="none")
        self.local_points_text.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(4, 0))
        self.local_points_text.insert("1.0", "0.20 0.30\n0.45 0.10\n0.15 0.60")

        self.generated_global_text = tk.Text(frame, width=28, height=8, wrap="none", state="disabled")
        self.generated_global_text.grid(row=1, column=1, sticky="nsew", pady=(4, 0))

    def _build_actions(self, master: ttk.Frame) -> None:
        frame = ttk.Frame(master)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Button(frame, text="Рассчитать", command=self.calculate).pack(side="left")
        ttk.Button(frame, text="Очистить результаты", command=self.clear_results).pack(side="left", padx=(8, 0))

    def _build_preview(self, master: ttk.Frame) -> None:
        frame = ttk.LabelFrame(master, text="Карта яркости", padding=10)
        frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        self.preview_info_var = tk.StringVar(value="После расчета здесь появится треугольник с яркостями в точках.")
        ttk.Label(frame, textvariable=self.preview_info_var).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.preview_canvas = tk.Canvas(
            frame,
            width=620,
            height=330,
            bg="#f7fbff",
            highlightthickness=1,
            highlightbackground="#c7d4df",
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        self._draw_empty_preview()

    def _build_results(self, master: ttk.Frame) -> None:
        ttk.Label(master, text="Результаты").grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.results_text = tk.Text(master, wrap="word", font=("Consolas", 10))
        self.results_text.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(master, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)

    def clear_results(self) -> None:
        self.results_text.delete("1.0", tk.END)
        self.generated_global_text.configure(state="normal")
        self.generated_global_text.delete("1.0", tk.END)
        self.generated_global_text.configure(state="disabled")
        self.preview_info_var.set("После расчета здесь появится треугольник с яркостями в точках.")
        self._draw_empty_preview()

    def _set_generated_global_points(self, points: list[Vec3]) -> None:
        self.generated_global_text.configure(state="normal")
        self.generated_global_text.delete("1.0", tk.END)
        if points:
            lines = [f"{point.x:.4f} {point.y:.4f} {point.z:.4f}" for point in points]
            self.generated_global_text.insert("1.0", "\n".join(lines))
        self.generated_global_text.configure(state="disabled")

    def _draw_empty_preview(self) -> None:
        self.preview_canvas.delete("all")
        width = int(self.preview_canvas.cget("width"))
        height = int(self.preview_canvas.cget("height"))
        self.preview_canvas.create_text(
            width / 2,
            height / 2,
            text="После расчета здесь будет карта яркости для точек треугольника",
            fill="#506070",
            font=("Segoe UI", 12),
        )

    def _draw_preview(
        self,
        p0: Vec3,
        p1: Vec3,
        p2: Vec3,
        normal: Vec3,
        preview_points: list[PreviewPoint],
    ) -> None:
        projection_name, projector = choose_projection(normal)
        max_component = 0.0
        for preview_point in preview_points:
            max_component = max(
                max_component,
                preview_point.brightness_rgb.x,
                preview_point.brightness_rgb.y,
                preview_point.brightness_rgb.z,
            )

        if max_component < EPSILON:
            self.preview_info_var.set(
                f"Проекция на плоскость {projection_name}. Все расчетные точки имеют нулевую яркость."
            )
        else:
            self.preview_info_var.set(
                f"Проекция на плоскость {projection_name}. Цвет точки показывает L_total с нормировкой по максимуму {max_component:.4f}."
            )

        projected_points: list[tuple[float, float]] = [
            projector(p0),
            projector(p1),
            projector(p2),
        ]
        projected_points.extend(projector(preview_point.point) for preview_point in preview_points)

        xs = [point[0] for point in projected_points]
        ys = [point[1] for point in projected_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = int(self.preview_canvas.cget("width"))
        height = int(self.preview_canvas.cget("height"))
        margin = 40
        legend_width = 240
        plot_width = max(220, width - legend_width - 2 * margin)
        plot_height = max(160, height - 2 * margin)
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)
        scale = min(plot_width / range_x, plot_height / range_y)
        plot_origin_x = margin
        plot_origin_y = margin
        scaled_width = range_x * scale
        scaled_height = range_y * scale
        x_offset = plot_origin_x + (plot_width - scaled_width) / 2
        y_offset = plot_origin_y + (plot_height - scaled_height) / 2

        def to_canvas(point: Vec3) -> tuple[float, float]:
            px, py = projector(point)
            canvas_x = x_offset + (px - min_x) * scale
            canvas_y = plot_origin_y + scaled_height - (py - min_y) * scale + (plot_height - scaled_height) / 2
            return canvas_x, canvas_y

        self.preview_canvas.delete("all")

        self.preview_canvas.create_rectangle(
            plot_origin_x - 12,
            plot_origin_y - 12,
            plot_origin_x + plot_width + 12,
            plot_origin_y + plot_height + 12,
            outline="#d5e0ea",
            width=1,
        )

        triangle_coords = [*to_canvas(p0), *to_canvas(p1), *to_canvas(p2)]
        self.preview_canvas.create_polygon(
            triangle_coords,
            fill="#dbe9ff",
            outline="#1c427e",
            width=2,
        )

        for label, point in (("P0", p0), ("P1", p1), ("P2", p2)):
            x, y = to_canvas(point)
            self.preview_canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#0b1c32", outline="")
            self.preview_canvas.create_text(x + 18, y - 12, text=label, fill="#10233f", font=("Segoe UI", 10, "bold"))

        for preview_point in preview_points:
            x, y = to_canvas(preview_point.point)
            if max_component < EPSILON:
                point_color = "#9aa5b1"
            else:
                point_color = vec_to_hex(preview_point.brightness_rgb * (1.0 / max_component))
            self.preview_canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=point_color, outline="#243447", width=1)
            self.preview_canvas.create_text(
                x + 16,
                y - 12,
                text=f"T{preview_point.index}",
                fill="#243447",
                font=("Segoe UI", 9, "bold"),
            )

        legend_left = plot_origin_x + plot_width + 24
        legend_top = 18
        legend_text_width = max(120, width - legend_left - 18)
        self.preview_canvas.create_text(
            legend_left,
            legend_top,
            anchor="nw",
            text="Яркость в точках",
            fill="#243447",
            font=("Segoe UI", 10, "bold"),
        )

        for row_index, preview_point in enumerate(preview_points):
            row_y = legend_top + 28 + row_index * 26
            if max_component < EPSILON:
                point_color = "#9aa5b1"
            else:
                point_color = vec_to_hex(preview_point.brightness_rgb * (1.0 / max_component))
            self.preview_canvas.create_rectangle(
                legend_left,
                row_y,
                legend_left + 14,
                row_y + 14,
                fill=point_color,
                outline="#243447",
            )
            self.preview_canvas.create_text(
                legend_left + 22,
                row_y + 7,
                anchor="w",
                text=f"T{preview_point.index}: L={format_legend_vec(preview_point.brightness_rgb)}",
                fill="#243447",
                font=("Consolas", 8),
                width=legend_text_width - 22,
            )

        self.preview_canvas.create_text(
            14,
            height - 14,
            anchor="w",
            text="Цвет точки показывает относительную величину L_total; точные значения перечислены справа.",
            fill="#425466",
            font=("Segoe UI", 9),
        )

    def calculate(self) -> None:
        try:
            p0 = self.p0_entry.get_vec3("P0")
            p1 = self.p1_entry.get_vec3("P1")
            p2 = self.p2_entry.get_vec3("P2")
            observer = self.observer_entry.get_vec3("Наблюдатель V")
            color_rgb = self.color_entry.get_vec3("Цвет поверхности")
            material = Material(
                color_rgb=color_rgb,
                kd=parse_scalar(self.kd_entry.get(), "kd"),
                ks=parse_scalar(self.ks_entry.get(), "ks"),
                shininess=parse_scalar(self.shininess_entry.get(), "Блеск n"),
            )

            lights = [light_input.get_light() for light_input in self.light_inputs]
            if not lights:
                raise ValueError("Нужно задать хотя бы один источник света.")

            geometric_normal = compute_normal(p0, p1, p2)
            normal, normal_was_flipped = orient_normal_towards_scene(
                p0,
                p1,
                p2,
                geometric_normal,
                observer,
                lights,
            )
            local_points = parse_lines(self.local_points_text.get("1.0", tk.END), 2, "Локальные точки")

            output_lines = [
                "Расчет яркости точки на поверхности",
                "",
                f"Геометрическая нормаль N0 = {format_vec(geometric_normal)}",
                f"Рабочая нормаль N = {format_vec(normal)}",
                f"Наблюдатель V = {format_vec(observer)}",
                f"Цвет поверхности = {format_vec(material.color_rgb)}",
                f"kd = {material.kd:.4f}, ks = {material.ks:.4f}, n = {material.shininess:.4f}",
                "Диаграмма излучения принята в упрощенном виде: f(theta) = max(0, cos(theta)).",
                "",
            ]

            if normal_was_flipped:
                output_lines.extend(
                    [
                        "Нормаль была автоматически развернута к наблюдателю и источникам света.",
                        "",
                    ]
                )

            point_rows: list[tuple[int, float, float, Vec3, PointCalculation]] = []
            for idx, (x_local, y_local) in enumerate(local_points, start=1):
                validate_local_point(x_local, y_local)
                global_point = local_to_global(p0, p1, p2, x_local, y_local)
                result = compute_point_lighting(global_point, normal, observer, lights, material)
                point_rows.append((idx, x_local, y_local, global_point, result))

            if point_rows:
                output_lines.append("Освещенность для точек в локальных координатах")
                output_lines.append("-" * 70)
                for idx, x_local, y_local, global_point, result in point_rows:
                    output_lines.append(
                        f"{idx}. local=({x_local:.4f}, {y_local:.4f}) -> P={format_vec(global_point)}"
                    )
                    output_lines.append(f"   E_total = {format_vec(result.illuminance_total)}")
                output_lines.append("")

                output_lines.append("Освещенность для тех же точек в глобальных координатах")
                output_lines.append("-" * 70)
                for idx, _, _, global_point, result in point_rows:
                    output_lines.append(f"{idx}. P = {format_vec(global_point)}")
                    output_lines.append(f"   E_total = {format_vec(result.illuminance_total)}")
                output_lines.append("")

                output_lines.append("Яркость для тех же точек")
                output_lines.append("-" * 70)
                for idx, x_local, y_local, _, result in point_rows:
                    output_lines.extend(
                        self._format_result_block(idx, f"local=({x_local:.4f}, {y_local:.4f})", result)
                    )
                output_lines.append("")

            if not point_rows:
                output_lines.append("Нет точек для расчета.")

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", "\n".join(output_lines).rstrip() + "\n")
            self._set_generated_global_points([global_point for _, _, _, global_point, _ in point_rows])
            self._draw_preview(
                p0,
                p1,
                p2,
                normal,
                [
                    PreviewPoint(index=idx, point=global_point, brightness_rgb=result.brightness_total)
                    for idx, _, _, global_point, result in point_rows
                ],
            )
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error))

    def _format_result_block(self, index: int, label: str, result: PointCalculation) -> list[str]:
        lines = [
            f"{index}. {label}",
            f"   P = {format_vec(result.point)}",
            f"   E_total = {format_vec(result.illuminance_total)}",
            f"   L_total = {format_vec(result.brightness_total)}",
        ]
        for contribution in result.contributions:
            lines.append(
                "   "
                f"L{contribution.index}: R={contribution.distance:.4f}, "
                f"cos(theta)={contribution.theta_cos:.4f}, "
                f"cos(alpha)={contribution.alpha_cos:.4f}, "
                f"BRDF={contribution.brdf:.4f}"
            )
            lines.append(f"      E = {format_vec(contribution.illuminance_rgb)}")
            lines.append(f"      L = {format_vec(contribution.brightness_rgb)}")
        return lines


def main() -> None:
    root = tk.Tk()
    app = App(root)
    app.calculate()
    root.mainloop()


if __name__ == "__main__":
    main()
