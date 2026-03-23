from __future__ import annotations

import math
import random
from dataclasses import dataclass


A = 2.0
B = 5.0
TRUE_INTEGRAL = (B**3 - A**3) / 3.0
SAMPLE_SIZES = (100, 1000, 10000, 100000)
BASE_SEED = 20260323


def f(x: float) -> float:
    return x * x


def delta_estimate(n: int) -> float:
    return TRUE_INTEGRAL / math.sqrt(n)


def uniform_sample(rng: random.Random) -> float:
    return A + (B - A) * rng.random()


def seed_from_key(key: str, n: int) -> int:
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(key))
    return BASE_SEED + checksum + n * 17


def allocate_counts(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


@dataclass(frozen=True)
class Density:
    name: str
    power: int
    normalized_formula: str

    @property
    def exponent(self) -> int:
        return self.power + 1

    @property
    def coefficient(self) -> float:
        return self.exponent / (B**self.exponent - A**self.exponent)

    def pdf(self, x: float) -> float:
        return self.coefficient * (x**self.power)

    def sample(self, rng: random.Random) -> float:
        left = A**self.exponent
        right = B**self.exponent
        return (left + (right - left) * rng.random()) ** (1.0 / self.exponent)


@dataclass(frozen=True)
class MethodResult:
    method_name: str
    n: int
    estimate: float

    @property
    def abs_error(self) -> float:
        return abs(self.estimate - TRUE_INTEGRAL)

    @property
    def delta(self) -> float:
        return delta_estimate(self.n)


def simple_monte_carlo(n: int, rng: random.Random) -> float:
    total = 0.0
    width = B - A
    for _ in range(n):
        total += f(uniform_sample(rng))
    return width * total / n


def make_strata(step: float) -> list[tuple[float, float]]:
    edges = [A]
    while edges[-1] < B:
        edges.append(min(B, edges[-1] + step))
    return list(zip(edges[:-1], edges[1:]))


def stratified_monte_carlo(n: int, step: float, rng: random.Random) -> float:
    strata = make_strata(step)
    counts = allocate_counts(n, len(strata))
    estimate = 0.0

    for (left, right), local_count in zip(strata, counts):
        width = right - left
        local_sum = 0.0
        for _ in range(local_count):
            x = left + width * rng.random()
            local_sum += f(x)
        estimate += width * local_sum / local_count

    return estimate


def importance_sampling(n: int, density: Density, rng: random.Random) -> float:
    total = 0.0
    for _ in range(n):
        x = density.sample(rng)
        total += f(x) / density.pdf(x)
    return total / n


def balance_weight(x: float, own: Density, other: Density) -> float:
    own_pdf = own.pdf(x)
    other_pdf = other.pdf(x)
    return own_pdf / (own_pdf + other_pdf)


def power_weight(x: float, own: Density, other: Density) -> float:
    own_pdf = own.pdf(x)
    other_pdf = other.pdf(x)
    own_sq = own_pdf * own_pdf
    other_sq = other_pdf * other_pdf
    return own_sq / (own_sq + other_sq)


def multiple_importance_sampling(
    n: int,
    density_1: Density,
    density_2: Density,
    weight_kind: str,
    rng: random.Random,
) -> float:
    counts = allocate_counts(n, 2)
    n1, n2 = counts
    weight_fn = balance_weight if weight_kind == "balance" else power_weight

    first_sum = 0.0
    for _ in range(n1):
        x = density_1.sample(rng)
        first_sum += weight_fn(x, density_1, density_2) * f(x) / density_1.pdf(x)

    second_sum = 0.0
    for _ in range(n2):
        x = density_2.sample(rng)
        second_sum += weight_fn(x, density_2, density_1) * f(x) / density_2.pdf(x)

    return first_sum / n1 + second_sum / n2


def russian_roulette_monte_carlo(n: int, survival_probability: float, rng: random.Random) -> float:
    width = B - A
    total = 0.0
    for _ in range(n):
        x = uniform_sample(rng)
        contribution = width * f(x)
        if rng.random() < survival_probability:
            total += contribution / survival_probability
    return total / n


def collect_results() -> dict[str, list[MethodResult]]:
    densities = {
        "p1(x) = x": Density("p1(x) = x", 1, "q1(x) = 2x / 21"),
        "p2(x) = x^2": Density("p2(x) = x^2", 2, "q2(x) = x^2 / 39"),
        "p3(x) = x^3": Density("p3(x) = x^3", 3, "q3(x) = 4x^3 / 609"),
    }

    results: dict[str, list[MethodResult]] = {}

    methods = [
        ("Простой Монте-Карло", lambda n, rng: simple_monte_carlo(n, rng)),
        ("Стратификация, шаг 1.0", lambda n, rng: stratified_monte_carlo(n, 1.0, rng)),
        ("Стратификация, шаг 0.5", lambda n, rng: stratified_monte_carlo(n, 0.5, rng)),
        (
            "Выборка по значимости, p1(x) = x",
            lambda n, rng: importance_sampling(n, densities["p1(x) = x"], rng),
        ),
        (
            "Выборка по значимости, p2(x) = x^2",
            lambda n, rng: importance_sampling(n, densities["p2(x) = x^2"], rng),
        ),
        (
            "Выборка по значимости, p3(x) = x^3",
            lambda n, rng: importance_sampling(n, densities["p3(x) = x^3"], rng),
        ),
        (
            "MIS, средняя плотность",
            lambda n, rng: multiple_importance_sampling(
                n,
                densities["p1(x) = x"],
                densities["p3(x) = x^3"],
                "balance",
                rng,
            ),
        ),
        (
            "MIS, средний квадрат плотностей",
            lambda n, rng: multiple_importance_sampling(
                n,
                densities["p1(x) = x"],
                densities["p3(x) = x^3"],
                "power",
                rng,
            ),
        ),
        (
            "Русская рулетка, R = 0.5",
            lambda n, rng: russian_roulette_monte_carlo(n, 0.5, rng),
        ),
        (
            "Русская рулетка, R = 0.75",
            lambda n, rng: russian_roulette_monte_carlo(n, 0.75, rng),
        ),
        (
            "Русская рулетка, R = 0.95",
            lambda n, rng: russian_roulette_monte_carlo(n, 0.95, rng),
        ),
    ]

    for method_name, runner in methods:
        method_results: list[MethodResult] = []
        for n in SAMPLE_SIZES:
            rng = random.Random(seed_from_key(method_name, n))
            method_results.append(MethodResult(method_name, n, runner(n, rng)))
        results[method_name] = method_results

    return results


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def border(separator: str = "+", fill: str = "-") -> str:
        return separator + separator.join(fill * (width + 2) for width in widths) + separator

    def format_row(row: list[str]) -> str:
        cells: list[str] = []
        for index, cell in enumerate(row):
            if index == 0:
                cells.append(f" {cell:>{widths[index]}} ")
            else:
                cells.append(f" {cell:>{widths[index]}} ")
        return "|" + "|".join(cells) + "|"

    lines = [border(), format_row(headers), border()]
    for row in rows:
        lines.append(format_row(row))
    lines.append(border())
    return "\n".join(lines)


def build_method_table(rows: list[MethodResult]) -> str:
    table_rows = [
        [
            str(row.n),
            f"{TRUE_INTEGRAL:.6f}",
            f"{row.estimate:.6f}",
            f"{row.abs_error:.6f}",
            f"{row.delta:.6f}",
        ]
        for row in rows
    ]
    return format_table(
        ["N", "I_true", "I_MC", "|I_MC - I_true|", "Delta I"],
        table_rows,
    )


def build_console_report(results: dict[str, list[MethodResult]]) -> str:
    sections: list[str] = []
    sections.append("LAB 2: MONTE CARLO INTEGRATION")
    sections.append(f"f(x) = x^2, interval = [{A:.0f}, {B:.0f}], I_true = {TRUE_INTEGRAL:.6f}")
    sections.append(f"N = {', '.join(str(n) for n in SAMPLE_SIZES)}, seed = {BASE_SEED}")
    sections.append("importance pdfs: q1(x)=2x/21, q2(x)=x^2/39, q3(x)=4x^3/609")
    sections.append("")

    for method_name, rows in results.items():
        sections.append(method_name)
        sections.append(build_method_table(rows))
        sections.append("")

    return "\n".join(sections)


def main() -> None:
    results = collect_results()
    report = build_console_report(results)
    print(report)


if __name__ == "__main__":
    main()
