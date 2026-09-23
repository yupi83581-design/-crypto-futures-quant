"""Deterministic parameter-perturbation robustness evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Sequence

@dataclass(frozen=True)
class RobustnessResult:
    values: tuple[float, ...]
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    stable: bool


def evaluate_robustness(*, base_parameters: Any, perturbations: Sequence[Any], evaluator: Callable[[Any], float], tolerance: float = 0.0) -> RobustnessResult:
    if not perturbations: raise ValueError("perturbations must not be empty")
    if tolerance < 0: raise ValueError("tolerance must be non-negative")
    values = tuple(float(evaluator(p)) for p in perturbations)
    if any(v != v or v in (float("inf"), float("-inf")) for v in values): raise ValueError("metrics must be finite")
    baseline = float(evaluator(base_parameters))
    if baseline != baseline or baseline in (float("inf"), float("-inf")): raise ValueError("baseline metric must be finite")
    stable = all(abs(v - baseline) <= max(abs(baseline) * tolerance, tolerance) for v in values)
    variance = sum((v - mean(values)) ** 2 for v in values) / len(values)
    return RobustnessResult(values, min(values), max(values), mean(values), variance ** 0.5, stable)
