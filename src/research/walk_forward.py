"""Leakage-safe deterministic walk-forward evaluation primitives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence

@dataclass(frozen=True)
class WalkForwardFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    predictions: tuple[float, ...]
    actual: tuple[int, ...]

@dataclass(frozen=True)
class WalkForwardResult:
    folds: tuple[WalkForwardFold, ...]
    predictions: tuple[float, ...]
    actual: tuple[int, ...]
    test_indices: tuple[int, ...]


def run_walk_forward(*, observations: Sequence[Any], actual: Sequence[int], model_factory: Callable[[], Any], feature_builder: Callable[[Sequence[Any]], Any] | None = None, train_size: int, test_size: int, step: int | None = None, expanding: bool = False, calibrator_factory: Callable[[], Any] | None = None) -> WalkForwardResult:
    """Fit and evaluate chronological folds without future-data leakage.

    Model and optional calibrator are freshly created per fold. Any feature
    builder receives only the fold's train/test observations, never future
    observations. ``model.fit`` receives train data only; test data is passed
    only to prediction. A calibrator, when supplied, is fit only on train
    predictions/labels and transforms test probabilities only.
    """
    n = len(observations)
    if len(actual) != n: raise ValueError("observations and actual must have the same length")
    if train_size < 1 or test_size < 1: raise ValueError("train_size and test_size must be positive")
    if step is None: step = test_size
    if step < 1: raise ValueError("step must be positive")
    if n < train_size + test_size: raise ValueError("insufficient observations for one walk-forward fold")
    if expanding is False and train_size > n: raise ValueError("train_size exceeds observations")

    folds: list[WalkForwardFold] = []
    all_predictions: list[float] = []
    all_actual: list[int] = []
    all_indices: list[int] = []
    test_start = train_size
    while test_start + test_size <= n:
        train_start = 0 if expanding else test_start - train_size
        train_end = test_start
        test_end = test_start + test_size
        train_obs = observations[train_start:train_end]
        test_obs = observations[test_start:test_end]
        train_y = tuple(int(x) for x in actual[train_start:train_end])
        test_y = tuple(int(x) for x in actual[test_start:test_end])
        if not all(x in (0, 1) for x in train_y + test_y): raise ValueError("actual outcomes must be binary 0 or 1")
        model = model_factory()
        if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must provide fit and predict_proba")
        train_x = feature_builder(train_obs) if feature_builder else train_obs
        test_x = feature_builder(test_obs) if feature_builder else test_obs
        model.fit(train_x, train_y)
        train_prob = tuple(float(x) for x in model.predict_proba(train_x))
        if len(train_prob) != len(train_y): raise ValueError("model training probabilities must match train observations")
        calibrator = calibrator_factory() if calibrator_factory else None
        if calibrator is not None:
            if not callable(getattr(calibrator, "fit", None)) or not callable(getattr(calibrator, "predict", None)):
                raise TypeError("calibrator must provide fit and predict")
            calibrator.fit(train_prob, train_y)
        raw_test = tuple(float(x) for x in model.predict_proba(test_x))
        if len(raw_test) != len(test_y): raise ValueError("model test probabilities must match test observations")
        test_prob = tuple(float(x) for x in calibrator.predict(raw_test)) if calibrator else raw_test
        if len(test_prob) != len(test_y): raise ValueError("calibrated probabilities must match test observations")
        if any(not 0.0 <= p <= 1.0 for p in test_prob): raise ValueError("probabilities must be between 0 and 1")
        fold = WalkForwardFold(train_start, train_end, test_start, test_end, test_prob, test_y)
        folds.append(fold); all_predictions.extend(test_prob); all_actual.extend(test_y); all_indices.extend(range(test_start, test_end))
        test_start += step
    return WalkForwardResult(tuple(folds), tuple(all_predictions), tuple(all_actual), tuple(all_indices))
