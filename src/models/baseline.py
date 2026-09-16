"""Deterministic baseline models for quantitative research."""

from __future__ import annotations
import math
from typing import Sequence

class MajorityClassBaseline:
    """Predicts the class observed most often during training."""
    def __init__(self) -> None: self._majority_class: int | None = None
    def fit(self, labels: Sequence[int]) -> "MajorityClassBaseline":
        validated=_validate_labels(labels)
        if not validated: raise ValueError("labels must not be empty")
        ones=sum(validated); zeros=len(validated)-ones; self._majority_class=1 if ones>zeros else 0; return self
    def predict(self, count: int | Sequence[float]) -> list[int]:
        self._require_fitted(); return [self._majority_class] * _prediction_count(count)  # type: ignore[list-item]
    def predict_proba(self, features: int | Sequence[float]) -> list[float]:
        self._require_fitted(); return [float(self._majority_class)] * _prediction_count(features)
    def _require_fitted(self) -> None:
        if self._majority_class is None: raise RuntimeError("model must be fitted before prediction")

class LogisticRegressionBaseline:
    """Small deterministic one-feature logistic regression model."""
    def __init__(self, learning_rate: float=0.05, iterations: int=5000, regularization: float=0.01) -> None:
        if learning_rate<=0: raise ValueError("learning_rate must be positive")
        if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1: raise ValueError("iterations must be a positive integer")
        if regularization<0: raise ValueError("regularization must be non-negative")
        self.learning_rate=float(learning_rate); self.iterations=iterations; self.regularization=float(regularization); self._weight=None; self._bias=None; self._feature_mean=None; self._feature_scale=None
    def fit(self, features: Sequence[float], labels: Sequence[int]) -> "LogisticRegressionBaseline":
        x=_validate_features(features); y=_validate_labels(labels)
        if len(x)!=len(y): raise ValueError("features and labels must contain the same number of observations")
        if not x: raise ValueError("features and labels must not be empty")
        mean_x=sum(x)/len(x); variance=sum((v-mean_x)**2 for v in x)/len(x); scale=math.sqrt(variance) or 1.0; standardized=[(v-mean_x)/scale for v in x]; weight=bias=0.0; n=float(len(x))
        for _ in range(self.iterations):
            probabilities=[_sigmoid(weight*v+bias) for v in standardized]
            wg=sum((p-t)*v for p,t,v in zip(probabilities,y,standardized))/n; bg=sum(p-t for p,t in zip(probabilities,y))/n
            if self.regularization>0: wg+=self.regularization*weight
            weight-=self.learning_rate*wg; bias-=self.learning_rate*bg
        self._weight=weight; self._bias=bias; self._feature_mean=mean_x; self._feature_scale=scale; return self
    def predict_proba(self, features: Sequence[float]) -> list[float]:
        self._require_fitted(); return [self._predict_probability(v) for v in _validate_features(features)]
    def predict(self, features: Sequence[float], threshold: float=.5) -> list[int]:
        if not 0<=threshold<=1: raise ValueError("threshold must be between 0 and 1")
        return [int(p>=threshold) for p in self.predict_proba(features)]
    def _predict_probability(self, feature: float) -> float:
        assert self._weight is not None and self._bias is not None and self._feature_mean is not None and self._feature_scale is not None
        return _sigmoid(self._weight*((feature-self._feature_mean)/self._feature_scale)+self._bias)
    def _require_fitted(self) -> None:
        if self._weight is None or self._bias is None or self._feature_mean is None or self._feature_scale is None: raise RuntimeError("model must be fitted before prediction")

def _prediction_count(value: int | Sequence[float]) -> int:
    if isinstance(value,bool): raise ValueError("count must be a non-negative integer")
    if isinstance(value,int):
        if value<0: raise ValueError("count must be a non-negative integer")
        return value
    if isinstance(value,(str,bytes)): raise ValueError("prediction input must be a count or feature sequence")
    try: return len(value)
    except TypeError as exc: raise ValueError("prediction input must be a count or feature sequence") from exc

def _validate_features(features: Sequence[float]) -> list[float]:
    validated=[]
    for i,v in enumerate(features):
        if isinstance(v,bool) or not isinstance(v,(int,float)): raise ValueError(f"feature at index {i} must be numeric, got {v!r}")
        v=float(v)
        if not math.isfinite(v): raise ValueError(f"feature at index {i} must be finite, got {v!r}")
        validated.append(v)
    return validated

def _validate_labels(labels: Sequence[int]) -> list[int]:
    validated=[]
    for i,label in enumerate(labels):
        if isinstance(label,bool) or not isinstance(label,int): raise ValueError(f"label at index {i} must be an integer 0 or 1")
        if label not in (0,1): raise ValueError(f"label at index {i} must be 0 or 1, got {label!r}")
        validated.append(label)
    return validated

def _sigmoid(value: float) -> float:
    if value>=0:
        e=math.exp(-value); return 1/(1+e)
    e=math.exp(value); return e/(1+e)
