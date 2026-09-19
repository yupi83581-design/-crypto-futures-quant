import pytest
from src.research.robustness import evaluate_robustness

def test_robustness_is_deterministic_and_summarized():
    r=evaluate_robustness(base_parameters=10, perturbations=[9,10,11], evaluator=lambda x: float(x), tolerance=.2)
    assert r.values==(9.0,10.0,11.0); assert r.minimum==9; assert r.maximum==11; assert r.stable
    assert r==evaluate_robustness(base_parameters=10, perturbations=[9,10,11], evaluator=lambda x: float(x), tolerance=.2)

def test_rejects_empty_or_negative_tolerance():
    with pytest.raises(ValueError): evaluate_robustness(base_parameters=1, perturbations=[], evaluator=float)
    with pytest.raises(ValueError): evaluate_robustness(base_parameters=1, perturbations=[1], evaluator=float, tolerance=-1)

def test_rejects_nonfinite_metric():
    with pytest.raises(ValueError): evaluate_robustness(base_parameters=1, perturbations=[1], evaluator=lambda x: float('inf'))
