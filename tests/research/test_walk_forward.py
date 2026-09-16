import pytest
from src.research.walk_forward import run_walk_forward

class Model:
    def __init__(self): self.fit_calls=[]
    def fit(self, x, y): self.fit_calls.append((tuple(x), tuple(y))); self.bias=sum(y)/len(y)
    def predict_proba(self, x): return [self.bias]*len(x)

def test_walk_forward_is_chronological_and_oos():
    seen=[]
    def factory():
        m=Model(); return m
    result=run_walk_forward(observations=list(range(10)), actual=[0,1]*5, model_factory=factory, train_size=4, test_size=2)
    assert result.test_indices == (4,5,6,7,8,9)
    assert all(f.train_end <= f.test_start for f in result.folds)
    assert len(result.predictions)==len(result.actual)==6

def test_expanding_window_is_deterministic():
    args=dict(observations=list(range(9)), actual=[0,1,0,1,0,1,0,1,0], model_factory=Model, train_size=3, test_size=2, expanding=True)
    assert run_walk_forward(**args)==run_walk_forward(**args)

def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError): run_walk_forward(observations=[1,2], actual=[1], model_factory=Model, train_size=1, test_size=1)

def test_rejects_insufficient_data():
    with pytest.raises(ValueError): run_walk_forward(observations=[1,2], actual=[0,1], model_factory=Model, train_size=2, test_size=1)

def test_rejects_invalid_binary_actual():
    with pytest.raises(ValueError, match="binary"): run_walk_forward(observations=[1,2,3], actual=[0,2,1], model_factory=Model, train_size=2, test_size=1)

def test_calibrator_is_fit_only_on_train():
    class Cal:
        def __init__(self): self.n=0
        def fit(self,p,y): self.n=len(y)
        def predict(self,p): return p
    result=run_walk_forward(observations=list(range(6)), actual=[0,1,0,1,0,1], model_factory=Model, train_size=3, test_size=1, calibrator_factory=Cal)
    assert len(result.predictions)==3
