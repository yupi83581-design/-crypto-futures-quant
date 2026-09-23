import pytest
from src.research.pbo import pbo_cs_cv

def test_formal_cscv_has_auditable_path_count():
    result = pbo_cs_cv(
        [
            [0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10, 0.11],
            [0.08, 0.09, 0.07, 0.08, 0.09, 0.07, 0.08, 0.09],
        ],
        block_count=4,
    )
    assert result.block_count == 4
    assert result.path_count == 6
    assert len(result.omega_values) == 6
    assert len(result.logit_values) == 6

def test_tie_handling_is_deterministic():
    values=[[0.1]*8,[0.1]*8]
    first=pbo_cs_cv(values, block_count=4)
    second=pbo_cs_cv(values, block_count=4)
    assert first == second

def test_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        pbo_cs_cv([[0.1]*8,[0.1]*7], block_count=4)

def test_rejects_odd_block_count():
    with pytest.raises(ValueError):
        pbo_cs_cv([[0.1]*8,[0.1]*8], block_count=3)

def test_deterministic():
    values=[
        [0.10,0.11,0.09,0.10,0.11,0.09,0.10,0.11],
        [0.08,0.09,0.07,0.08,0.09,0.07,0.08,0.09],
        [0.05,0.06,0.04,0.05,0.06,0.04,0.05,0.06],
    ]
    assert pbo_cs_cv(values, block_count=4) == pbo_cs_cv(values, block_count=4)
