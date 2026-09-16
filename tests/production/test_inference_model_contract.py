from src.features.rsi import compute_rsi
from src.models.baseline import MajorityClassBaseline, LogisticRegressionBaseline
from src.production.inference import InferenceInput, ProductionInferenceEngine


def _records(count=20):
    return [
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "event_time": f"2026-01-01T00:{index:02d}:00+00:00",
            "close": 100.0 + index,
        }
        for index in range(count)
    ]


class _Adapter:
    def __init__(self, records):
        self.records = records

    def fetch_market_data(self, *, symbol, timeframe, start_time, end_time):
        return self.records


def _input():
    return InferenceInput(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-01-01T02:00:00+00:00",
    )


def test_majority_baseline_integrates_with_production_inference():
    model = MajorityClassBaseline().fit([0, 0, 1, 0])
    engine = ProductionInferenceEngine(
        market_data_adapter=_Adapter(_records()),
        model=model,
        feature_builder=compute_rsi,
    )

    result = engine.run(_input())

    assert result.observations == 20
    assert result.usable_observations == 6
    assert result.latest_feature == 100.0
    assert result.probability == 0.0


def test_logistic_baseline_integrates_with_production_inference():
    model = LogisticRegressionBaseline(iterations=500).fit(
        [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        [0, 0, 0, 1, 1, 1],
    )
    engine = ProductionInferenceEngine(
        market_data_adapter=_Adapter(_records()),
        model=model,
        feature_builder=compute_rsi,
    )

    result = engine.run(_input())

    assert result.observations == 20
    assert result.usable_observations == 6
    assert result.latest_feature == 100.0
    assert 0.0 <= result.probability <= 1.0
