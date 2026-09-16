from datetime import datetime, timezone

import pytest

from src.production.inference import (
    InferenceInput,
    InferenceResult,
    ProductionInferenceEngine,
)


class FakeMarketDataAdapter:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def fetch_market_data(
        self,
        *,
        symbol,
        timeframe,
        start_time,
        end_time,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return self.records


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.received_features = None

    def predict_proba(self, features):
        self.received_features = list(features)
        return self.probabilities


class ModelWithoutPredictProba:
    pass


def make_records(count=20):
    return [
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "event_time": (
                f"2026-01-01T00:{index:02d}:00+00:00"
            ),
            "close": 100.0 + index,
        }
        for index in range(count)
    ]


def make_engine(
    records=None,
    probabilities=None,
    feature_values=None,
):
    adapter = FakeMarketDataAdapter(
        records if records is not None else make_records()
    )

    model = FakeModel(
        probabilities
        if probabilities is not None
        else [0.70, 0.75, 0.80]
    )

    def feature_builder(_records):
        if feature_values is not None:
            return feature_values

        return [
            None,
        ] * 17 + [
            45.0,
            55.0,
            60.0,
        ]

    engine = ProductionInferenceEngine(
        market_data_adapter=adapter,
        model=model,
        feature_builder=feature_builder,
    )

    return engine, adapter, model


def make_input(
    symbol="BTCUSDT",
    timeframe="5m",
    start_time="2026-01-01T00:00:00+00:00",
    end_time="2026-01-01T02:00:00+00:00",
):
    return InferenceInput(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )


def test_constructor_accepts_valid_dependencies():
    engine, _, _ = make_engine()

    assert isinstance(
        engine,
        ProductionInferenceEngine,
    )


def test_constructor_rejects_missing_market_data_method():
    with pytest.raises(
        TypeError,
        match="fetch_market_data",
    ):
        ProductionInferenceEngine(
            market_data_adapter=object(),
            model=FakeModel([0.5]),
            feature_builder=lambda records: [1.0],
        )


def test_constructor_rejects_missing_predict_proba():
    with pytest.raises(
        TypeError,
        match="predict_proba",
    ):
        ProductionInferenceEngine(
            market_data_adapter=FakeMarketDataAdapter(
                make_records()
            ),
            model=ModelWithoutPredictProba(),
            feature_builder=lambda records: [1.0],
        )


def test_constructor_rejects_non_callable_feature_builder():
    with pytest.raises(
        TypeError,
        match="feature_builder",
    ):
        ProductionInferenceEngine(
            market_data_adapter=FakeMarketDataAdapter(
                make_records()
            ),
            model=FakeModel([0.5]),
            feature_builder=None,
        )


def test_run_returns_probability():
    engine, adapter, model = make_engine()

    result = engine.run(make_input())

    assert isinstance(result, InferenceResult)
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "5m"
    assert result.observations == 20
    assert result.usable_observations == 3
    assert result.latest_feature == 60.0
    assert result.probability == 0.80

    assert model.received_features == [
        45.0,
        55.0,
        60.0,
    ]

    assert adapter.calls == [
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "start_time": (
                "2026-01-01T00:00:00+00:00"
            ),
            "end_time": (
                "2026-01-01T02:00:00+00:00"
            ),
        }
    ]


def test_feature_length_must_match_records():
    engine, _, _ = make_engine(
        feature_values=[1.0, 2.0],
    )

    with pytest.raises(
        ValueError,
        match="feature sequence length",
    ):
        engine.run(make_input())


def test_none_features_are_filtered_after_alignment():
    engine, _, model = make_engine(
        records=make_records(5),
        feature_values=[
            None,
            None,
            25.0,
            None,
            40.0,
        ],
        probabilities=[
            0.25,
            0.65,
        ],
    )

    result = engine.run(make_input())

    assert result.observations == 5
    assert result.usable_observations == 2
    assert result.latest_feature == 40.0
    assert result.probability == 0.65
    assert model.received_features == [
        25.0,
        40.0,
    ]


def test_empty_market_data_is_rejected():
    engine, _, _ = make_engine(records=[])

    with pytest.raises(
        ValueError,
        match="no market data",
    ):
        engine.run(make_input())


def test_non_list_market_data_is_rejected():
    engine, _, _ = make_engine()

    engine._market_data_adapter.records = (
        tuple(make_records())
    )

    with pytest.raises(
        TypeError,
        match="must return a list",
    ):
        engine.run(make_input())


def test_empty_features_are_rejected():
    engine, _, _ = make_engine(
        feature_values=[None] * 20,
    )

    with pytest.raises(
        ValueError,
        match="no usable features",
    ):
        engine.run(make_input())


def test_non_sequence_features_are_rejected():
    engine, _, _ = make_engine()

    engine._feature_builder = lambda records: 123

    with pytest.raises(
        TypeError,
        match="feature_builder must return a sequence",
    ):
        engine.run(make_input())


def test_string_features_are_rejected():
    engine, _, _ = make_engine()

    engine._feature_builder = lambda records: "123"

    with pytest.raises(
        TypeError,
        match="feature_builder must return a sequence",
    ):
        engine.run(make_input())


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_features_are_rejected(value):
    engine, _, _ = make_engine(
        feature_values=[value] * 20,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        engine.run(make_input())


def test_boolean_feature_is_rejected():
    engine, _, _ = make_engine(
        feature_values=[True] * 20,
    )

    with pytest.raises(
        TypeError,
        match="feature must be numeric",
    ):
        engine.run(make_input())


def test_non_numeric_feature_is_rejected():
    engine, _, _ = make_engine(
        feature_values=["50.0"] * 20,
    )

    with pytest.raises(
        TypeError,
        match="feature must be numeric",
    ):
        engine.run(make_input())


def test_non_sequence_model_output_is_rejected():
    engine, _, _ = make_engine()

    engine._model.predict_proba = (
        lambda features: 0.7
    )

    with pytest.raises(
        TypeError,
        match="must return a sequence",
    ):
        engine.run(make_input())


def test_string_model_output_is_rejected():
    engine, _, _ = make_engine()

    engine._model.predict_proba = (
        lambda features: "0.7"
    )

    with pytest.raises(
        TypeError,
        match="must return a sequence",
    ):
        engine.run(make_input())


def test_probability_count_must_match_feature_count():
    engine, _, _ = make_engine(
        feature_values=[
            40.0,
            50.0,
            60.0,
        ],
        probabilities=[0.8],
    )

    with pytest.raises(
        ValueError,
        match="probability count",
    ):
        engine.run(make_input())


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
    ],
)
def test_probability_must_be_between_zero_and_one(
    probability,
):
    engine, _, _ = make_engine(
        feature_values=[50.0] * 20,
        probabilities=[probability] * 20,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        engine.run(make_input())


@pytest.mark.parametrize(
    "probability",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_probability_must_be_finite(probability):
    engine, _, _ = make_engine(
        feature_values=[50.0] * 20,
        probabilities=[probability] * 20,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        engine.run(make_input())


def test_boolean_probability_is_rejected():
    engine, _, _ = make_engine(
        feature_values=[50.0] * 20,
        probabilities=[True] * 20,
    )

    with pytest.raises(
        TypeError,
        match="probability must be numeric",
    ):
        engine.run(make_input())


@pytest.mark.parametrize(
    "probability",
    [
        "0.5",
        None,
        object(),
    ],
)
def test_non_numeric_probability_is_rejected(
    probability,
):
    engine, _, _ = make_engine(
        feature_values=[50.0] * 20,
        probabilities=[probability] * 20,
    )

    with pytest.raises(
        TypeError,
        match="probability must be numeric",
    ):
        engine.run(make_input())


def test_empty_symbol_is_rejected_at_input_construction():
    with pytest.raises(
        ValueError,
        match="symbol",
    ):
        InferenceInput(
            symbol="",
            timeframe="5m",
            start_time="start",
            end_time="end",
        )


@pytest.mark.parametrize(
    "symbol",
    [
        " ",
        "\t",
        "\n",
    ],
)
def test_whitespace_symbol_is_rejected_when_run(
    symbol,
):
    engine, _, _ = make_engine()

    with pytest.raises(
        ValueError,
        match="symbol",
    ):
        engine.run(
            InferenceInput(
                symbol=symbol,
                timeframe="5m",
                start_time="start",
                end_time="end",
            )
        )


def test_non_string_symbol_is_rejected():
    engine, _, _ = make_engine()

    with pytest.raises(
        TypeError,
        match="symbol",
    ):
        engine.run(
            InferenceInput(
                symbol=123,
                timeframe="5m",
                start_time="start",
                end_time="end",
            )
        )


def test_empty_timeframe_is_rejected():
    engine, _, _ = make_engine()

    with pytest.raises(
        ValueError,
        match="timeframe",
    ):
        engine.run(
            InferenceInput(
                symbol="BTCUSDT",
                timeframe="",
                start_time="start",
                end_time="end",
            )
        )


def test_whitespace_timeframe_is_rejected():
    engine, _, _ = make_engine()

    with pytest.raises(
        ValueError,
        match="timeframe",
    ):
        engine.run(
            InferenceInput(
                symbol="BTCUSDT",
                timeframe=" ",
                start_time="start",
                end_time="end",
            )
        )


def test_non_string_timeframe_is_rejected():
    engine, _, _ = make_engine()

    with pytest.raises(
        TypeError,
        match="timeframe",
    ):
        engine.run(
            InferenceInput(
                symbol="BTCUSDT",
                timeframe=5,
                start_time="start",
                end_time="end",
            )
        )


@pytest.mark.parametrize(
    "field",
    [
        "start_time",
        "end_time",
    ],
)
def test_empty_time_fields_are_rejected(field):
    engine, _, _ = make_engine()

    kwargs = {
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "start_time": "start",
        "end_time": "end",
    }

    kwargs[field] = ""

    with pytest.raises(
        ValueError,
        match=field,
    ):
        engine.run(
            InferenceInput(**kwargs)
        )


@pytest.mark.parametrize(
    "field",
    [
        "start_time",
        "end_time",
    ],
)
def test_non_string_or_datetime_time_is_rejected(
    field,
):
    engine, _, _ = make_engine()

    kwargs = {
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "start_time": "start",
        "end_time": "end",
    }

    kwargs[field] = 123

    with pytest.raises(
        TypeError,
        match=field,
    ):
        engine.run(
            InferenceInput(**kwargs)
        )


def test_timezone_naive_datetime_is_rejected():
    engine, _, _ = make_engine()

    with pytest.raises(
        ValueError,
        match="timezone",
    ):
        engine.run(
            InferenceInput(
                symbol="BTCUSDT",
                timeframe="5m",
                start_time=datetime(
                    2026,
                    1,
                    1,
                ),
                end_time="2026-01-01T02:00:00+00:00",
            )
        )


def test_timezone_aware_datetime_is_accepted():
    engine, adapter, _ = make_engine()

    start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    end = datetime(
        2026,
        1,
        1,
        2,
        tzinfo=timezone.utc,
    )

    result = engine.run(
        InferenceInput(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=start,
            end_time=end,
        )
    )

    assert result.probability == 0.80
    assert adapter.calls[0]["start_time"] == start
    assert adapter.calls[0]["end_time"] == end


def test_result_is_deterministic_for_same_inputs():
    engine, _, _ = make_engine()

    data = make_input()

    first = engine.run(data)
    second = engine.run(data)

    assert first == second


def test_model_is_not_fitted_by_inference_engine():
    class TrackingModel:
        def __init__(self):
            self.fit_called = False

        def fit(self, features, labels):
            self.fit_called = True

        def predict_proba(self, features):
            return [0.5] * len(features)

    adapter = FakeMarketDataAdapter(
        make_records()
    )
    model = TrackingModel()

    engine = ProductionInferenceEngine(
        market_data_adapter=adapter,
        model=model,
        feature_builder=lambda records: [
            50.0
        ] * len(records),
    )

    engine.run(make_input())

    assert model.fit_called is False


def test_latest_probability_matches_latest_usable_feature():
    engine, _, model = make_engine(
        records=make_records(4),
        feature_values=[
            None,
            20.0,
            30.0,
            40.0,
        ],
        probabilities=[
            0.10,
            0.30,
            0.90,
        ],
    )

    result = engine.run(make_input())

    assert model.received_features == [
        20.0,
        30.0,
        40.0,
    ]
    assert result.latest_feature == 40.0
    assert result.probability == 0.90


def test_model_runtime_error_for_unfitted_model_is_preserved():
    class UnfittedModel:
        def predict_proba(self, features):
            raise RuntimeError(
                "model must be fitted before prediction"
            )

    engine = ProductionInferenceEngine(
        market_data_adapter=FakeMarketDataAdapter(
            make_records()
        ),
        model=UnfittedModel(),
        feature_builder=lambda records: [
            50.0
        ] * len(records),
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        engine.run(make_input())
