from dataclasses import FrozenInstanceError

import pytest

from src.production.orchestrator import (
    OrchestratorInput,
    OrchestratorResult,
    ProductionOrchestrator,
)


def make_orchestrator(
    calls,
    *,
    with_paper=True,
    with_monitor=True,
):
    def feature_builder(data):
        calls.append("features")
        return {"rsi": 22.5, "symbol": data.symbol}

    def model(features):
        calls.append("model")
        return {"probability": 0.72, "features": features}

    def decision(model_output):
        calls.append("decision")
        return {"approved": True, "model": model_output}

    def risk(decision):
        calls.append("risk")
        return {"risk_approved": True, "decision": decision}

    def paper(risk_result):
        calls.append("paper")
        return {"paper_trade": True, "risk": risk_result}

    def monitor(payload):
        calls.append("monitor")
        return {"recorded": True, "payload": payload}

    return ProductionOrchestrator(
        feature_builder=feature_builder,
        model=model,
        decision_engine=decision,
        risk_engine=risk,
        paper_engine=paper if with_paper else None,
        monitor=monitor if with_monitor else None,
    )


class TestOrchestratorInput:
    def test_valid_input_is_accepted(self):
        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0, 102.0],
        )

        assert data.symbol == "BTCUSDT"
        assert list(data.prices) == [100.0, 101.0, 102.0]
        assert data.market_data is None

    def test_market_data_is_preserved(self):
        market_data = {"volume": 123}

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0],
            market_data=market_data,
        )

        assert data.market_data is market_data

    def test_empty_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="symbol"):
            OrchestratorInput(
                symbol="",
                prices=[100.0],
            )

    def test_whitespace_symbol_is_rejected_when_run(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="   ",
            prices=[100.0],
        )

        with pytest.raises(ValueError, match="symbol"):
            orchestrator.run(data)

    def test_non_numeric_price_is_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, "bad"],
        )

        with pytest.raises(TypeError, match="numeric"):
            orchestrator.run(data)

    def test_boolean_price_is_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, True],
        )

        with pytest.raises(TypeError, match="numeric"):
            orchestrator.run(data)

    def test_zero_price_is_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 0.0],
        )

        with pytest.raises(ValueError, match="positive"):
            orchestrator.run(data)

    def test_negative_price_is_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, -1.0],
        )

        with pytest.raises(ValueError, match="positive"):
            orchestrator.run(data)

    def test_empty_prices_are_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[],
        )

        with pytest.raises(ValueError, match="empty"):
            orchestrator.run(data)

    def test_string_prices_container_is_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices="100,101",
        )

        with pytest.raises(TypeError, match="numeric"):
            orchestrator.run(data)

    def test_non_iterable_prices_are_rejected(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=123,
        )

        with pytest.raises(TypeError, match="numeric"):
            orchestrator.run(data)


class TestProductionOrchestrator:
    def test_complete_pipeline_runs_in_correct_order(self):
        calls = []
        orchestrator = make_orchestrator(calls)

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0, 102.0],
        )

        result = orchestrator.run(data)

        assert calls == [
            "features",
            "model",
            "decision",
            "risk",
            "paper",
            "monitor",
        ]

        assert isinstance(result, OrchestratorResult)

    def test_pipeline_without_optional_components(self):
        calls = []
        orchestrator = make_orchestrator(
            calls,
            with_paper=False,
            with_monitor=False,
        )

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert calls == [
            "features",
            "model",
            "decision",
            "risk",
        ]
        assert result.paper_result is None
        assert result.monitoring_result is None

    def test_symbol_is_preserved_in_result(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="ETHUSDT",
            prices=[2000.0, 2010.0],
        )

        result = orchestrator.run(data)

        assert result.symbol == "ETHUSDT"

    def test_features_are_preserved(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert result.features["rsi"] == 22.5
        assert result.features["symbol"] == "BTCUSDT"

    def test_model_output_is_preserved(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert result.model_output["probability"] == 0.72
        assert result.model_output["features"] is result.features

    def test_decision_is_preserved(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert result.decision["approved"] is True
        assert result.decision["model"] is result.model_output

    def test_risk_is_preserved(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert result.risk["risk_approved"] is True
        assert result.risk["decision"] is result.decision

    def test_paper_result_is_preserved(self):
        orchestrator = make_orchestrator([])

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert result.paper_result["paper_trade"] is True
        assert result.paper_result["risk"] is result.risk

    def test_monitor_receives_complete_pipeline_state(self):
        captured = {}

        def monitor(payload):
            captured.update(payload)
            return "monitor-ok"

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: {"feature": 1},
            model=lambda features: {"model": 2},
            decision_engine=lambda model: {"decision": 3},
            risk_engine=lambda decision: {"risk": 4},
            paper_engine=lambda risk: {"paper": 5},
            monitor=monitor,
        )

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0],
        )

        result = orchestrator.run(data)

        assert captured["input"] is data
        assert captured["features"] is result.features
        assert captured["model_output"] is result.model_output
        assert captured["decision"] is result.decision
        assert captured["risk"] is result.risk
        assert captured["paper_result"] is result.paper_result
        assert result.monitoring_result == "monitor-ok"

    def test_feature_builder_receives_original_input(self):
        captured = {}

        def feature_builder(data):
            captured["data"] = data
            return "features"

        orchestrator = ProductionOrchestrator(
            feature_builder=feature_builder,
            model=lambda value: "model",
            decision_engine=lambda value: "decision",
            risk_engine=lambda value: "risk",
        )

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0],
        )

        orchestrator.run(data)

        assert captured["data"] is data

    def test_each_stage_receives_previous_stage_output(self):
        received = {}

        def feature_builder(data):
            received["feature_input"] = data
            return "FEATURES"

        def model(features):
            received["model_input"] = features
            return "MODEL"

        def decision(model_output):
            received["decision_input"] = model_output
            return "DECISION"

        def risk(decision_result):
            received["risk_input"] = decision_result
            return "RISK"

        orchestrator = ProductionOrchestrator(
            feature_builder=feature_builder,
            model=model,
            decision_engine=decision,
            risk_engine=risk,
        )

        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0],
        )

        orchestrator.run(data)

        assert received["feature_input"] is data
        assert received["model_input"] == "FEATURES"
        assert received["decision_input"] == "MODEL"
        assert received["risk_input"] == "DECISION"


class TestOrchestratorDependencies:
    @pytest.mark.parametrize(
        "argument",
        [
            "feature_builder",
            "model",
            "decision_engine",
            "risk_engine",
        ],
    )
    def test_required_dependency_must_be_callable(self, argument):
        kwargs = {
            "feature_builder": lambda value: value,
            "model": lambda value: value,
            "decision_engine": lambda value: value,
            "risk_engine": lambda value: value,
        }

        kwargs[argument] = None

        with pytest.raises(TypeError, match=f"{argument} must be callable"):
            ProductionOrchestrator(**kwargs)

    def test_non_callable_paper_engine_is_rejected(self):
        with pytest.raises(TypeError, match="paper_engine"):
            ProductionOrchestrator(
                feature_builder=lambda value: value,
                model=lambda value: value,
                decision_engine=lambda value: value,
                risk_engine=lambda value: value,
                paper_engine="invalid",
            )

    def test_non_callable_monitor_is_rejected(self):
        with pytest.raises(TypeError, match="monitor"):
            ProductionOrchestrator(
                feature_builder=lambda value: value,
                model=lambda value: value,
                decision_engine=lambda value: value,
                risk_engine=lambda value: value,
                monitor="invalid",
            )

    def test_optional_dependencies_can_be_none(self):
        orchestrator = ProductionOrchestrator(
            feature_builder=lambda value: value,
            model=lambda value: value,
            decision_engine=lambda value: value,
            risk_engine=lambda value: value,
            paper_engine=None,
            monitor=None,
        )

        assert orchestrator is not None


class TestOrchestratorFailurePropagation:
    def test_feature_error_is_propagated(self):
        def feature_builder(data):
            raise RuntimeError("feature failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=feature_builder,
            model=lambda value: value,
            decision_engine=lambda value: value,
            risk_engine=lambda value: value,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="feature failure"):
            orchestrator.run(data)

    def test_model_error_is_propagated(self):
        def model(features):
            raise RuntimeError("model failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: "features",
            model=model,
            decision_engine=lambda value: value,
            risk_engine=lambda value: value,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="model failure"):
            orchestrator.run(data)

    def test_decision_error_is_propagated(self):
        def decision(model_output):
            raise RuntimeError("decision failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: "features",
            model=lambda value: "model",
            decision_engine=decision,
            risk_engine=lambda value: value,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="decision failure"):
            orchestrator.run(data)

    def test_risk_error_is_propagated(self):
        def risk(decision):
            raise RuntimeError("risk failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: "features",
            model=lambda value: "model",
            decision_engine=lambda value: "decision",
            risk_engine=risk,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="risk failure"):
            orchestrator.run(data)

    def test_paper_error_is_propagated(self):
        def paper(risk):
            raise RuntimeError("paper failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: "features",
            model=lambda value: "model",
            decision_engine=lambda value: "decision",
            risk_engine=lambda value: "risk",
            paper_engine=paper,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="paper failure"):
            orchestrator.run(data)

    def test_monitor_error_is_propagated(self):
        def monitor(payload):
            raise RuntimeError("monitor failure")

        orchestrator = ProductionOrchestrator(
            feature_builder=lambda data: "features",
            model=lambda value: "model",
            decision_engine=lambda value: "decision",
            risk_engine=lambda value: "risk",
            monitor=monitor,
        )

        data = OrchestratorInput("BTCUSDT", [100.0])

        with pytest.raises(RuntimeError, match="monitor failure"):
            orchestrator.run(data)


class TestResultImmutability:
    def test_orchestrator_result_is_frozen(self):
        result = OrchestratorResult(
            symbol="BTCUSDT",
            features={},
            model_output={},
            decision={},
            risk={},
        )

        with pytest.raises(FrozenInstanceError):
            result.symbol = "ETHUSDT"

    def test_orchestrator_input_is_frozen(self):
        data = OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0],
        )

        with pytest.raises(FrozenInstanceError):
            data.symbol = "ETHUSDT"
