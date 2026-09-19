from src.market_integrity.detector import MarketIntegrityResult
from src.monitoring.journal import Journal
from src.paper.engine import PaperTradingEngine
from src.production.paper_cycle import ProductionPaperCycle


def normal_integrity():
    return MarketIntegrityResult(
        status="NORMAL",
        spoofing_risk=0.0,
        liquidity_withdrawal_risk=0.0,
        volume_anomaly_risk=0.0,
        reasons=(),
    )


def test_missing_integrity_evidence_blocks_paper_entry():
    journal = Journal()
    cycle = ProductionPaperCycle(
        paper_engine=PaperTradingEngine(),
        journal=journal,
    )

    result = cycle.run(
        symbol="BTCUSDT",
        probability=0.80,
        entry_price=100.0,
        stop_price=99.0,
    )

    assert result.decision.approved is False
    assert result.decision.reason == "market integrity evidence unavailable"
    assert result.paper_position is None
    assert journal.snapshot().approved_entries == 0


def test_suspicious_integrity_blocks_paper_entry():
    journal = Journal()
    cycle = ProductionPaperCycle(
        paper_engine=PaperTradingEngine(),
        journal=journal,
    )
    integrity = MarketIntegrityResult(
        status="SUSPICIOUS",
        spoofing_risk=0.9,
        liquidity_withdrawal_risk=0.0,
        volume_anomaly_risk=0.0,
        reasons=("concentrated displayed liquidity",),
    )

    result = cycle.run(
        symbol="BTCUSDT",
        probability=0.80,
        entry_price=100.0,
        stop_price=99.0,
        market_integrity=integrity,
    )

    assert result.decision.approved is False
    assert result.paper_position is None


def test_normal_integrity_runs_ev_risk_decision_and_paper_entry():
    journal = Journal()
    paper = PaperTradingEngine()
    cycle = ProductionPaperCycle(
        paper_engine=paper,
        journal=journal,
    )

    result = cycle.run(
        symbol="BTCUSDT",
        probability=0.80,
        entry_price=100.0,
        stop_price=99.0,
        market_integrity=normal_integrity(),
    )

    assert result.expected_value is not None
    assert result.risk is not None
    assert result.risk.approved is True
    assert result.decision.approved is True
    assert result.paper_position is not None
    assert paper.position is not None
    assert journal.snapshot().approved_entries == 1


def test_low_probability_is_rejected_without_opening_paper_position():
    journal = Journal()
    paper = PaperTradingEngine()
    cycle = ProductionPaperCycle(
        paper_engine=paper,
        journal=journal,
    )

    result = cycle.run(
        symbol="BTCUSDT",
        probability=0.20,
        entry_price=100.0,
        stop_price=99.0,
        market_integrity=normal_integrity(),
    )

    assert result.decision.approved is False
    assert result.paper_position is None
    assert paper.position is None
