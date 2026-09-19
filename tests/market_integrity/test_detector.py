from src.market_integrity.detector import (
    MarketIntegrityResult,
    OrderBookLevel,
    assess_market_integrity,
)


def test_normal_book_is_normal():
    result = assess_market_integrity(
        bid_depth=[OrderBookLevel(99, 10), OrderBookLevel(98, 9), OrderBookLevel(97, 8)],
        ask_depth=[OrderBookLevel(101, 10), OrderBookLevel(102, 9), OrderBookLevel(103, 8)],
    )
    assert isinstance(result, MarketIntegrityResult)
    assert result.status == "NORMAL"


def test_large_concentrated_level_is_suspicious():
    result = assess_market_integrity(
        bid_depth=[OrderBookLevel(99, 100), OrderBookLevel(98, 5), OrderBookLevel(97, 5)],
        ask_depth=[OrderBookLevel(101, 10), OrderBookLevel(102, 9), OrderBookLevel(103, 8)],
    )
    assert result.spoofing_risk >= 0.75
    assert result.status == "SUSPICIOUS"


def test_liquidity_withdrawal_is_suspicious():
    prior = [OrderBookLevel(99, 50), OrderBookLevel(98, 40), OrderBookLevel(97, 30)]
    current = [OrderBookLevel(99, 10), OrderBookLevel(98, 5), OrderBookLevel(97, 5)]
    result = assess_market_integrity(
        bid_depth=current,
        ask_depth=prior,
        prior_bid_depth=prior,
        prior_ask_depth=prior,
    )
    assert result.liquidity_withdrawal_risk >= 0.50
    assert result.status == "SUSPICIOUS"


def test_volume_anomaly_is_suspicious():
    result = assess_market_integrity(
        bid_depth=[OrderBookLevel(99, 10), OrderBookLevel(98, 9), OrderBookLevel(97, 8)],
        ask_depth=[OrderBookLevel(101, 10), OrderBookLevel(102, 9), OrderBookLevel(103, 8)],
        recent_volume=300,
        baseline_volume=100,
    )
    assert result.volume_anomaly_risk >= 0.75
    assert result.status == "SUSPICIOUS"
