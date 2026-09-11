from dataclasses import FrozenInstanceError

import pytest

from src.data.integrity.availability import (
    AVAILABILITY_RULES,
    AvailabilityMode,
    AvailabilityRule,
)


def test_interval_rule_requires_positive_interval():
    rule = AvailabilityRule(
        data_type="ohlcv",
        mode=AvailabilityMode.INTERVAL,
        expected_interval_seconds=300,
    )

    assert rule.data_type == "ohlcv"
    assert rule.mode is AvailabilityMode.INTERVAL
    assert rule.expected_interval_seconds == 300


def test_interval_rule_requires_interval_value():
    with pytest.raises(
        ValueError,
        match="interval data requires expected_interval_seconds",
    ):
        AvailabilityRule(
            data_type="ohlcv",
            mode=AvailabilityMode.INTERVAL,
        )


def test_interval_must_be_positive():
    with pytest.raises(
        ValueError,
        match="expected_interval_seconds must be positive",
    ):
        AvailabilityRule(
            data_type="ohlcv",
            mode=AvailabilityMode.INTERVAL,
            expected_interval_seconds=0,
        )


def test_data_type_must_not_be_empty():
    with pytest.raises(ValueError, match="data_type must not be empty"):
        AvailabilityRule(
            data_type="",
            mode=AvailabilityMode.EVENT,
        )


def test_event_rule_does_not_require_interval():
    rule = AvailabilityRule(
        data_type="trades",
        mode=AvailabilityMode.EVENT,
    )

    assert rule.mode is AvailabilityMode.EVENT
    assert rule.expected_interval_seconds is None


def test_default_market_data_rules_exist():
    assert AVAILABILITY_RULES["ohlcv"].expected_interval_seconds == 300
    assert AVAILABILITY_RULES["trades"].mode is AvailabilityMode.EVENT
    assert AVAILABILITY_RULES["order_book_l2"].mode is AvailabilityMode.EVENT
    assert AVAILABILITY_RULES["funding"].mode is AvailabilityMode.EVENT


def test_availability_rule_is_immutable():
    rule = AvailabilityRule(
        data_type="ohlcv",
        mode=AvailabilityMode.INTERVAL,
        expected_interval_seconds=300,
    )

    with pytest.raises(FrozenInstanceError):
        rule.data_type = "trades"
