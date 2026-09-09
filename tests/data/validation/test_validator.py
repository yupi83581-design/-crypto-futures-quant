"""
tests/data/validation/test_validator.py

Integrity Tests - Phase 1, Crypto Futures Quant Engine.

Verifies src/data/validation/validator.py against the locked data contract
before the Collector is allowed to produce data at volume.
"""

from __future__ import annotations

import copy

import pytest

from src.data.validation.validator import (
    SequenceState,
    ValidationCategory,
    load_schema,
    validate_batch,
    validate_schema,
    validate_sequence,
    validate_timestamp_integrity,
)


BASE_RECORD = {
    "exchange": "BINANCE",
    "market_type": "FUTURES",
    "symbol": "BTCUSDT",
    "data_type": "OHLCV",
    "event_time": "2026-01-01T00:00:00Z",
    "available_time": "2026-01-01T00:00:01Z",
    "ingestion_time": "2026-01-01T00:00:02Z",
    "schema_version": "1.0.0",
    "timeframe": "5m",
    "open": 100.0,
    "high": 101.0,
    "low": 99.0,
    "close": 100.5,
    "volume": 10.0,
}


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def make_record(**overrides):
    record = copy.deepcopy(BASE_RECORD)
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_valid_schema_record_passes(schema):
    result = validate_schema(make_record(), schema)
    assert result.valid is True
    assert result.errors == []


def test_invalid_schema_record_rejected(schema):
    record = make_record()
    del record["close"]
    result = validate_schema(record, schema)
    assert result.valid is False
    assert any(e.category == ValidationCategory.SCHEMA for e in result.errors)


def test_invalid_schema_wrong_type_rejected(schema):
    result = validate_schema(make_record(volume="not-a-number"), schema)
    assert result.valid is False


# ---------------------------------------------------------------------------
# Timestamp integrity: event_time <= available_time <= ingestion_time
# ---------------------------------------------------------------------------

def test_valid_timestamp_order_passes():
    result = validate_timestamp_integrity(make_record())
    assert result.valid is True


def test_event_time_after_available_time_rejected():
    record = make_record(
        event_time="2026-01-01T00:00:05Z",
        available_time="2026-01-01T00:00:01Z",
        ingestion_time="2026-01-01T00:00:02Z",
    )
    result = validate_timestamp_integrity(record)
    assert result.valid is False
    assert any(e.category == ValidationCategory.TIMESTAMP for e in result.errors)


def test_available_time_after_ingestion_time_rejected():
    record = make_record(
        event_time="2026-01-01T00:00:00Z",
        available_time="2026-01-01T00:00:05Z",
        ingestion_time="2026-01-01T00:00:02Z",
    )
    result = validate_timestamp_integrity(record)
    assert result.valid is False
    assert any(e.category == ValidationCategory.TIMESTAMP for e in result.errors)


def test_missing_timestamp_field_rejected():
    record = make_record()
    del record["ingestion_time"]
    result = validate_timestamp_integrity(record)
    assert result.valid is False


# ---------------------------------------------------------------------------
# Sequence / duplicate detection
# ---------------------------------------------------------------------------

def test_valid_sequence_progression_passes():
    state = SequenceState()
    r1 = make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None})
    r2 = make_record(sequence_info={"sequence_id": 2, "previous_sequence_id": 1})

    assert validate_sequence(r1, state).valid is True
    assert validate_sequence(r2, state).valid is True


def test_duplicate_sequence_id_rejected():
    state = SequenceState()
    record = make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None})

    first = validate_sequence(record, state)
    second = validate_sequence(copy.deepcopy(record), state)

    assert first.valid is True
    assert second.valid is False
    assert any(e.category == ValidationCategory.DUPLICATE for e in second.errors)


def test_out_of_order_sequence_rejected():
    state = SequenceState()
    r1 = make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None})
    r3 = make_record(sequence_info={"sequence_id": 3, "previous_sequence_id": 2})

    validate_sequence(r1, state)
    result = validate_sequence(r3, state)

    assert result.valid is False
    assert any(e.category == ValidationCategory.SEQUENCE for e in result.errors)


def test_record_without_sequence_info_is_skipped():
    record = make_record()
    record.pop("sequence_info", None)
    assert validate_sequence(record, SequenceState()).valid is True


# ---------------------------------------------------------------------------
# State passed explicitly / no hidden global state
# ---------------------------------------------------------------------------

def test_sequence_state_is_not_shared_across_independent_states():
    record = make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None})

    result_a = validate_sequence(copy.deepcopy(record), SequenceState())
    result_b = validate_sequence(copy.deepcopy(record), SequenceState())

    # Same sequence_id=1 accepted independently in two unrelated states —
    # proves state is passed explicitly, never held globally in the module.
    assert result_a.valid is True
    assert result_b.valid is True


def test_validate_batch_returns_independent_state_per_call(schema):
    records = [
        make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None}),
        make_record(sequence_info={"sequence_id": 2, "previous_sequence_id": 1}),
    ]

    results_1, state_1 = validate_batch(copy.deepcopy(records), schema=schema)
    results_2, state_2 = validate_batch(copy.deepcopy(records), schema=schema)

    assert all(r.valid for r in results_1)
    assert all(r.valid for r in results_2)
    assert state_1 is not state_2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_validation_is_deterministic_across_repeated_runs(schema):
    records = [
        make_record(sequence_info={"sequence_id": 1, "previous_sequence_id": None}),
        make_record(sequence_info={"sequence_id": 2, "previous_sequence_id": 1}),
        make_record(sequence_info={"sequence_id": 2, "previous_sequence_id": 1}),  # duplicate
    ]

    run_1, _ = validate_batch(copy.deepcopy(records), schema=schema)
    run_2, _ = validate_batch(copy.deepcopy(records), schema=schema)

    outcomes_1 = [(r.valid, [(e.category, e.message) for e in r.errors]) for r in run_1]
    outcomes_2 = [(r.valid, [(e.category, e.message) for e in r.errors]) for r in run_2]
