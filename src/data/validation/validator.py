"""
src/data/validation/validator.py

Data Validation - Phase 1, Crypto Futures Quant Engine.

Validates incoming market-data records against the locked data contract in
src/data/schemas/market-data.schema.json before they enter the collector's
persisted dataset.

Design constraints (Phase 0 Replay-First / Multi-Pair-Ready principle):
- Every function is pure: (record, config/state) -> result. No hidden
  reliance on wall-clock time, no global mutable state used for decisions.
- Sequence/duplicate tracking state is passed in and returned explicitly,
  not held in module-level globals, so replay can reconstruct state
  deterministically instead of depending on process lifetime.

Out of scope here (handled in later Phase 1 steps):
- Gap detection / REST backfill / reconciliation (Collector + Gap Recovery)
- Automated test suite (Integrity Tests step)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "market-data.schema.json"


class ValidationCategory(str, Enum):
    SCHEMA = "SCHEMA"
    TIMESTAMP = "TIMESTAMP"
    DUPLICATE = "DUPLICATE"
    SEQUENCE = "SEQUENCE"


@dataclass
class ValidationError:
    category: ValidationCategory
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def add(self, category: ValidationCategory, message: str) -> None:
        self.valid = False
        self.errors.append(ValidationError(category, message))


@dataclass
class SequenceState:
    """Per (exchange, market_type, symbol, data_type) sequence tracking state.

    Owned and persisted by the caller (e.g. the Collector), passed into
    validate_record/validate_batch, and returned updated. This keeps the
    validator itself stateless between calls, which is what makes replay
    deterministic: reconstruct the same SequenceState and you get the same
    validation outcome for the same record stream, regardless of when or
    how many times it is replayed.
    """
    last_sequence_id: dict[tuple, Union[int, str]] = field(default_factory=dict)
    seen_sequence_ids: dict[tuple, set] = field(default_factory=dict)


def load_schema(schema_path: Path = SCHEMA_PATH) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def validate_schema(record: dict, schema: dict) -> ValidationResult:
    result = ValidationResult(valid=True)
    validator = jsonschema.Draft202012Validator(schema)
    for err in validator.iter_errors(record):
        result.add(ValidationCategory.SCHEMA, err.message)
    return result


def validate_timestamp_integrity(record: dict) -> ValidationResult:
    result = ValidationResult(valid=True)
    required = ("event_time", "available_time", "ingestion_time")
    if not all(k in record for k in required):
        result.add(
            ValidationCategory.TIMESTAMP,
            f"Missing one of required timestamp fields: {required}",
        )
        return result

    try:
        event_time = _parse_timestamp(record["event_time"])
        available_time = _parse_timestamp(record["available_time"])
        ingestion_time = _parse_timestamp(record["ingestion_time"])
    except ValueError as exc:
        result.add(ValidationCategory.TIMESTAMP, f"Unparseable timestamp: {exc}")
        return result

    if not (event_time <= available_time <= ingestion_time):
        result.add(
            ValidationCategory.TIMESTAMP,
            "Decision-time invariant violated: expected "
            "event_time <= available_time <= ingestion_time, got "
            f"event_time={event_time.isoformat()}, "
            f"available_time={available_time.isoformat()}, "
            f"ingestion_time={ingestion_time.isoformat()}",
        )

    return result


def _sequence_key(record: dict) -> tuple:
    return (
        record.get("exchange"),
        record.get("market_type"),
        record.get("symbol"),
        record.get("data_type"),
    )


def validate_sequence(record: dict, state: SequenceState) -> ValidationResult:
    """Duplicate/out-of-order check for records carrying sequence_info.
    Records without sequence_info are skipped: not every data type in the
    schema provides one, and the schema intentionally does not force it."""
    result = ValidationResult(valid=True)
    sequence_info = record.get("sequence_info")
    if sequence_info is None:
        return result

    key = _sequence_key(record)
    seq_id = sequence_info["sequence_id"]

    seen = state.seen_sequence_ids.setdefault(key, set())
    if seq_id in seen:
        result.add(
            ValidationCategory.DUPLICATE,
            f"Duplicate sequence_id={seq_id} for key={key}",
        )
        return result
    seen.add(seq_id)

    prev_seq_id = sequence_info.get("previous_sequence_id")
    last_seen = state.last_sequence_id.get(key)
    if last_seen is not None and prev_seq_id is not None and prev_seq_id != last_seen:
        result.add(
            ValidationCategory.SEQUENCE,
            f"Out-of-order or missing sequence for key={key}: "
            f"expected previous_sequence_id={last_seen}, got {prev_seq_id}",
        )

    state.last_sequence_id[key] = seq_id
    return result


def validate_record(
    record: dict,
    schema: dict,
    sequence_state: SequenceState,
) -> ValidationResult:
    combined = ValidationResult(valid=True)

    for sub_result in (
        validate_schema(record, schema),
        validate_timestamp_integrity(record),
        validate_sequence(record, sequence_state),
    ):
        combined.valid = combined.valid and sub_result.valid
        combined.errors.extend(sub_result.errors)

    return combined


def validate_batch(
    records: list[dict],
    schema: Optional[dict] = None,
    sequence_state: Optional[SequenceState] = None,
) -> tuple[list[ValidationResult], SequenceState]:
    schema = schema if schema is not None else load_schema()
    sequence_state = sequence_state if sequence_state is not None else SequenceState()

    results = [
        validate_record(record, schema, sequence_state) for record in records
    ]
    return results, sequence_state
