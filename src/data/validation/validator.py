"""
Data Validation - Phase 1, Crypto Futures Quant Engine.

Validates incoming market-data records against the locked data contract
before they enter the collector's persisted dataset.

Design:
- Deterministic validation.
- No wall-clock dependency.
- No global mutable decision state.
- Sequence state is explicit and caller-owned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import jsonschema


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "market-data.schema.json"
)


class ValidationStatus(str, Enum):
    """Overall validation status."""

    VALID = "VALID"
    INVALID = "INVALID"


class ValidationCategory(str, Enum):
    """Validation error categories."""

    SCHEMA = "SCHEMA"
    TIMESTAMP = "TIMESTAMP"
    DUPLICATE = "DUPLICATE"
    SEQUENCE = "SEQUENCE"


@dataclass
class ValidationError:
    """A single validation error."""

    category: ValidationCategory
    message: str


@dataclass
class ValidationResult:
    """Result returned by validation operations."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def status(self) -> ValidationStatus:
        """Return the explicit contract status."""
        return (
            ValidationStatus.VALID
            if self.valid
            else ValidationStatus.INVALID
        )

    def add(
        self,
        category: ValidationCategory,
        message: str,
    ) -> None:
        """Record an error and mark the result invalid."""
        self.valid = False
        self.errors.append(
            ValidationError(category, message)
        )


@dataclass
class SequenceState:
    """
    Per (exchange, market_type, symbol, data_type) sequence state.

    State is owned by the caller so replay remains deterministic.
    """

    last_sequence_id: dict[tuple, Union[int, str]] = field(
        default_factory=dict
    )
    seen_sequence_ids: dict[tuple, set] = field(
        default_factory=dict
    )


def load_schema(
    schema_path: Path = SCHEMA_PATH,
) -> dict:
    """Load the locked market-data JSON Schema."""
    with open(schema_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601/RFC-3339 timestamp with timezone information."""
    if not isinstance(value, str):
        raise ValueError(
            f"Timestamp must be a string, got {type(value).__name__}"
        )

    normalized = value

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO-8601/RFC-3339 timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "Timestamp must include an explicit timezone/offset: "
            f"{value!r}"
        )

    return parsed


def _collect_datetime_fields(
    schema: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """Collect schema-defined date-time fields recursively."""
    fields: list[tuple[str, ...]] = []

    if not isinstance(schema, dict):
        return fields

    if schema.get("format") == "date-time" and path:
        fields.append(path)

    properties = schema.get("properties")

    if isinstance(properties, dict):
        for name, child_schema in properties.items():
            fields.extend(
                _collect_datetime_fields(
                    child_schema,
                    path + (name,),
                )
            )

    items = schema.get("items")

    if isinstance(items, dict):
        fields.extend(
            _collect_datetime_fields(
                items,
                path + ("[]",),
            )
        )

    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)

        if isinstance(branches, list):
            for branch in branches:
                fields.extend(
                    _collect_datetime_fields(
                        branch,
                        path,
                    )
                )

    return fields


def _get_nested_value(
    record: Any,
    path: tuple[str, ...],
) -> tuple[bool, Any]:
    """Retrieve a value using a schema-derived property path."""
    current = record

    for part in path:
        if part == "[]":
            return False, None

        if not isinstance(current, dict):
            return False, None

        if part not in current:
            return False, None

        current = current[part]

    return True, current


def validate_schema(
    record: dict,
    schema: dict,
) -> ValidationResult:
    """Validate a record against the JSON Schema contract."""
    result = ValidationResult(valid=True)

    validator = jsonschema.Draft202012Validator(schema)

    for error in validator.iter_errors(record):
        result.add(
            ValidationCategory.SCHEMA,
            error.message,
        )

    return result


def validate_timestamp_integrity(
    record: dict,
    schema: Optional[dict] = None,
) -> ValidationResult:
    """
    Validate:
        event_time <= available_time <= ingestion_time

    Also validate every schema-defined date-time field.
    """
    result = ValidationResult(valid=True)

    if schema is None:
        schema = load_schema()

    required = (
        "event_time",
        "available_time",
        "ingestion_time",
    )

    if not all(key in record for key in required):
        result.add(
            ValidationCategory.TIMESTAMP,
            f"Missing one of required timestamp fields: {required}",
        )
        return result

    try:
        event_time = _parse_timestamp(record["event_time"])
        available_time = _parse_timestamp(record["available_time"])
        ingestion_time = _parse_timestamp(record["ingestion_time"])
    except (TypeError, ValueError) as exc:
        result.add(
            ValidationCategory.TIMESTAMP,
            f"Unparseable timestamp: {exc}",
        )
        return result

    if not event_time <= available_time <= ingestion_time:
        result.add(
            ValidationCategory.TIMESTAMP,
            "Decision-time invariant violated: expected "
            "event_time <= available_time <= ingestion_time, got "
            f"event_time={event_time.isoformat()}, "
            f"available_time={available_time.isoformat()}, "
            f"ingestion_time={ingestion_time.isoformat()}",
        )

    datetime_fields = _collect_datetime_fields(schema)

    for path in datetime_fields:
        exists, value = _get_nested_value(record, path)

        if not exists:
            continue

        try:
            _parse_timestamp(value)
        except (TypeError, ValueError) as exc:
            field_name = ".".join(path)

            result.add(
                ValidationCategory.TIMESTAMP,
                f"Invalid timestamp field {field_name}: {exc}",
            )

    return result


def _sequence_key(record: dict) -> tuple:
    """Build the deterministic sequence-tracking key."""
    return (
        record.get("exchange"),
        record.get("market_type"),
        record.get("symbol"),
        record.get("data_type"),
    )


def validate_sequence(
    record: dict,
    state: SequenceState,
) -> ValidationResult:
    """Validate duplicate and out-of-order sequence information."""
    result = ValidationResult(valid=True)

    sequence_info = record.get("sequence_info")

    if sequence_info is None:
        return result

    key = _sequence_key(record)
    seq_id = sequence_info["sequence_id"]

    seen = state.seen_sequence_ids.setdefault(
        key,
        set(),
    )

    if seq_id in seen:
        result.add(
            ValidationCategory.DUPLICATE,
            f"Duplicate sequence_id={seq_id} for key={key}",
        )
        return result

    seen.add(seq_id)

    previous_sequence_id = sequence_info.get(
        "previous_sequence_id"
    )

    last_seen = state.last_sequence_id.get(key)

    if (
        last_seen is not None
        and previous_sequence_id is not None
        and previous_sequence_id != last_seen
    ):
        result.add(
            ValidationCategory.SEQUENCE,
            "Out-of-order or missing sequence for "
            f"key={key}: expected previous_sequence_id="
            f"{last_seen}, got {previous_sequence_id}",
        )

    state.last_sequence_id[key] = seq_id

    return result


def validate_record(
    record: dict,
    schema: dict,
    sequence_state: Optional[SequenceState] = None,
) -> ValidationResult:
    """
    Validate one market-data record.

    sequence_state is optional for the simple two-argument contract.
    When omitted, a fresh state is used for this validation call.
    """
    if sequence_state is None:
        sequence_state = SequenceState()

    combined = ValidationResult(valid=True)

    sub_results = (
        validate_schema(record, schema),
        validate_timestamp_integrity(record, schema),
        validate_sequence(record, sequence_state),
    )

    for sub_result in sub_results:
        combined.valid = (
            combined.valid
            and sub_result.valid
        )
        combined.errors.extend(
            sub_result.errors
        )

    return combined


def validate_batch(
    records: list[dict],
    schema: Optional[dict] = None,
    sequence_state: Optional[SequenceState] = None,
) -> tuple[list[ValidationResult], SequenceState]:
    """Validate a batch while preserving explicit sequence state."""
    if schema is None:
        schema = load_schema()

    if sequence_state is None:
        sequence_state = SequenceState()

    results = [
        validate_record(
            record,
            schema,
            sequence_state,
        )
        for record in records
    ]

    return results, sequence_state
