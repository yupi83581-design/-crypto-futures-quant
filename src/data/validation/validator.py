"""
src/data/validation/validator.py

Data Validation - Phase 1, Crypto Futures Quant Engine.

Validates incoming market-data records against the locked data contract
in src/data/schemas/market-data.schema.json before they enter the
collector's persisted dataset.

Design constraints (Phase 0 Replay-First / Multi-Pair-Ready principle):
- Every function is pure: (record, config/state) -> result.
- No hidden reliance on wall-clock time.
- No global mutable state used for decisions.
- Sequence/duplicate tracking state is passed in and returned explicitly.

Out of scope here:
- Gap detection / REST backfill / reconciliation
- Collector implementation
- Automated test suite
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

    def add(
        self,
        category: ValidationCategory,
        message: str,
    ) -> None:
        self.valid = False
        self.errors.append(
            ValidationError(category, message)
        )


@dataclass
class SequenceState:
    """
    Per (exchange, market_type, symbol, data_type) sequence tracking state.

    Owned and persisted by the caller and passed into validation functions.
    No module-level mutable state is used.

    This keeps replay deterministic: reconstructing the same SequenceState
    and validating the same record stream produces the same result.
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
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_timestamp(value: str) -> datetime:
    """
    Parse an ISO-8601/RFC-3339 timestamp.

    A timezone/offset is mandatory so timestamps are never ambiguous.
    """

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
            f"Timestamp must include an explicit timezone/offset: {value!r}"
        )

    return parsed


def _collect_datetime_fields(
    schema: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """
    Collect schema-defined fields whose JSON Schema declares:
        format: date-time

    This avoids hardcoding timestamp field names such as funding_time.
    Only timestamp fields actually defined by the schema are considered.
    """

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

    for keyword in (
        "allOf",
        "anyOf",
        "oneOf",
    ):
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
    """
    Retrieve a value using a schema-derived property path.

    Array traversal is intentionally conservative. The current market-data
    contract uses date-time fields on record objects; nested array support is
    included so the helper remains safe if future schema versions introduce
    date-time fields inside arrays.
    """

    current = record

    for part in path:
        if part == "[]":
            if not isinstance(current, list):
                return False, None

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
    result = ValidationResult(valid=True)

    validator = jsonschema.Draft202012Validator(schema)

    for err in validator.iter_errors(record):
        result.add(
            ValidationCategory.SCHEMA,
            err.message,
        )

    return result


def validate_timestamp_integrity(
    record: dict,
    schema: Optional[dict] = None,
) -> ValidationResult:
    """
    Validate:

        event_time <= available_time <= ingestion_time

    and validate every schema-defined `format: date-time` field.

    JSON Schema format checking is intentionally not relied upon here because
    Draft 2020-12 validators do not enforce format semantics unless a
    FormatChecker is explicitly configured.
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

    if not (
        event_time
        <= available_time
        <= ingestion_time
    ):
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


def _sequence_key(
    record: dict,
) -> tuple:
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
    """
    Duplicate/out-of-order check for records carrying sequence_info.

    Records without sequence_info are skipped because not every data type
    in the schema provides sequence information.
    """

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

    prev_seq_id = sequence_info.get(
        "previous_sequence_id"
    )

    last_seen = state.last_sequence_id.get(key)

    if (
        last_seen is not None
        and prev_seq_id is not None
        and prev_seq_id != last_seen
    ):
        result.add(
            ValidationCategory.SEQUENCE,
            "Out-of-order or missing sequence for "
            f"key={key}: expected "
            f"previous_sequence_id={last_seen}, "
            f"got {prev_seq_id}",
        )

    state.last_sequence_id[key] = seq_id

    return result


def validate_record(
    record: dict,
    schema: dict,
    sequence_state: SequenceState,
) -> ValidationResult:
    combined = ValidationResult(valid=True)

    sub_results = (
        validate_schema(record, schema),
        validate_timestamp_integrity(
            record,
            schema,
        ),
        validate_sequence(
            record,
            sequence_state,
        ),
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
    schema = (
        schema
        if schema is not None
        else load_schema()
    )

    sequence_state = (
        sequence_state
        if sequence_state is not None
        else SequenceState()
    )

    results = [
        validate_record(
            record,
            schema,
            sequence_state,
        )
        for record in records
    ]

    return results, sequence_state
