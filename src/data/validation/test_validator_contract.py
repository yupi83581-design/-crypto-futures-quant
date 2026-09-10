"""Regression tests for the market-data validation contract."""

from src.data.validation.validator import (
    ValidationStatus,
    load_schema,
    validate_record,
)


def test_market_data_schema_is_loaded():
    schema = load_schema()

    assert isinstance(schema, dict)
    assert schema.get("$schema")
    assert schema.get("properties")


def test_invalid_record_is_rejected_by_contract():
    schema = load_schema()

    result = validate_record({}, schema)

    assert result.status == ValidationStatus.INVALID
    assert result.errors
