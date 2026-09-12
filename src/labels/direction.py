"""Forward-return direction labels for quantitative research."""

from __future__ import annotations

from typing import Any


def compute_direction_labels(
    records: list[dict[str, Any]],
    horizon: int = 3,
) -> list[int | None]:
    """Compute future-direction labels for an ordered OHLCV series.

    For record i, the label compares close[i + horizon] with close[i].

    Label definition:
        1 -> future close is higher
        0 -> future close is equal or lower
        None -> insufficient future data

    The output always has the same length as records.
    """
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon < 1
    ):
        raise ValueError(
            f"horizon must be a positive integer, got {horizon!r}"
        )

    closes: list[float] = []

    for index, record in enumerate(records):
        if "close" not in record:
            raise ValueError(
                f"record at index {index} is missing required 'close' field"
            )

        close = record["close"]

        if isinstance(close, bool) or not isinstance(close, (int, float)):
            raise ValueError(
                f"record at index {index} has non-numeric 'close': {close!r}"
            )

        closes.append(float(close))

    labels: list[int | None] = [None] * len(records)

    for index in range(len(records) - horizon):
        current_close = closes[index]
        future_close = closes[index + horizon]

        labels[index] = 1 if future_close > current_close else 0

    return labels
