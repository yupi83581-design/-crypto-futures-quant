"""JSONL-backed historical market data source."""

import json
from pathlib import Path
from typing import Any

from src.data.historical.source import HistoricalDataSource


class JsonlHistoricalDataSource(HistoricalDataSource):
    """Load historical market-data records from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Fetch records matching the requested historical scope."""
        records: list[dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record on line {line_number} must be a JSON object"
                    )

                if record.get("symbol") != symbol:
                    continue

                if record.get("timeframe") != timeframe:
                    continue

                event_time = record.get("event_time")
                if event_time is None:
                    continue

                if not (start_time <= event_time <= end_time):
                    continue

                records.append(record)

        return records
