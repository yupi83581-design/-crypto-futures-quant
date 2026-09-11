"""Detection of missing intervals in time-series market data."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Gap:
    """A missing interval between two consecutive events."""

    previous_event: datetime
    next_event: datetime
    missing_duration: timedelta


def detect_gaps(
    event_times: list[datetime],
    expected_interval: timedelta,
) -> list[Gap]:
    """Detect missing expected intervals in a sequence of event times."""
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")

    if len(event_times) < 2:
        return []

    ordered_times = sorted(event_times)
    gaps: list[Gap] = []

    for previous_event, next_event in zip(
        ordered_times,
        ordered_times[1:],
    ):
        delta = next_event - previous_event

        if delta > expected_interval:
            gaps.append(
                Gap(
                    previous_event=previous_event,
                    next_event=next_event,
                    missing_duration=delta - expected_interval,
                )
            )

    return gaps
