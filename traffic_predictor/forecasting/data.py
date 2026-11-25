"""Loading and cleaning the junction vehicle-count dataset.

The CSV has one column per junction ("Cross 1" ... "Cross 6") and one row per
reading. It has no timestamps. Two facts about it are inferred from the data,
not stated by its source, and everything that relies on them says so:

* Readings are 5 minutes apart. The series repeats every 288 rows (the
  autocorrelation peaks at 288, 576, 864, ...), and 288 x 5 min = 24 h.
* Row 0 is 00:00. With that alignment the daily low falls around 02:00-03:00
  and the peak around 08:00-09:00, which matches ordinary urban traffic.

The data also has sensor outages: runs of 12 or more consecutive rows in which
all six junctions read exactly 0 (about 17.5% of rows, all in days 2-26). Real
traffic never empties six junctions at once for hours, so these rows are
treated as missing rather than as zero traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MINUTES = 5
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  # 288
STEPS_PER_WEEK = 7 * STEPS_PER_DAY  # 2016
OUTAGE_MIN_RUN = 12  # one hour of simultaneous zeros at every junction


@dataclass(frozen=True)
class Dataset:
    """Cleaned counts: `counts` has NaN where a reading was an outage."""

    counts: pd.DataFrame  # shape (steps, junctions), float, NaN = missing
    raw: pd.DataFrame  # as read from disk, columns normalised
    outage: np.ndarray  # bool per step

    @property
    def junctions(self) -> list[str]:
        return list(self.counts.columns)

    @property
    def n_steps(self) -> int:
        return len(self.counts)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV and normalise column names (the file has ' Cross 2' etc.)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"{path}: expected one column per junction, got {df.shape}")
    df = df.apply(pd.to_numeric, errors="raise").astype(float)
    if (df < 0).any().any():
        raise ValueError(f"{path}: vehicle counts cannot be negative")
    return df


def find_outages(raw: pd.DataFrame, min_run: int = OUTAGE_MIN_RUN) -> np.ndarray:
    """Mark rows inside runs of >= `min_run` rows where every junction is 0."""
    all_zero = (raw.to_numpy() == 0).all(axis=1)
    outage = np.zeros(len(raw), dtype=bool)
    start = None
    for i, z in enumerate(np.append(all_zero, False)):
        if z and start is None:
            start = i
        elif not z and start is not None:
            if i - start >= min_run:
                outage[start:i] = True
            start = None
    return outage


def load_dataset(path: str | Path) -> Dataset:
    raw = load_raw(path)
    outage = find_outages(raw)
    counts = raw.copy()
    counts.loc[outage, :] = np.nan
    return Dataset(counts=counts, raw=raw, outage=outage)


def step_day(step: int) -> int:
    """1-based day number of a step."""
    return step // STEPS_PER_DAY + 1


def step_clock(step: int) -> str:
    """Inferred clock time of a step, HH:MM."""
    minutes = (step % STEPS_PER_DAY) * STEP_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def step_label(step: int) -> str:
    return f"Day {step_day(step)} · {step_clock(step)}"
