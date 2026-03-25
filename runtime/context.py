"""JobContext — the runtime context passed to user scripts via run(ctx)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from runtime import transforms


@dataclass
class GpuInfo:
    """GPU hardware information available to the job."""

    type: str | None = None
    memory_gb: float | None = None


class DataLoader:
    """Load datasets from the job's data directory."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(
        self,
        file: str,
        entity: str | None = None,
        time: str | None = None,
    ) -> pd.DataFrame:
        """Load a CSV or Parquet file, tagging panel metadata in df.attrs.

        Args:
            file: Filename relative to data_dir.
            entity: Entity identifier column name (stored in df.attrs).
            time: Time column name (stored in df.attrs).

        Returns:
            DataFrame with attrs['entity'] and attrs['time'] set.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        path = self._data_dir / file
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        df.attrs["entity"] = entity
        df.attrs["time"] = time
        return df


class _TransformNamespace:
    """Namespace exposing transform functions as ctx.transform.*."""

    winsorize = staticmethod(transforms.winsorize)
    lag = staticmethod(transforms.lag)
    lead = staticmethod(transforms.lead)
    diff = staticmethod(transforms.diff)
    balance_panel = staticmethod(transforms.balance_panel)
    dummy = staticmethod(transforms.dummy)
    standardize = staticmethod(transforms.standardize)
    merge = staticmethod(transforms.merge)
    recode = staticmethod(transforms.recode)


class JobContext:
    """Runtime context passed to user scripts.

    Usage in user code::

        def run(ctx):
            df = ctx.data.load("panel.csv", entity="firm_id", time="year")
            df = ctx.transform.winsorize(df, "revenue", pct=0.01)
            ctx.progress(0.5, "Data loaded and cleaned")
            # ... compute ...
            ctx.progress(1.0, "Done")

    Args:
        data_dir: Path to the job's input data directory.
        output_dir: Path to the job's output directory.
        config: Job configuration dict (from submission).
        progress_callback: Optional callable(pct, msg) for progress updates.
        gpu: GPU hardware info (None means no GPU).
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
        gpu: GpuInfo | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._output_dir = Path(output_dir)
        self._config = config
        self._progress_callback = progress_callback
        self._gpu = gpu or GpuInfo()
        self._data = DataLoader(self._data_dir)
        self._transform = _TransformNamespace()

    @property
    def data_dir(self) -> Path:
        """Path to the job's input data directory."""
        return self._data_dir

    @property
    def output_dir(self) -> Path:
        """Path to the job's output directory."""
        return self._output_dir

    @property
    def config(self) -> dict[str, Any]:
        """Job configuration dict."""
        return self._config

    @property
    def data(self) -> DataLoader:
        """Data loader for the job's data directory."""
        return self._data

    @property
    def transform(self) -> _TransformNamespace:
        """Panel-aware data transformations."""
        return self._transform

    @property
    def gpu(self) -> GpuInfo:
        """GPU hardware information."""
        return self._gpu

    def progress(self, pct: float, msg: str = "") -> None:
        """Report progress to the orchestrator.

        Args:
            pct: Progress fraction (0.0 to 1.0).
            msg: Human-readable status message.
        """
        if self._progress_callback is not None:
            self._progress_callback(pct, msg)
