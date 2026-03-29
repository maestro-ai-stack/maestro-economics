"""JobContext — the runtime context passed to user scripts via run(ctx)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import transforms


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
        elif suffix == ".dta":
            df = pd.DataFrame(pd.read_stata(path))
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
        checkpoint_dir: Path for checkpoint persistence (R2 mount).
        log_dir: Path for numbered log chunks (R2 mount, no append).
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
        gpu: GpuInfo | None = None,
        checkpoint_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._output_dir = Path(output_dir)
        self._config = config
        self._progress_callback = progress_callback
        self._gpu = gpu or GpuInfo()
        self._data = DataLoader(self._data_dir)
        self._transform = _TransformNamespace()
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self._output_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = Path(log_dir) if log_dir else self._output_dir / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_seq = self._next_log_seq()

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

    # -- Checkpoint API (persists to R2 via FUSE mount) --

    def save_checkpoint(self, **arrays: Any) -> None:
        """Save checkpoint to R2. FUSE-safe: writes directly, no temp+rename."""
        import io
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        path = self._checkpoint_dir / "checkpoint_latest.npz"
        path.write_bytes(buf.getvalue())

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load last checkpoint. Returns None if no checkpoint exists."""
        path = self._checkpoint_dir / "checkpoint_latest.npz"
        if not path.exists():
            return None
        import io
        data = io.BytesIO(path.read_bytes())
        return dict(np.load(data, allow_pickle=False))

    @property
    def has_checkpoint(self) -> bool:
        """True if a checkpoint exists from a previous run."""
        return (self._checkpoint_dir / "checkpoint_latest.npz").exists()

    # -- Log API (numbered chunks — FUSE can't append) --

    def _next_log_seq(self) -> int:
        """Find the next available log sequence number."""
        existing = sorted(self._log_dir.glob("log_*.txt"))
        if not existing:
            return 0
        last = existing[-1].stem  # e.g. "log_0042"
        return int(last.split("_")[1]) + 1

    def log(self, msg: str) -> None:
        """Write a log chunk to R2. Each call creates a new numbered file."""
        chunk = self._log_dir / f"log_{self._log_seq:04d}.txt"
        chunk.write_text(msg + "\n")
        self._log_seq += 1
