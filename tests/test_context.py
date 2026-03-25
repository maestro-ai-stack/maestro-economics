"""Tests for runtime.context — JobContext, DataLoader, GpuInfo."""

import csv
import os
from pathlib import Path

import pandas as pd
import pytest

from runtime.context import JobContext, DataLoader, GpuInfo


# ---------------------------------------------------------------------------
# GpuInfo
# ---------------------------------------------------------------------------

class TestGpuInfo:
    def test_defaults_to_none(self):
        gpu = GpuInfo()
        assert gpu.type is None
        assert gpu.memory_gb is None

    def test_with_values(self):
        gpu = GpuInfo(type="RTX 4090", memory_gb=24.0)
        assert gpu.type == "RTX 4090"
        assert gpu.memory_gb == 24.0


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,year,x\nA,2000,1\nA,2001,2\n")
        loader = DataLoader(data_dir=tmp_path)
        df = loader.load("data.csv", entity="id", time="year")
        assert len(df) == 2
        assert df.attrs["entity"] == "id"
        assert df.attrs["time"] == "year"

    def test_load_parquet(self, tmp_path):
        pq_file = tmp_path / "data.parquet"
        pd.DataFrame({"id": [1, 2], "v": [10, 20]}).to_parquet(pq_file)
        loader = DataLoader(data_dir=tmp_path)
        df = loader.load("data.parquet", entity="id")
        assert len(df) == 2
        assert df.attrs["entity"] == "id"
        assert df.attrs.get("time") is None

    def test_load_missing_file_raises(self, tmp_path):
        loader = DataLoader(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.csv")

    def test_load_unsupported_format_raises(self, tmp_path):
        (tmp_path / "data.xlsx").write_text("fake")
        loader = DataLoader(data_dir=tmp_path)
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load("data.xlsx")


# ---------------------------------------------------------------------------
# JobContext
# ---------------------------------------------------------------------------

class TestJobContext:
    def test_paths(self, tmp_path):
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        data_dir.mkdir()
        out_dir.mkdir()
        ctx = JobContext(
            data_dir=data_dir,
            output_dir=out_dir,
            config={},
        )
        assert ctx.data_dir == data_dir
        assert ctx.output_dir == out_dir

    def test_config_access(self):
        cfg = {"n_simulations": 50, "seed": 42}
        ctx = JobContext(
            data_dir=Path("/tmp"),
            output_dir=Path("/tmp"),
            config=cfg,
        )
        assert ctx.config["n_simulations"] == 50
        assert ctx.config["seed"] == 42

    def test_data_loader_accessible(self, tmp_path):
        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
        )
        assert isinstance(ctx.data, DataLoader)

    def test_transform_access(self, tmp_path):
        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
        )
        # transform namespace should expose all transform functions
        assert callable(ctx.transform.winsorize)
        assert callable(ctx.transform.lag)
        assert callable(ctx.transform.merge)

    def test_progress_callback(self, tmp_path):
        calls = []

        def on_progress(pct, msg):
            calls.append((pct, msg))

        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
            progress_callback=on_progress,
        )
        ctx.progress(0.5, "halfway")
        ctx.progress(1.0, "done")
        assert len(calls) == 2
        assert calls[0] == (0.5, "halfway")
        assert calls[1] == (1.0, "done")

    def test_progress_no_callback(self, tmp_path):
        """progress() should not raise even without a callback."""
        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
        )
        ctx.progress(0.5, "ok")  # should not raise

    def test_gpu_info(self, tmp_path):
        gpu = GpuInfo(type="A100", memory_gb=80.0)
        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
            gpu=gpu,
        )
        assert ctx.gpu.type == "A100"
        assert ctx.gpu.memory_gb == 80.0

    def test_gpu_defaults(self, tmp_path):
        ctx = JobContext(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={},
        )
        assert ctx.gpu.type is None
        assert ctx.gpu.memory_gb is None
