"""RunPod Serverless handler + local execution engine."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import platform
import signal
import sys
from pathlib import Path
from typing import Any, Callable

from .context import GpuInfo, JobContext


class _Timeout:
    """Context manager for signal-based timeout (Unix only).

    On non-Unix platforms the timeout is silently ignored.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._is_unix = platform.system() != "Windows"

    def _handler(self, signum: int, frame: Any) -> None:
        raise TimeoutError(
            f"Job exceeded timeout of {self.seconds} seconds"
        )

    def __enter__(self) -> "_Timeout":
        if self._is_unix and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._is_unix and self.seconds > 0:
            signal.alarm(0)


def execute_job(
    script_path: str,
    data_dir: str,
    output_dir: str,
    config: dict[str, Any],
    progress_cb: Callable[[float, str], None],
    gpu_type: str | None = None,
    gpu_memory_gb: float | None = None,
    timeout: int = 0,
) -> dict[str, Any]:
    """Execute a user script's run(ctx) function.

    Loads the script via importlib, calls its ``run(ctx)`` entry point,
    captures stdout/stderr to ``job.log``, and saves the return dict
    as ``results.json`` in output_dir.

    Args:
        script_path: Absolute path to the user's Python script.
        data_dir: Path to the job's input data directory.
        output_dir: Path to the job's output directory (created if missing).
        config: Job configuration dict forwarded to the context.
        progress_cb: Callable(pct, msg) for progress updates.
        gpu_type: GPU model name (e.g. "RTX 4090"), or None.
        gpu_memory_gb: GPU VRAM in GB, or None.
        timeout: Maximum execution time in seconds (0 = no limit).

    Returns:
        Result dict from the user's ``run(ctx)`` function.

    Raises:
        ValueError: If the script cannot be loaded or lacks a ``run`` function.
    """
    os.makedirs(output_dir, exist_ok=True)

    gpu = GpuInfo(type=gpu_type, memory_gb=gpu_memory_gb)

    ctx = JobContext(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        config=config,
        progress_callback=progress_cb,
        gpu=gpu,
    )

    # Load user module
    if not os.path.isfile(script_path):
        raise ValueError(f"Cannot load script: {script_path}")
    spec = importlib.util.spec_from_file_location("user_script", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise ValueError("Script must define a run(ctx) function")

    # Capture stdout/stderr
    log_path = os.path.join(output_dir, "job.log")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    log_buffer = io.StringIO()

    try:
        sys.stdout = log_buffer
        sys.stderr = log_buffer
        with _Timeout(timeout):
            result = module.run(ctx)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        with open(log_path, "w") as f:
            f.write(log_buffer.getvalue())

    if not isinstance(result, dict):
        result = {"output": result}

    # Save results.json
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result
