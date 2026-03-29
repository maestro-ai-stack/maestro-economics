"""Modal Cloud GPU worker for maestro-economics jobs.

Receives a job payload, reads script + data from R2 (FUSE mount),
executes via runtime/handler.py, writes results/logs/checkpoints
directly to R2.

GPU tiers: T4, L4, A10G, L40S, A100, H100 — each gets a dedicated function
so Modal can route to the right hardware class.
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal image — shared across all GPU tiers
# ---------------------------------------------------------------------------

economics_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core GPU stack
        "jax[cuda12]==0.5.0",
        "jaxlib==0.5.0",
        "jaxopt>=0.8",
        # Numerics
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.12",
        "pyarrow>=15.0",
        # Optimization
        "cma>=4.0",
        # Networking / API
        "httpx>=0.27",
        "fastapi[standard]",
        # Common research deps (pre-installed to avoid cold-start reinstalls)
        "statsmodels>=0.14",
        "scikit-learn>=1.4",
        "duckdb>=0.10",
        "matplotlib>=3.8",
    )
    .add_local_dir("runtime", "/app/runtime")
)

# ---------------------------------------------------------------------------
# R2 mount via CloudBucketMount (FUSE-based, lazy-loading)
# ---------------------------------------------------------------------------

r2_bucket = modal.CloudBucketMount(
    bucket_name="ra-compute-prod",
    bucket_endpoint_url="https://4a6e409c279e108484bed901bf0f2ddb.r2.cloudflarestorage.com",
    secret=modal.Secret.from_name("r2-credentials"),
)

app = modal.App("maestro-economics", image=economics_image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R2_MOUNT = Path("/r2")
LOCAL_WORK = Path("/tmp/mecon_job")

# GPU model → Modal GPU string + approximate VRAM
GPU_SPECS: dict[str, tuple[str, float]] = {
    "T4": ("T4", 16.0),
    "L4": ("L4", 24.0),
    "A10G": ("A10G", 24.0),
    "L40S": ("L40S", 48.0),
    "A100": ("A100", 80.0),
    "H100": ("H100", 80.0),
}


# ---------------------------------------------------------------------------
# Webhook helper (for DB status + credit reconciliation)
# ---------------------------------------------------------------------------

def _report_status(
    callback_url: str,
    job_id: str,
    status: str,
    progress: float = 0.0,
    message: str = "",
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """POST status update to the RA Suite callback webhook."""
    import httpx

    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    try:
        httpx.post(callback_url, json=payload, timeout=30)
    except Exception:
        pass  # Non-fatal — webhook is best-effort


# ---------------------------------------------------------------------------
# Core execution logic (shared by all GPU-tier functions)
# ---------------------------------------------------------------------------

def _run_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Read from R2 mount, execute script, write results to R2.

    R2 layout:
      /r2/uploads/{job_id}/project.zip   — CLI uploaded zip
      /r2/jobs/{job_id}/output/          — results written here
      /r2/jobs/{job_id}/checkpoints/     — checkpoint persistence
      /r2/jobs/{job_id}/logs/            — numbered log chunks
    """
    import sys
    sys.path.insert(0, "/app")

    from runtime.handler import execute_job

    job_id: str = payload["job_id"]
    callback_url: str = payload["callback_url"]
    gpu_type: str = payload.get("gpu_type", "L4")
    timeout_sec: int = payload.get("timeout", 3600)
    config: dict[str, Any] = payload.get("config", {})
    upload_files: list[str] = payload.get("upload_files", [])

    _, gpu_memory = GPU_SPECS.get(gpu_type, ("L4", 24.0))

    # -- R2 paths --
    r2_uploads = R2_MOUNT / "uploads" / job_id
    r2_job = R2_MOUNT / "jobs" / job_id
    r2_output = r2_job / "output"
    r2_checkpoints = r2_job / "checkpoints"
    r2_logs = r2_job / "logs"

    for d in (r2_output, r2_checkpoints, r2_logs):
        d.mkdir(parents=True, exist_ok=True)

    # Local workspace for code execution (Python imports need local FS)
    local_code = LOCAL_WORK / "code"
    if local_code.exists():
        shutil.rmtree(local_code)
    local_code.mkdir(parents=True, exist_ok=True)

    # -- 1. Load code from R2 --
    _report_status(callback_url, job_id, "downloading", 0.0, "Loading code from R2")

    # Write job metadata
    meta = {"job_id": job_id, "gpu_type": gpu_type, "timeout": timeout_sec, "config": config}
    (r2_job / "meta.json").write_text(json.dumps(meta, indent=2))

    workspace_path = payload.get("workspace_path")
    script_path = local_code / "run.py"
    data_dir = local_code / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if workspace_path:
        # Workspace mode: read from persistent R2 workspace
        r2_workspace = R2_MOUNT / workspace_path
        entry = config.pop("entry", "run.py")
        # Copy .py files to local (fast imports), leave data on R2 mount (lazy read)
        for f in r2_workspace.rglob("*"):
            if f.is_file():
                rel = f.relative_to(r2_workspace)
                if f.suffix in (".py", ".json", ".yaml", ".toml", ".txt"):
                    dest = local_code / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
        script_path = local_code / entry
        sys.path.insert(0, str(local_code))
        # Data stays on R2 mount for lazy parquet reads
        ws_data = r2_workspace / "data"
        if ws_data.exists() and any(ws_data.iterdir()):
            data_dir = ws_data
    else:
        # Legacy mode: find uploaded files
        zip_file = None
        script_file = None
        if r2_uploads.exists():
            for f in r2_uploads.iterdir():
                if f.suffix == ".zip":
                    zip_file = f
                elif f.suffix == ".py":
                    script_file = f

        if zip_file:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(local_code)
            entry = config.pop("entry", "run.py")
            entry_path = local_code / entry
            if not entry_path.exists():
                py_files = list(local_code.rglob("*.py"))
                raise ValueError(f"Entry point '{entry}' not found. Files: {py_files}")
            script_path = entry_path
            proj_data = local_code / "data"
            if proj_data.exists():
                data_dir = proj_data
            sys.path.insert(0, str(local_code))
        elif script_file:
            shutil.copy2(script_file, script_path)
        elif config.get("script_code"):
            script_path.write_text(config.pop("script_code"))
        else:
            raise ValueError(f"No script found in uploads for job {job_id}")

    # -- 2. Execute the user script --
    _report_status(callback_url, job_id, "running", 0.05, "Starting execution")

    def progress_cb(pct: float, msg: str) -> None:
        scaled = 0.05 + pct * 0.85
        _report_status(callback_url, job_id, "running", scaled, msg)

    try:
        result = execute_job(
            script_path=str(script_path),
            data_dir=str(data_dir),
            output_dir=str(r2_output),
            config=config,
            progress_cb=progress_cb,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory,
            timeout=timeout_sec,
            checkpoint_dir=str(r2_checkpoints),
            log_dir=str(r2_logs),
        )
    except TimeoutError as e:
        _report_status(callback_url, job_id, "failed", 0.0, str(e), error=str(e))
        return {"status": "timeout", "error": str(e)}
    except Exception as e:
        tb = traceback.format_exc()
        _report_status(callback_url, job_id, "failed", 0.0, str(e), error=tb)
        return {"status": "error", "error": tb}

    # -- 3. Report completion --
    _report_status(callback_url, job_id, "completed", 1.0, "Job finished", result=result)
    return {"status": "completed", "result": result}


# ---------------------------------------------------------------------------
# One @app.function per GPU tier, all with R2 mount
# ---------------------------------------------------------------------------

@app.function(gpu="T4", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_t4(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "T4")
    return _run_job(payload)


@app.function(gpu="L4", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_l4(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "L4")
    return _run_job(payload)


@app.function(gpu="A10G", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_a10g(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "A10G")
    return _run_job(payload)


@app.function(gpu="L40S", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_l40s(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "L40S")
    return _run_job(payload)


@app.function(gpu="A100", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_a100(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "A100")
    return _run_job(payload)


@app.function(gpu="H100", timeout=7200, retries=0, volumes={"/r2": r2_bucket})
def run_job_h100(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "H100")
    return _run_job(payload)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

GPU_FUNCTIONS = {
    "T4": run_job_t4,
    "L4": run_job_l4,
    "A10G": run_job_a10g,
    "L40S": run_job_l40s,
    "A100": run_job_a100,
    "H100": run_job_h100,
}


def dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    """Route a job payload to the correct GPU-tier function."""
    gpu_type = payload.get("gpu_type", "L4")
    fn = GPU_FUNCTIONS.get(gpu_type)
    if fn is None:
        raise ValueError(f"Unknown gpu_type: {gpu_type}. Valid: {list(GPU_FUNCTIONS)}")
    return fn.remote(payload)


# ---------------------------------------------------------------------------
# Web endpoint — HTTP interface for RA Suite API
# ---------------------------------------------------------------------------

@app.function(timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
    """HTTP endpoint: POST /submit_job with JSON payload.

    Called by RA Suite API. Spawns the GPU function asynchronously.
    """
    gpu_type = payload.get("gpu_type", "L4")
    fn = GPU_FUNCTIONS.get(gpu_type)
    if fn is None:
        return {"error": f"Unknown gpu_type: {gpu_type}", "valid": list(GPU_FUNCTIONS)}

    call = fn.spawn(payload)
    return {"call_id": call.object_id, "gpu_type": gpu_type, "status": "spawned"}
