"""Modal Cloud GPU worker for maestro-economics jobs.

Receives a job payload, downloads script + data from R2, executes via
the existing runtime/handler.py, uploads results back to R2, and reports
completion via webhook.

GPU tiers: T4, L4, A10G, L40S, A100, H100 — each gets a dedicated function
so Modal can route to the right hardware class.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
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
        "jax[cuda12]==0.5.0",
        "jaxlib==0.5.0",
        "jaxopt>=0.8",
        "numpy>=1.26",
        "pandas>=2.2",
        "httpx>=0.27",
        "pyarrow>=15.0",
        "fastapi[standard]",
    )
    .add_local_dir("runtime", "/app/runtime")
)

app = modal.App("maestro-economics", image=economics_image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORK_DIR = Path("/tmp/mecon_job")
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
SCRIPT_PATH = WORK_DIR / "script.py"

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
# R2 transfer helpers
# ---------------------------------------------------------------------------

def _download_from_r2(url: str, dest: Path) -> None:
    """Download a file from an R2 presigned GET URL."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)


def _upload_to_r2(url: str, src: Path) -> None:
    """Upload a file to an R2 presigned PUT URL."""
    import httpx

    with open(src, "rb") as f:
        data = f.read()
    resp = httpx.put(url, content=data, timeout=300)
    resp.raise_for_status()


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
        # Non-fatal — the job still ran, webhook is best-effort
        pass


# ---------------------------------------------------------------------------
# Core execution logic (shared by all GPU-tier functions)
# ---------------------------------------------------------------------------

def _run_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Download data, execute script, upload results, report status.

    Args:
        payload: Job descriptor with keys:
            job_id, r2_urls, callback_url, gpu_type, timeout, config

    Returns:
        Result dict from execute_job or an error dict.
    """
    import sys
    sys.path.insert(0, "/app")

    from runtime.handler import execute_job

    job_id: str = payload["job_id"]
    r2_urls: dict[str, Any] = payload["r2_urls"]
    callback_url: str = payload["callback_url"]
    gpu_type: str = payload.get("gpu_type", "L4")
    timeout_sec: int = payload.get("timeout", 3600)
    config: dict[str, Any] = payload.get("config", {})

    _, gpu_memory = GPU_SPECS.get(gpu_type, ("L4", 24.0))

    # Clean workspace
    for d in (DATA_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # -- 1. Download script + data files from R2 --
    _report_status(callback_url, job_id, "downloading", 0.0, "Downloading script and data")

    script_url = r2_urls.get("script", "")
    if not script_url:
        # Inline script mode: check for config["script_code"]
        script_code = config.pop("script_code", None)
        if script_code:
            SCRIPT_PATH.write_text(script_code)
        else:
            raise ValueError("No script URL or inline script_code provided")
    elif script_url.endswith(".zip") or r2_urls.get("is_zip"):
        # Zip project mode: download zip, extract, find entry point
        zip_dest = WORK_DIR / "project.zip"
        _download_from_r2(script_url, zip_dest)
        project_dir = WORK_DIR / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(project_dir)
        zip_dest.unlink()
        # Entry point: from config meta, or default run.py
        entry = config.pop("entry", "run.py")
        entry_path = project_dir / entry
        if not entry_path.exists():
            raise ValueError(f"Entry point '{entry}' not found in zip. Files: {list(project_dir.rglob('*.py'))}")
        # Copy entry to SCRIPT_PATH, add project dir to sys.path
        import shutil
        shutil.copy2(entry_path, SCRIPT_PATH)
        sys.path.insert(0, str(project_dir))
    else:
        _download_from_r2(script_url, SCRIPT_PATH)

    data_files: dict[str, str] = r2_urls.get("data_files", {})
    for filename, url in data_files.items():
        _download_from_r2(url, DATA_DIR / filename)

    # -- 2. Execute the user script --
    _report_status(callback_url, job_id, "running", 0.05, "Starting execution")

    def progress_cb(pct: float, msg: str) -> None:
        # Scale user progress (0-1) into our 0.05-0.90 range
        scaled = 0.05 + pct * 0.85
        _report_status(callback_url, job_id, "running", scaled, msg)

    # Timeout via SIGALRM
    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Job exceeded {timeout_sec}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)

    try:
        result = execute_job(
            script_path=str(SCRIPT_PATH),
            data_dir=str(DATA_DIR),
            output_dir=str(OUTPUT_DIR),
            config=config,
            progress_cb=progress_cb,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory,
        )
    except TimeoutError as e:
        _report_status(callback_url, job_id, "failed", 0.0, str(e), error=str(e))
        return {"status": "timeout", "error": str(e)}
    except Exception as e:
        tb = traceback.format_exc()
        _report_status(callback_url, job_id, "failed", 0.0, str(e), error=tb)
        return {"status": "error", "error": tb}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # -- 3. Save result dict as JSON in output dir --
    import json as _json
    results_json = OUTPUT_DIR / "results.json"
    results_json.write_text(_json.dumps(result, indent=2, default=str))

    # -- 4. Zip output dir and upload to R2 --
    _report_status(callback_url, job_id, "uploading", 0.90, "Uploading results")

    results_zip = WORK_DIR / "results.zip"
    with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for output_file in OUTPUT_DIR.rglob("*"):
            if output_file.is_file():
                zf.write(output_file, output_file.relative_to(OUTPUT_DIR))

    # Upload results zip if URL provided
    upload_urls: dict[str, str] = r2_urls.get("upload", {})
    results_url = upload_urls.get("results.zip")
    if results_url:
        _upload_to_r2(results_url, results_zip)

    # Also upload individual files if URLs provided
    for output_file in OUTPUT_DIR.iterdir():
        if output_file.is_file() and output_file.name in upload_urls:
            _upload_to_r2(upload_urls[output_file.name], output_file)

    # -- 5. Report completion --
    _report_status(callback_url, job_id, "completed", 1.0, "Job finished", result=result)

    return {"status": "completed", "result": result}


# ---------------------------------------------------------------------------
# One @app.function per GPU tier
#
# Modal does not support dynamic gpu= at call time, so we define a function
# per tier. The deploy script or API dispatches to the right one.
# ---------------------------------------------------------------------------

@app.function(gpu="T4", timeout=7200, retries=0)
def run_job_t4(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "T4")
    return _run_job(payload)


@app.function(gpu="L4", timeout=7200, retries=0)
def run_job_l4(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "L4")
    return _run_job(payload)


@app.function(gpu="A10G", timeout=7200, retries=0)
def run_job_a10g(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "A10G")
    return _run_job(payload)


@app.function(gpu="L40S", timeout=7200, retries=0)
def run_job_l40s(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "L40S")
    return _run_job(payload)


@app.function(gpu="A100", timeout=7200, retries=0)
def run_job_a100(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "A100")
    return _run_job(payload)


@app.function(gpu="H100", timeout=7200, retries=0)
def run_job_h100(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("gpu_type", "H100")
    return _run_job(payload)


# ---------------------------------------------------------------------------
# Dispatch helper — call from Python or the deploy script
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
    """Route a job payload to the correct GPU-tier function.

    Usage (remote call):
        with app.run():
            result = dispatch(payload)
    """
    gpu_type = payload.get("gpu_type", "L4")
    fn = GPU_FUNCTIONS.get(gpu_type)
    if fn is None:
        raise ValueError(f"Unknown gpu_type: {gpu_type}. Valid: {list(GPU_FUNCTIONS)}")
    return fn.remote(payload)


# ---------------------------------------------------------------------------
# Web endpoint — HTTP interface for RA Suite API to call
# ---------------------------------------------------------------------------

@app.function(timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
    """HTTP endpoint: POST /submit_job with JSON payload.

    Called by RA Suite API (modal.ts submitToModal).
    Spawns the GPU function asynchronously and returns the call ID.
    """
    gpu_type = payload.get("gpu_type", "L4")
    fn = GPU_FUNCTIONS.get(gpu_type)
    if fn is None:
        return {"error": f"Unknown gpu_type: {gpu_type}", "valid": list(GPU_FUNCTIONS)}

    # Spawn async (non-blocking) — returns immediately with call ID
    call = fn.spawn(payload)
    return {"call_id": call.object_id, "gpu_type": gpu_type, "status": "spawned"}
