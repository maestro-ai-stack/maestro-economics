"""mecon — RA Compute CLI for maestro-economics."""

import json
import os
import sys
import tempfile
import time
import zipfile
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
import httpx

CONFIG_DIR = os.path.expanduser("~/.mecon")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_API_BASE = "https://ra.maestro.onl"
API_PREFIX = "/api/ra/compute/v1"


def get_config() -> dict:
    """Load config from: CLI --endpoint > env var > config file > default."""
    api_key = os.environ.get("MECON_API_KEY")
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    # CLI --endpoint flag takes highest priority
    ctx = click.get_current_context(silent=True)
    endpoint = ctx.obj.get("endpoint_override") if ctx and ctx.obj else None
    return {
        "api_key": api_key or cfg.get("api_key", ""),
        "api_base": endpoint or cfg.get("api_base", DEFAULT_API_BASE),
    }


def api(
    method: str,
    path: str,
    json_data: dict | None = None,
    timeout: float = 30,
    raw: bool = False,
) -> dict | httpx.Response:
    """Make an authenticated API call. Exits on missing key or HTTP error."""
    cfg = get_config()
    if not cfg["api_key"]:
        click.echo("Error: No API key. Run 'mecon setup' first.", err=True)
        sys.exit(1)
    url = f"{cfg['api_base']}{API_PREFIX}{path}"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    try:
        resp = httpx.request(
            method, url, json=json_data, headers=headers, timeout=timeout
        )
    except httpx.ConnectError:
        click.echo(f"Error: cannot connect to {cfg['api_base']}. Check your network.", err=True)
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo("Error: request timed out. Try again or increase --timeout.", err=True)
        sys.exit(1)
    except httpx.HTTPError as exc:
        click.echo(f"Error: network request failed: {exc}", err=True)
        sys.exit(1)
    if resp.status_code >= 400:
        click.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        sys.exit(1)
    if raw:
        return resp
    return resp.json()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


def _get_version() -> str:
    try:
        return pkg_version("maestro-economics")
    except Exception:
        return "0.1.0"


@click.group()
@click.version_option(version=_get_version(), prog_name="mecon")
@click.option("--endpoint", envvar="MECON_API_BASE", default=None,
              help="API endpoint URL (default: https://ra.maestro.onl)")
@click.pass_context
def main(ctx, endpoint):
    """mecon -- RA Compute CLI for economists."""
    ctx.ensure_object(dict)
    if endpoint:
        ctx.obj["endpoint_override"] = endpoint


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@main.command()
def setup():
    """Set API key and base URL interactively."""
    api_key = click.prompt("API key", default="", show_default=False)
    api_base = click.prompt("API base URL", default=DEFAULT_API_BASE)
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({"api_key": api_key, "api_base": api_base}, f, indent=2)
    os.chmod(CONFIG_FILE, 0o600)
    click.echo(f"Config saved to {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------


@main.command()
def balance():
    """Show credit balance."""
    data = api("GET", "/balance")
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@main.command()
@click.argument("job_id")
def status(job_id: str):
    """Show job status JSON."""
    data = api("GET", f"/jobs/{job_id}")
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@main.command(name="list")
@click.option("--status", "job_status", default=None, help="Filter by status")
@click.option("--limit", default=10, help="Max jobs to show")
def list_jobs(job_status: str | None, limit: int):
    """List recent jobs."""
    params: dict = {"limit": limit}
    if job_status:
        params["status"] = job_status
    # Build query string manually
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    data = api("GET", f"/jobs?{qs}")
    jobs = data if isinstance(data, list) else data.get("jobs", [])
    if not jobs:
        click.echo("No jobs found.")
        return
    for job in jobs:
        jid = job.get("job_id", "")[:8]
        st = job.get("status", "unknown")
        created = job.get("created_at", "")
        click.echo(f"{jid}  {st:<12}  {created}")


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@main.command()
@click.argument("job_id")
def cancel(job_id: str):
    """Cancel a running job."""
    data = api("POST", f"/jobs/{job_id}/cancel")
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@main.command()
@click.argument("job_id")
@click.option("--interval", default=3, help="Poll interval in seconds")
def watch(job_id: str, interval: int):
    """Live-poll job progress until completion."""
    terminal_states = {"completed", "failed", "cancelled"}
    while True:
        data = api("GET", f"/jobs/{job_id}")
        st = data.get("status", "unknown")
        progress = data.get("progress", "")
        line = f"[{job_id[:8]}] {st}  {progress}"
        click.echo(f"\r{line:<60}", nl=False)
        if st in terminal_states:
            click.echo()  # newline
            if st == "completed":
                click.echo("Job completed.")
            elif st == "failed":
                click.echo(f"Job failed: {data.get('error', '')}")
            else:
                click.echo("Job cancelled.")
            break
        time.sleep(interval)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


@main.command()
@click.argument("job_id")
def logs(job_id: str):
    """Download and display job.log."""
    data = api("GET", f"/jobs/{job_id}/logs", raw=True)
    if isinstance(data, httpx.Response):
        click.echo(data.text)
    else:
        # API returned JSON with log content
        click.echo(data.get("content", json.dumps(data, indent=2)))


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@main.command()
@click.argument("job_id")
@click.option("-o", "--output", default=".", help="Output directory")
def download(job_id: str, output: str):
    """Download result files for a job."""
    data = api("GET", f"/jobs/{job_id}/results")
    files = data if isinstance(data, list) else data.get("files", [])
    if not files:
        click.echo("No result files available.")
        return
    os.makedirs(output, exist_ok=True)
    for finfo in files:
        url = finfo.get("url", "")
        name = finfo.get("name", finfo.get("filename", "unknown"))
        click.echo(f"Downloading {name}...")
        resp = httpx.get(url, timeout=120)
        filepath = os.path.join(output, name)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        click.echo(f"  Saved to {filepath}")
    click.echo("Download complete.")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _zip_directory(dir_path: str) -> str:
    """Zip a directory into a temp .zip file. Returns path to zip."""
    dir_path = os.path.abspath(dir_path)
    base_name = os.path.basename(dir_path.rstrip("/"))
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, prefix=f"{base_name}_")
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            # Skip hidden dirs, __pycache__, .git
            _dirs[:] = [d for d in _dirs if not d.startswith(".") and d != "__pycache__"]
            for fname in files:
                if fname.startswith(".") or fname.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, dir_path)
                zf.write(full, arcname)
    return tmp.name


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--data", "data_files", multiple=True, type=click.Path(exists=True),
              help="Data files to upload (repeatable)")
@click.option("--gpu", default="l4", help="GPU type (default: l4)")
@click.option("--timeout", "job_timeout", default=3600, type=int,
              help="Job timeout in seconds")
@click.option("--entry", default=None, help="Entry point file inside directory (default: run.py)")
@click.option("--no-watch", is_flag=True, help="Don't poll after submission")
def submit(target: str, data_files: tuple, gpu: str, job_timeout: int, entry: str | None, no_watch: bool):
    """Submit a job. TARGET can be a .py file or a directory.

    Single file:   mecon submit script.py
    Directory:     mecon submit ./my_project/
    With data:     mecon submit script.py --data input.csv
    """
    is_dir = os.path.isdir(target)
    zip_path = None

    if is_dir:
        # Zip the directory
        entry_file = entry or "run.py"
        entry_full = os.path.join(target, entry_file)
        if not os.path.exists(entry_full):
            click.echo(f"Error: entry point '{entry_file}' not found in {target}", err=True)
            click.echo("Use --entry to specify the main script.", err=True)
            sys.exit(1)
        click.echo(f"Zipping {target} (entry: {entry_file})...")
        zip_path = _zip_directory(target)
        zip_size = os.path.getsize(zip_path) / 1024
        click.echo(f"  {zip_size:.0f} KB")
        all_files = [zip_path] + list(data_files)
        filenames = [os.path.basename(zip_path)] + [os.path.basename(f) for f in data_files]
        job_meta = {"entry": entry_file}
    else:
        all_files = [target] + list(data_files)
        filenames = [os.path.basename(f) for f in all_files]
        job_meta = {}

    try:
        # 1. Create job
        click.echo("Creating job...")
        resp = api("POST", "/jobs", json_data={
            "gpu_type": gpu,
            "timeout_seconds": job_timeout,
            "files": filenames,
            **({"meta": job_meta} if job_meta else {}),
        })
        job = resp.get("data", resp)
        job_id = job["job_id"]
        click.echo(f"Job created: {job_id[:8]}")

        # 2. Upload files via presigned URLs
        upload_urls = job.get("upload_urls", {})
        for filepath, fname in zip(all_files, filenames):
            url = upload_urls.get(fname)
            if not url:
                click.echo(f"Warning: no upload URL for {fname}, skipping.", err=True)
                continue
            click.echo(f"Uploading {fname}...")
            with open(filepath, "rb") as f:
                content = f.read()
            put_resp = httpx.put(url, content=content, timeout=120)
            if put_resp.status_code >= 400:
                click.echo(f"Upload failed for {fname}: {put_resp.status_code}", err=True)
                sys.exit(1)

        # 3. Trigger run
        click.echo("Starting job...")
        api("POST", f"/jobs/{job_id}/run")

        if no_watch:
            click.echo(f"Job {job_id[:8]} submitted. Use 'mecon watch {job_id}' to monitor.")
            return

        # 4. Poll until done
        click.echo("Watching progress...")
        terminal_states = {"completed", "failed", "cancelled"}
        while True:
            time.sleep(3)
            data = api("GET", f"/jobs/{job_id}")
            st = data.get("status", "unknown")
            progress = data.get("progress", "")
            click.echo(f"\r[{job_id[:8]}] {st}  {progress:<40}", nl=False)
            if st in terminal_states:
                click.echo()
                if st == "completed":
                    click.echo("Job completed successfully.")
                elif st == "failed":
                    click.echo(f"Job failed: {data.get('error', '')}")
                break
    finally:
        if zip_path and os.path.exists(zip_path):
            os.unlink(zip_path)


# ---------------------------------------------------------------------------
# whoami
# ---------------------------------------------------------------------------


@main.command()
def whoami():
    """Show current configuration and account info."""
    cfg = get_config()
    api_base = cfg["api_base"]
    api_key = cfg["api_key"]
    key_display = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else ("(not set)" if not api_key else api_key)
    click.echo(f"API base:  {api_base}")
    click.echo(f"API key:   {key_display}")
    if api_key:
        try:
            url = f"{api_base}{API_PREFIX}/balance"
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = httpx.request("GET", url, headers=headers, timeout=10)
            if resp.status_code < 400:
                data = resp.json()
                if "email" in data:
                    click.echo(f"Account:   {data['email']}")
                if "balance" in data:
                    click.echo(f"Balance:   {data['balance']}")
            else:
                click.echo(f"Account:   (could not fetch — HTTP {resp.status_code})")
        except httpx.HTTPError:
            click.echo("Account:   (could not connect)")


if __name__ == "__main__":
    main()
