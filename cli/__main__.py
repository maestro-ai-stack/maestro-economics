"""mecon — RA Compute CLI for maestro-economics."""

import json
import os
import sys
import tempfile
import time
import tomllib
import zipfile
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
import httpx

CONFIG_DIR = os.path.expanduser("~/.mecon")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.toml")

DEFAULT_API_BASE = "https://ra.maestro.onl"
DEFAULT_PROFILE = "default"
API_PREFIX = "/api/ra/compute/v1"


def _read_all_profiles() -> dict:
    """Read all profiles from config.toml."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def _write_all_profiles(profiles: dict) -> None:
    """Write all profiles to config.toml (simple TOML serializer)."""
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    lines = []
    for name, vals in profiles.items():
        lines.append(f"[{name}]")
        for k, v in vals.items():
            lines.append(f'{k} = "{v}"')
        lines.append("")
    with open(CONFIG_FILE, "w") as f:
        f.write("\n".join(lines))
    os.chmod(CONFIG_FILE, 0o600)


def _get_active_profile() -> str:
    """Get active profile from: CLI --profile > env var > 'default'."""
    ctx = click.get_current_context(silent=True)
    if ctx and ctx.obj:
        p = ctx.obj.get("profile")
        if p:
            return p
    return os.environ.get("MECON_PROFILE", DEFAULT_PROFILE)


def get_config() -> dict:
    """Load config. Priority: CLI flags > env vars > profile in config.toml > defaults."""
    profiles = _read_all_profiles()
    profile_name = _get_active_profile()
    cfg = profiles.get(profile_name, profiles.get(DEFAULT_PROFILE, {}))

    # Env vars override profile values
    api_key = os.environ.get("MECON_API_KEY") or cfg.get("api_key", "")
    api_base = cfg.get("api_base", DEFAULT_API_BASE)

    # CLI --endpoint flag takes highest priority
    ctx = click.get_current_context(silent=True)
    if ctx and ctx.obj:
        endpoint = ctx.obj.get("endpoint_override")
        if endpoint:
            api_base = endpoint

    return {"api_key": api_key, "api_base": api_base}


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
@click.option("--profile", envvar="MECON_PROFILE", default=None,
              help="Config profile (default: 'default')")
@click.option("--endpoint", envvar="MECON_API_BASE", default=None,
              help="API endpoint URL override")
@click.pass_context
def main(ctx, profile, endpoint):
    """mecon -- RA Compute CLI for economists."""
    ctx.ensure_object(dict)
    if profile:
        ctx.obj["profile"] = profile
    if endpoint:
        ctx.obj["endpoint_override"] = endpoint


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@main.command()
@click.option("--profile", "profile_name", default=None,
              help="Profile name (default: 'default')")
def setup(profile_name: str | None):
    """Set API key and endpoint for a profile.

    Examples:
        mecon setup                     # configure [default] profile
        mecon setup --profile local     # configure [local] profile
        mecon setup --profile preview   # configure [preview] profile
    """
    name = profile_name or _get_active_profile()
    profiles = _read_all_profiles()
    existing = profiles.get(name, {})

    api_key = click.prompt("API key", default=existing.get("api_key", ""), show_default=False)
    default_base = existing.get("api_base", DEFAULT_API_BASE)
    api_base = click.prompt("API endpoint", default=default_base)

    profiles[name] = {"api_key": api_key, "api_base": api_base}
    _write_all_profiles(profiles)
    click.echo(f"Profile [{name}] saved to {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


@main.command()
def profiles():
    """List all configured profiles."""
    all_profiles = _read_all_profiles()
    active = _get_active_profile()
    if not all_profiles:
        click.echo("No profiles configured. Run 'mecon setup' to create one.")
        return
    for name, vals in all_profiles.items():
        marker = "*" if name == active else " "
        base = vals.get("api_base", DEFAULT_API_BASE)
        key = vals.get("api_key", "")
        key_display = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "(not set)"
        click.echo(f"  {marker} [{name}]  {base}  key={key_display}")


# ---------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------


@main.command()
def balance():
    """Show credit balance."""
    data = api("GET", "/balance")
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

MECONIGNORE_TEMPLATE = """# mecon ignore — files excluded from GPU job upload
# Raw data (convert to .parquet first)
*.mat
*.h5
*.hdf5
*.dta
*.npz
*.pkl
*.sav

# Results from previous runs
results/
output/
*.log

# Plots and figures
*.png
*.jpg
*.pdf

# OS / editor junk
.DS_Store
Thumbs.db
"""

RUN_PY_TEMPLATE = '''"""GPU estimation job — edit this file."""


def run(ctx):
    import pandas as pd

    ctx.progress(0.1, "Loading data")
    # df = pd.read_parquet(f"{ctx.data_dir}/panel.parquet")

    ctx.progress(0.5, "Running estimation")
    # ... your code here ...

    ctx.progress(1.0, "Done")
    return {
        "estimates": {},
        "diagnostics": {"converged": True},
    }
'''


@main.command()
@click.argument("name", default=".")
def init(name: str):
    """Initialize a GPU compute project directory.

    Creates the standard structure with run.py, data/, output/, .meconignore.

    Examples:
        mecon init my_estimation
        mecon init .                 # initialize current directory
    """
    project_dir = os.path.abspath(name)
    if name != "." and os.path.exists(project_dir):
        click.echo(f"Error: {project_dir} already exists.", err=True)
        sys.exit(1)

    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)

    # .meconignore
    ignore_path = os.path.join(project_dir, ".meconignore")
    if not os.path.exists(ignore_path):
        with open(ignore_path, "w") as f:
            f.write(MECONIGNORE_TEMPLATE)

    # run.py
    run_path = os.path.join(project_dir, "run.py")
    if not os.path.exists(run_path):
        with open(run_path, "w") as f:
            f.write(RUN_PY_TEMPLATE)

    click.echo(f"Initialized project at {project_dir}")
    click.echo("  run.py         — entry point (edit this)")
    click.echo("  data/          — put .csv/.parquet here")
    click.echo("  output/        — results written here (auto-uploaded)")
    click.echo("  .meconignore   — exclude patterns")
    click.echo()
    click.echo("Next: add data, edit run.py, then: mecon submit .")


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
@click.option("-o", "--output", default="./results", help="Output directory")
def download(job_id: str, output: str):
    """Download results for a completed job."""
    # Get job status first
    job_data = api("GET", f"/jobs/{job_id}")
    job = job_data.get("data", job_data) if isinstance(job_data, dict) else job_data
    status = job.get("status", "unknown") if isinstance(job, dict) else "unknown"

    if status != "completed":
        click.echo(f"Job status is '{status}'. Results available only for completed jobs.", err=True)
        if status == "failed":
            error = job.get("error_message", "") if isinstance(job, dict) else ""
            click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    # Get download URLs
    data = api("GET", f"/jobs/{job_id}/results")
    files = data.get("files", []) if isinstance(data, dict) else data
    if not files:
        # Try result_url from job directly
        result_url = job.get("result_url", "") if isinstance(job, dict) else ""
        if result_url:
            files = [{"name": "results.zip", "url": result_url}]
        else:
            click.echo("No result files available.")
            return

    os.makedirs(output, exist_ok=True)
    for finfo in files:
        url = finfo.get("url", "")
        name = finfo.get("name", finfo.get("filename", "result"))
        click.echo(f"Downloading {name}...")
        resp = httpx.get(url, timeout=300, follow_redirects=True)
        if resp.status_code >= 400:
            click.echo(f"  Download failed: HTTP {resp.status_code}", err=True)
            continue
        filepath = os.path.join(output, name)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        click.echo(f"  Saved to {filepath}")
        # Auto-extract zip files
        if name.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(output)
            os.unlink(filepath)
            click.echo(f"  Extracted to {output}/")
    click.echo("Download complete.")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


DEFAULT_IGNORE_PATTERNS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
}
DEFAULT_IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".mat", ".npz", ".pkl", ".h5", ".hdf5",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".log", ".DS_Store",
}


BANNED_EXTENSIONS = {".mat", ".h5", ".hdf5", ".dta", ".npz", ".pkl", ".sav"}


def _validate_project(dir_path: str, entry: str | None = None) -> None:
    """Validate project structure before submission. Exits on failure."""
    entry_file = entry or "run.py"
    errors = []

    # Check entry point exists
    if not os.path.exists(os.path.join(dir_path, entry_file)):
        errors.append(f"Entry point '{entry_file}' not found. Create it or use --entry.")

    # Check for banned large file formats
    banned_found = []
    for root, _dirs, files in os.walk(dir_path):
        _dirs[:] = [d for d in _dirs if d not in DEFAULT_IGNORE_PATTERNS]
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() in BANNED_EXTENSIONS:
                rel = os.path.relpath(os.path.join(root, fname), dir_path)
                banned_found.append(rel)

    if banned_found:
        errors.append(
            f"Found {len(banned_found)} raw data file(s) that should be converted to .parquet:\n"
            + "\n".join(f"    {f}" for f in banned_found[:5])
            + ("\n    ..." if len(banned_found) > 5 else "")
            + "\n  Convert with: df.to_parquet('data/name.parquet')"
            + "\n  Or add to .meconignore if not needed for this job."
        )

    # Check run.py has def run(ctx)
    entry_path = os.path.join(dir_path, entry_file)
    if os.path.exists(entry_path):
        with open(entry_path) as f:
            content = f.read()
        if "def run(ctx" not in content and "def run(" not in content:
            errors.append(f"'{entry_file}' must define: def run(ctx) -> dict")

    if errors:
        click.echo("Project validation failed:", err=True)
        for i, e in enumerate(errors, 1):
            click.echo(f"  {i}. {e}", err=True)
        click.echo("\nRun 'mecon init' to create a valid project structure.", err=True)
        sys.exit(1)


def _load_meconignore(dir_path: str) -> list[str]:
    """Load .meconignore patterns from project dir."""
    ignore_file = os.path.join(dir_path, ".meconignore")
    if not os.path.exists(ignore_file):
        return []
    with open(ignore_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _should_ignore(path: str, ignore_patterns: list[str]) -> bool:
    """Check if a relative path matches any ignore pattern."""
    import fnmatch
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def _zip_directory(dir_path: str) -> str:
    """Zip a directory into a temp .zip file, respecting .meconignore."""
    dir_path = os.path.abspath(dir_path)
    base_name = os.path.basename(dir_path.rstrip("/"))
    ignore_patterns = _load_meconignore(dir_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, prefix=f"{base_name}_")
    tmp.close()
    file_count = 0
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            # Skip default ignored dirs
            _dirs[:] = [d for d in _dirs
                        if d not in DEFAULT_IGNORE_PATTERNS and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                _, ext = os.path.splitext(fname)
                if ext in DEFAULT_IGNORE_EXTENSIONS:
                    continue
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, dir_path)
                if _should_ignore(arcname, ignore_patterns):
                    continue
                zf.write(full, arcname)
                file_count += 1
    zip_size = os.path.getsize(tmp.name)
    click.echo(f"  {file_count} files, {zip_size / 1024:.0f} KB")

    # Hard limit: 50MB. Data should be parquet/csv, not raw .mat/.h5
    max_size = 50 * 1024 * 1024
    if zip_size > max_size:
        os.unlink(tmp.name)
        click.echo(
            f"Error: zip is {zip_size / 1024 / 1024:.1f} MB (limit: 50 MB).\n"
            "Reduce data size:\n"
            "  - Convert .mat/.h5 to .parquet (extract only needed columns)\n"
            "  - Add large files to .meconignore\n"
            "  - Use --data for separate data files",
            err=True,
        )
        sys.exit(1)

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
        # Validate project structure
        _validate_project(target, entry)
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
