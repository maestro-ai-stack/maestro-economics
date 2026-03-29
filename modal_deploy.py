#!/usr/bin/env python3
"""Deploy maestro-economics worker to Modal Cloud.

Usage:
    # Deploy all GPU-tier functions
    python modal_deploy.py deploy

    # Submit a test job (L4 by default)
    python modal_deploy.py test --gpu L4

    # Check deployed app status
    python modal_deploy.py status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy the Modal app."""
    subprocess.run(
        ["modal", "deploy", "modal_worker.py"],
        check=True,
        cwd=args.project_dir,
    )
    print("Deployed maestro-economics to Modal Cloud.")
    print("Functions: run_job_t4, run_job_l4, run_job_a10g, run_job_l40s, run_job_a100, run_job_h100")


def cmd_status(args: argparse.Namespace) -> None:
    """Show deployed app status."""
    subprocess.run(
        ["modal", "app", "list"],
        check=True,
    )


def cmd_test(args: argparse.Namespace) -> None:
    """Submit a test job via Modal's Python API."""
    # Inline import so the CLI doesn't require modal installed globally
    try:
        import modal  # noqa: F401
    except ImportError:
        print("Error: `modal` package not installed. Run: pip install modal", file=sys.stderr)
        sys.exit(1)

    from modal_worker import app, dispatch

    payload = {
        "job_id": "test-001",
        "gpu_type": args.gpu,
        "timeout": 120,
        "config": {},
        "callback_url": "https://httpbin.org/post",
        "r2_urls": {
            "script": args.script_url,
            "data_files": {},
            "upload": {
                "results.json": args.upload_url or "https://httpbin.org/put",
                "job.log": args.upload_url or "https://httpbin.org/put",
            },
        },
    }

    print(f"Submitting test job with GPU={args.gpu}")
    print(json.dumps(payload, indent=2))

    with app.run():
        result = dispatch(payload)
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy maestro-economics to Modal")
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Path to maestro-economics project root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("deploy", help="Deploy the Modal app")
    sub.add_parser("status", help="Show deployed app status")

    test_p = sub.add_parser("test", help="Submit a test job")
    test_p.add_argument("--gpu", default="L4", choices=["T4", "L4", "A10G", "L40S", "A100", "H100"])
    test_p.add_argument("--script-url", required=True, help="R2 presigned GET URL for the test script")
    test_p.add_argument("--upload-url", default=None, help="R2 presigned PUT URL for results")

    args = parser.parse_args()

    if args.command == "deploy":
        cmd_deploy(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
