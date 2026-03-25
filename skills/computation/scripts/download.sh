#!/bin/bash
set -euo pipefail
[[ -z "${1:-}" ]] && { echo "Usage: download.sh <job_id> [--output dir]"; exit 1; }
JOB_ID="$1"; shift
mecon download "$JOB_ID" "$@"
