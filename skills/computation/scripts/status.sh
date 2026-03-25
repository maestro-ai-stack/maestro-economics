#!/bin/bash
set -euo pipefail
[[ -z "${1:-}" ]] && { echo "Usage: status.sh <job_id>"; exit 1; }
mecon status "$1"
