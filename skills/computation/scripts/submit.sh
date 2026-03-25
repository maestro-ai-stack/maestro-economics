#!/bin/bash
set -euo pipefail
CODE="" DATA="" GPU="4090" TIMEOUT="3600"
while [[ $# -gt 0 ]]; do
  case $1 in
    --code) CODE="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done
[[ -z "$CODE" ]] && { echo "Error: --code required"; exit 1; }
DATA_ARGS=""
[[ -n "$DATA" ]] && DATA_ARGS="--data $DATA"
mecon submit "$CODE" $DATA_ARGS --gpu "$GPU" --timeout "$TIMEOUT"
