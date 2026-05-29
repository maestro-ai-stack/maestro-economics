#!/usr/bin/env bash
set -euo pipefail

echo "== mecon binary =="
if ! command -v mecon >/dev/null 2>&1; then
  echo "mecon not found. Install with:"
  echo "python3 -m pip install --upgrade maestro-economics"
  exit 1
fi
command -v mecon

echo
echo "== mecon version =="
mecon --version

echo
echo "== submit resource support =="
if mecon submit --help | grep -q -- "--resource"; then
  echo "ok: mecon submit supports --resource"
else
  echo "missing: mecon submit does not expose --resource"
  echo "fix: python3 -m pip install --upgrade maestro-economics"
  exit 2
fi

echo
echo "== server resource profiles =="
mecon resources

echo
echo "== doctor =="
mecon doctor
