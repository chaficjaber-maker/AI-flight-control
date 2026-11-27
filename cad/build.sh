#!/usr/bin/env bash
# Build Pantsir-S1 radar CAD locally (requires solidcraft-cli on PATH).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v solidcraft-cli >/dev/null 2>&1; then
  echo "solidcraft-cli not found. Install SolidCraft, then retry." >&2
  echo "See: https://docs.solidcraft.io/guide/getting-started.html" >&2
  exit 1
fi

solidcraft-cli run pantsir-s1-radar.json --save pantsir-s1-radar.sc
echo "Wrote pantsir-s1-radar.sc"

if command -v solidcraft >/dev/null 2>&1; then
  echo "Opening in SolidCraft..."
  exec solidcraft pantsir-s1-radar.sc
else
  echo "Open manually: solidcraft pantsir-s1-radar.sc"
fi