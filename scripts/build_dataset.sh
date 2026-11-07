#!/usr/bin/env bash
# Fetch every icon source and build data/processed (one command, idempotent fetch).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m iconsmith.fetch data/raw
uv run python -m iconsmith.dataset_build data/raw data/processed
