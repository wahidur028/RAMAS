#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/infonet/wahid/leader_router_fresh}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PACKAGE_DIR"
sha256sum -c MANIFEST.sha256
python -m py_compile run_real_adapter.py validate_pipeline.py src/*.py tests/*.py
python tests/run_all.py
python validate_pipeline.py --project-root "$PROJECT_ROOT"
python run_real_adapter.py --project-root "$PROJECT_ROOT"
sha256sum -c MANIFEST.sha256
