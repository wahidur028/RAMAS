#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment.json")
    parser.add_argument("--skip-upstream-tests", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(["sha256sum", "-c", "PACKAGE_MANIFEST.sha256"], cwd=ROOT)
    run([sys.executable, "-m", "py_compile",
         str(ROOT / "RAMAS_FULL_PIPELINE_ISOLATION.py"),
         str(ROOT / "RAMAS_VERIFY_RESULTS.py")])
    run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"])
    if not args.skip_upstream_tests:
        stage = Path(config["stage64_code_root"]).expanduser().resolve()
        tests = stage / "tests"
        if not tests.is_dir():
            raise SystemExit(f"Upstream Stage-6.4 tests are missing: {tests}")
        run([sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-v"], cwd=stage)
    print("SELF_TEST=PASS")


if __name__ == "__main__":
    main()
