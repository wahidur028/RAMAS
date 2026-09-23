#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    raise SystemExit(0 if result.wasSuccessful() else 1)
