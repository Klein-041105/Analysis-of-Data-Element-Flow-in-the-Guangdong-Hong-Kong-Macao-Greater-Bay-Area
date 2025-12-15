"""Pytest configuration helpers for local test runs.

This file ensures the project root is on `sys.path` so tests can import
the `src` package without requiring the project to be installed.

It is safe for CI and local runs and is a lightweight alternative to
setting `PYTHONPATH` externally or doing an editable install.
"""
import sys
from pathlib import Path


# Insert project root (two levels up from tests directory) to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
