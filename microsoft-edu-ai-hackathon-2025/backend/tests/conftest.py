"""Shared pytest fixtures and path setup.

Prepends the backend directory to sys.path so tests can import modules as
``from utils.csv_utils import ...`` without installing the package.
"""
import os
import sys

# Disable rate limiting during tests: it is verified separately, and its
# in-memory storage spawns a background cleanup thread that conflicts with
# tests that monkeypatch threading.Thread.
os.environ.setdefault("RATELIMIT_ENABLED", "false")

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
