"""Shared pytest fixtures and path setup.

Prepends the backend directory to sys.path so tests can import modules as
``from utils.csv_utils import ...`` without installing the package.
"""
import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
