"""Shared test setup: one in-process app (lifespan included) for every test module."""

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

# The app resolves api_specs/, collections/, frontend/ and logs/ relative to cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
# Build & run's usage ledger goes to a throwaway folder, never into the repo.
os.environ["BUILD_DATA_DIR"] = tempfile.mkdtemp(prefix="build-ledger-")

from backend.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # `with` runs the lifespan, which loads the knowledge base, RAG index and APIWeaver
    with TestClient(app) as c:
        yield c
