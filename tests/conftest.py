import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
