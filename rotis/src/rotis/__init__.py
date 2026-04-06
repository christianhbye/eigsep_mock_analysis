"""rotis: joint beam and sky inference for the EIGSEP experiment."""

import os
from importlib.metadata import version

os.environ.setdefault("JAX_ENABLE_X64", "1")

__version__ = version("rotis")
