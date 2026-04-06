"""Configure JAX float64 before test modules import jax.numpy."""

import jax

jax.config.update("jax_enable_x64", True)
