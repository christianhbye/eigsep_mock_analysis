"""An affine-in-mask temperature response, checked against public eigsim calls.

Private eigsim setup helpers are reused only to avoid rebuilding sky/beam state
for every mask. The independent comparison calls the public simulate_path API.
This module is study machinery, not a second temperature simulator.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np


def make_temperature_functions(
    beam,
    frequencies_mhz,
    sky,
    times_jd,
    elevations_deg,
    azimuths_deg,
    *,
    misalignment=None,
    sky_alm=None,
    ground_temperature_k=300.0,
    site=None,
):
    """Return cached-setup and public full functions T(W), both in kelvin."""
    simulation = importlib.import_module("eigsim.simulate")
    beam = np.asarray(beam, dtype=np.float64)
    times = np.asarray(times_jd, dtype=np.float64)
    el = np.asarray(elevations_deg, dtype=np.float64)
    az = np.asarray(azimuths_deg, dtype=np.float64)
    if times.ndim != 1 or times.shape != el.shape or times.shape != az.shape:
        raise ValueError(
            "time and pointing arrays must be equal one-dimensional shapes"
        )
    if len(times) == 0 or not np.all(np.isfinite(np.stack([times, el, az]))):
        raise ValueError("time and pointing arrays must be nonempty and finite")
    sim_kw = {**(site or {}), "Tgnd": ground_temperature_k}
    setup = simulation._setup(
        beam,
        frequencies_mhz,
        sky,
        times,
        None,
        "mwss",
        {"horizon": jnp.ones(beam.shape[1:], dtype=jnp.float64)},
        sky_alm,
        sim_kw,
    )
    orientations, group = np.unique(
        np.column_stack([el, az]), axis=0, return_inverse=True
    )
    group = group.reshape(-1)
    blocks = [np.flatnonzero(group == g) for g in range(len(orientations))]
    order = np.argsort(np.concatenate(blocks))

    def cached(weight):
        updated = setup._replace(horizon=jnp.asarray(weight))
        pieces = [
            simulation._run_orientation(updated, e, a, setup.phases[idx], misalignment)
            for (e, a), idx in zip(orientations, blocks, strict=True)
        ]
        return jnp.concatenate(pieces)[order]

    def direct(weight):
        return simulation.simulate_path(
            beam,
            frequencies_mhz,
            sky,
            times,
            el,
            az,
            beam_kw={"horizon": weight},
            sky_alm=setup.sky_alm,
            misalignment=misalignment,
            **sim_kw,
        )

    return cached, direct


def affine_response(model, mask_shape):
    """Return T(0) and dT/dW, which is constant for this fixed model."""
    zero = jnp.zeros(mask_shape, dtype=jnp.float64)
    offset = model(zero)
    kernel = jax.jacrev(model)(zero)
    return np.asarray(offset), np.asarray(kernel)


def apply_response(offset_k, kernel_k, weight):
    """Evaluate T(0)+K.W, allowing JAX differentiation through W."""
    return jnp.asarray(offset_k) + jnp.einsum(
        "tfij,ij->tf", jnp.asarray(kernel_k), jnp.asarray(weight)
    )
