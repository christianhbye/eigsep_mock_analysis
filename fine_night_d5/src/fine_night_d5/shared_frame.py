"""Validated reuse of eigsim's existing fixed simulation frame.

The old Simulator is a restricted frame carrier. Its stored beam/sky/freq
metadata must not supply any new-frequency physics: only replacement beam
and sky ALM enter the unchanged orientation/convolution operators.
"""

import hashlib
import importlib
import json

import numpy as np


def array_identity(value):
    a = np.ascontiguousarray(value)
    return dict(
        shape=list(a.shape),
        dtype=a.dtype.str,
        sha256=hashlib.sha256(a.tobytes()).hexdigest(),
    )


def sky_identity(sky):
    return {
        k: getattr(sky, k)
        for k in ["sampling", "nside", "lmax", "coord", "_niter", "_engine"]
    }


def rescale_beam(beam, current_norm, reference_norm):
    """Positive per-frequency gauge with unchanged normalized response."""
    b, n, r = map(np.asarray, (beam, current_norm, reference_norm))
    if (
        b.ndim != 3
        or n.shape != (len(b),)
        or r.shape != n.shape
        or not all(np.isfinite(a).all() for a in [b, n, r])
        or np.any(n <= 0)
        or np.any(r <= 0)
    ):
        raise ValueError("positive finite per-frequency beam norms required")
    return b * (r / n)[:, None, None]


class SharedFrame:
    """One fully pinned context and frequency-chunk size, opt-in only."""

    def __init__(self, beam, frequencies, sky, times, horizon, site, context):
        self.physics = importlib.import_module("eigsim.simulate")
        self.context = json.dumps(context, sort_keys=True, allow_nan=False)
        self.times = np.asarray(times).copy()
        self.horizon = np.asarray(horizon).copy()
        self.site = dict(site)
        self.config = self.physics.load_config(None)
        self.sky_context = sky_identity(sky)
        f = np.asarray(frequencies)
        self.size = len(f)
        self._validate_axes(beam, f, sky)
        self.template = self.physics._setup(
            np.asarray(beam),
            f,
            sky,
            self.times / 86400 + 2440587.5,
            None,
            "mwss",
            {"horizon": self.horizon},
            None,
            {**self.site, "Tgnd": 0.0},
            fuse_orientation=True,
        )
        self.current_sky_alm = self.template.sky_alm
        self.current_sky_frequencies = f.copy()
        self.current_sky_data_identity = array_identity(sky.data)
        self.reference_norm = np.asarray(self.template.beam_norm)
        if not np.isfinite(self.reference_norm).all() or np.any(
            self.reference_norm <= 0
        ):
            raise ValueError("invalid reference beam normalization")
        self.signature = dict(
            context=context,
            times=array_identity(self.times),
            horizon=array_identity(self.horizon),
            site=self.site,
            configuration=self.config,
            sky=self.sky_context,
            beam_lmax=128,
            simulation_lmax=int(self.template.sim.lmax),
            frequency_chunk_size=self.size,
        )

    def _validate_axes(self, beam, frequencies, sky):
        b, f = np.asarray(beam), np.asarray(frequencies)
        if (
            f.shape != (self.size,)
            or not np.isfinite(f).all()
            or np.any(np.diff(f) <= 0)
            or b.shape != (self.size, 130, 258)
            or not np.isfinite(b).all()
            or not np.array_equal(np.asarray(sky.freqs), f)
            or sky_identity(sky) != self.sky_context
        ):
            raise ValueError(
                "actual frequency/beam/sky axes or angular context changed"
            )

    def sky_alm(self, sky, frequencies, *, times, horizon, site, context):
        """Actual new-frequency sky in the unchanged first-epoch frame."""
        self._check_context(times, horizon, site, context)
        f = np.asarray(frequencies)
        if (
            f.shape != (self.size,)
            or not np.array_equal(np.asarray(sky.freqs), f)
            or sky_identity(sky) != self.sky_context
        ):
            raise ValueError("actual sky frequency/angular context changed")
        import croissant as cro

        actual = self.physics.precompute_sky_alm(sky, self.times / 86400 + 2440587.5)
        self.current_sky_alm = cro.utils.reduce_lmax(actual, self.template.sim.lmax)
        self.current_sky_frequencies = f.copy()
        self.current_sky_data_identity = array_identity(sky.data)
        return self.current_sky_alm

    def _check_context(self, times, horizon, site, context):
        if (
            not np.array_equal(times, self.times)
            or not np.array_equal(horizon, self.horizon)
            or site != self.site
            or json.dumps(context, sort_keys=True, allow_nan=False) != self.context
            or self.physics.load_config(None) != self.config
        ):
            raise ValueError("fixed simulation context changed")

    def replace(
        self, beam, frequencies, sky, sky_alm, *, times, horizon, site, context
    ):
        """Replace physical operands; retain the validated frame function."""
        self._check_context(times, horizon, site, context)
        self._validate_axes(beam, frequencies, sky)
        if (
            not np.array_equal(frequencies, self.current_sky_frequencies)
            or array_identity(sky.data) != self.current_sky_data_identity
        ):
            raise ValueError("sky ALM not bound to actual frequencies/data")
        sky_alm = self.current_sky_alm
        import croissant as cro
        from eigsim.rotations import beam_to_alm

        actual_norm = np.asarray(
            cro.Beam(
                beam, frequencies, sampling="mwss", niter=0, horizon=horizon
            ).compute_norm()
        )
        scaled = rescale_beam(beam, actual_norm, self.reference_norm)
        alm = beam_to_alm(scaled, 128, "mwss", nside=None)
        if (
            np.shape(sky_alm) != np.shape(self.template.sky_alm)
            or not np.isfinite(np.asarray(sky_alm)).all()
        ):
            raise ValueError("actual new-frequency sky ALM shape/coverage differs")
        return self.template._replace(alm=alm, sky_alm=sky_alm), {
            "actual_beam_norm": actual_norm,
            "reference_beam_norm": self.reference_norm,
            "rescale_factor": self.reference_norm / actual_norm,
        }
