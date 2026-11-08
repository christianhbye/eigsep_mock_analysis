"""Explicit existing034/eigsim sky and ground columns at actual native rows."""

import numpy as np


def drive_angles(rotations, misalignment):
    """Recover Rx(el)Rz(az) angles and prove the declared mount factorization."""
    rotations, mis = np.asarray(rotations), np.asarray(misalignment)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or mis.shape != (3, 3):
        raise ValueError("rotation/misalignment axes differ")
    if not np.isfinite(rotations).all() or not np.isfinite(mis).all():
        raise ValueError("missing native pointing blocks model coverage")
    if not np.allclose(mis @ mis.T, np.eye(3), rtol=0, atol=1e-10) or not np.isclose(
        np.linalg.det(mis), 1, rtol=0, atol=1e-10
    ):
        raise ValueError("outer mount must be a proper rotation")
    drive = mis.T @ rotations
    az = np.arctan2(-drive[:, 0, 1], drive[:, 0, 0])
    el = np.arctan2(-drive[:, 1, 2], drive[:, 2, 2])
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    z = np.zeros_like(az)
    reconstructed = np.stack(
        [
            np.stack([ca, -sa, z], -1),
            np.stack([ce * sa, ce * ca, -se], -1),
            np.stack([se * sa, se * ca, ce], -1),
        ],
        -2,
    )
    if not np.allclose(mis @ reconstructed, rotations, rtol=0, atol=1e-10):
        raise ValueError("rotation is not the pinned outer-mount Rx(el)Rz(az) operator")
    return np.degrees(el), np.degrees(az)


class EigsimColumns:
    """Callable evaluator for average_native, using unchanged existing physics.

    Supply source/hash-pinned beam, sky, open-sky weight, misalignment and
    Marjum site from the approved034 input contract. No GSM generation,
    compact-source replacement, loss or prior tuning occurs in this adapter.
    Native frequencies must equal the explicit beam/sky coordinate axis.
    Model calls retain ordinary eigsim grouping/JIT; cost is a run-gate item.
    """

    def __init__(
        self,
        frequencies_mhz,
        beam,
        sky,
        horizon_weight,
        misalignment,
        site,
        sky_alm=None,
    ):
        self.f = np.asarray(frequencies_mhz)
        self.beam = np.asarray(beam)
        self.sky = sky
        self.horizon = np.asarray(horizon_weight)
        self.mis = np.asarray(misalignment)
        self.site = dict(site)
        self.sky_alm = sky_alm
        if (
            self.f.ndim != 1
            or self.beam.ndim != 3
            or self.beam.shape[0] != len(self.f)
            or self.horizon.shape != self.beam.shape[1:]
            or set(self.site) != {"lat", "lon", "alt"}
            or not np.isfinite(self.beam).all()
            or not np.isfinite(self.horizon).all()
        ):
            raise ValueError("explicit beam/sky/horizon/site axes required")

    def __call__(self, times, frequencies_mhz, rotations):
        if not np.array_equal(frequencies_mhz, self.f):
            raise ValueError("model/native frequency axis differs")
        import eigsim

        el, az = drive_angles(rotations, self.mis)
        tjd = np.asarray(times) / 86400 + 2440587.5
        s = eigsim.simulate_path(
            self.beam,
            self.f,
            self.sky,
            tjd,
            el,
            az,
            sampling="mwss",
            beam_kw={"horizon": self.horizon},
            sky_alm=self.sky_alm,
            misalignment=self.mis,
            Tgnd=0.0,
            **self.site,
        )
        ori, inverse = np.unique(np.column_stack([el, az]), axis=0, return_inverse=True)
        fg = eigsim.compute_fgnd(
            self.beam,
            self.f,
            ori[:, 0],
            ori[:, 1],
            sampling="mwss",
            beam_kw={"horizon": self.horizon},
            misalignment=self.mis,
        )
        return np.stack([np.asarray(s), np.asarray(fg)[inverse]], axis=-1)
