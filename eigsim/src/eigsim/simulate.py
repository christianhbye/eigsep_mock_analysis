"""Thin simulation wrapper around croissant-sim."""

import croissant as cro
import jax.numpy as jnp
import numpy as np

from .config import load_config
from .rotations import rotate_beam_data


def make_beam(
    data,
    freqs,
    sampling="mwss",
    elevation_deg=0.0,
    azimuth_deg=0.0,
    niter=0,
    **beam_kw,
):
    """Create a ``croissant.Beam`` with the pattern rotated by the drives.

    When both drive angles are zero the beam is returned unrotated.
    Any extra keyword arguments are forwarded to ``croissant.Beam``
    (e.g. *horizon*).  Do **not** pass ``beam_rot`` here — the full
    rotation is handled by the EIGSEP drive model.

    Parameters
    ----------
    data : array_like
        Unrotated beam power pattern.
    freqs : array_like
        Frequencies in MHz.
    sampling : str
        Sampling scheme.
    elevation_deg : float
        Elevation drive angle in degrees.
    azimuth_deg : float
        Turntable angle in degrees.
    niter : int
        Number of SHT iterations.
    **beam_kw
        Forwarded to ``croissant.Beam``.

    Returns
    -------
    beam : croissant.Beam

    """
    lmax = cro.utils.lmax_from_ntheta(np.asarray(data).shape[1], sampling)
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(np.asarray(data).shape[1])

    if not (np.isclose(elevation_deg, 0.0) and np.isclose(azimuth_deg, 0.0)):
        data = rotate_beam_data(
            data,
            lmax,
            sampling,
            elevation_deg,
            azimuth_deg,
            nside=nside,
            niter=niter,
        )

    return cro.Beam(data, freqs, sampling=sampling, niter=niter, **beam_kw)


def simulate(
    beam_data,
    freqs,
    sky,
    times_jd,
    elevations,
    azimuths,
    config=None,
    sampling="mwss",
    beam_kw=None,
    **sim_kw,
):
    """Run simulations for multiple beam orientations.

    For each ``(elevation, azimuth)`` pair the beam is rotated, a
    ``croissant.Simulator`` is constructed, and ``sim()`` is called.
    Simulator parameters are read from the EIGSEP config YAML and can
    be overridden via *sim_kw*.

    Parameters
    ----------
    beam_data : array_like
        Unrotated beam power pattern.
    freqs : array_like
        Frequencies in MHz.
    sky : croissant.Sky
        Sky model.
    times_jd : array_like
        Observation times in Julian day.
    elevations : array_like
        Elevation angles in degrees, one per orientation.
    azimuths : array_like
        Azimuth angles in degrees, one per orientation.
    config : str, Path, or None
        Path to EIGSEP config YAML.  ``None`` uses the default.
    sampling : str
        Beam sampling scheme.
    beam_kw : dict or None
        Extra kwargs for ``croissant.Beam``.
    **sim_kw
        Override Simulator kwargs (lon, lat, alt, world, Tgnd, lmax).

    Returns
    -------
    visibilities : jax.Array
        Simulated antenna temperatures, shape
        ``(N_orientations, N_times, N_freqs)``.

    """
    cfg = load_config(config)
    beam_kw = beam_kw or {}

    # Simulator defaults from config, with caller overrides
    loc = cfg["location"]
    defaults = dict(
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )
    defaults.update(sim_kw)

    results = []
    for elev, az in zip(elevations, azimuths):
        beam = make_beam(
            beam_data,
            freqs,
            sampling=sampling,
            elevation_deg=float(elev),
            azimuth_deg=float(az),
            **beam_kw,
        )
        sim = cro.Simulator(beam, sky, times_jd, freqs, **defaults)
        results.append(sim.sim())

    return jnp.stack(results)
