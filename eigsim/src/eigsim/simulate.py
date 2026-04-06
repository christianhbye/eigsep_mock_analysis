"""Thin simulation wrapper around croissant-sim."""

import croissant as cro
import jax.numpy as jnp
import numpy as np

from .config import load_config
from .rotations import beam_to_alm, rotate_alm_to_beam, rotate_beam_data


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

    *data* may be either a pixel-space array **or** pre-computed
    spherical-harmonic coefficients (a *jax.Array* returned by
    :func:`~eigsim.rotations.beam_to_alm`).  When alm are passed the
    forward SHT is skipped, which is faster when rotating the same
    beam to many orientations.

    Parameters
    ----------
    data : array_like
        Unrotated beam power pattern **or** pre-computed alm from
        :func:`~eigsim.rotations.beam_to_alm`.
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
    The receiver temperature from the config is added to the antenna
    temperature produced by the simulator, giving the total system
    temperature.

    The returned array is a JAX array suitable for automatic
    differentiation.  To add radiometer noise, use
    :func:`eigsim.noise.radiometer_noise` on the result.

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
    t_sys : jax.Array
        Noiseless system temperature (antenna + receiver), shape
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

    t_rcvr = cfg["receiver"]["temperature"]

    # Pre-compute the forward SHT of the unrotated beam once.
    beam_data = np.asarray(beam_data)
    lmax = cro.utils.lmax_from_ntheta(beam_data.shape[1], sampling)
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(beam_data.shape[1])
    alm = beam_to_alm(beam_data, lmax, sampling, nside=nside)

    # Pre-compute sky ALM once (avoids redundant SHT per orientation).
    first_beam = cro.Beam(beam_data, freqs, sampling=sampling, niter=0, **beam_kw)
    first_sim = cro.Simulator(first_beam, sky, times_jd, freqs, **defaults)
    sky_alm = first_sim.precompute_sky_alm()

    results = []
    for elev, az in zip(elevations, azimuths):
        if np.isclose(float(elev), 0.0) and np.isclose(float(az), 0.0):
            rotated = beam_data
        else:
            rotated = rotate_alm_to_beam(
                alm, lmax, sampling, float(elev), float(az), nside=nside
            )
        beam = cro.Beam(rotated, freqs, sampling=sampling, niter=0, **beam_kw)
        sim = cro.Simulator(beam, sky, times_jd, freqs, **defaults)
        results.append(sim.sim(sky_alm=sky_alm))

    t_ant = jnp.stack(results)
    return t_ant + t_rcvr
