"""Thin simulation wrapper around croissant-sim."""

from functools import partial

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from croissant.rotations import rotmat_to_eulerZYZ
from croissant.simulator import convolve
from s2fft.recursions.risbo_jax import compute_full as _risbo_compute_full

from .config import load_config
from .rotations import beam_to_alm, drive_rotation_matrix, rotate_beam_data


def _generate_rotate_dls(L, beta):
    """``s2fft.generate_rotate_dls`` with *beta* as a dynamic arg."""
    dl = jnp.zeros((L, 2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=jnp.float64)
    for el in range(L):
        dl_iter = _risbo_compute_full(dl_iter, beta, L, el)
        dl = dl.at[el].add(dl_iter)
    return dl


def _rotate_flms(flm, L, rotation, dl_array):
    """``s2fft.utils.rotation.rotate_flms`` with *rotation* dynamic."""
    alpha = jnp.exp(-1j * jnp.arange(-L + 1, L) * rotation[0])
    gamma = jnp.exp(-1j * jnp.arange(-L + 1, L) * rotation[2])

    flm_rotated = jnp.zeros_like(flm)
    for el in range(L):
        n_max = min(el, L - 1)
        m = jnp.arange(-el, el + 1)
        n = jnp.arange(-n_max, n_max + 1)

        flm_rotated = flm_rotated.at[el, L - 1 + m].add(
            jnp.einsum(
                "mn,n->m",
                jnp.einsum(
                    "mn,m->mn",
                    dl_array[el, m + L - 1][:, n + L - 1],
                    alpha[m + L - 1],
                    optimize=True,
                ),
                gamma[n + L - 1] * flm[el, n + L - 1],
            )
        )
    return flm_rotated


def _build_orientation_fn(beam_L, sim_L, sampling, nside, eul_topo):
    """Return a JIT-compiled function for one beam orientation.

    Compile-time constants (resolutions, frame-rotation angles) are
    captured in the closure.  The returned function accepts only dynamic
    JAX arrays, so it compiles **once** and handles every drive
    orientation without re-tracing.
    """

    @jax.jit
    def _run(
        beam_alm,
        euler_drive,
        dl_topo,
        horizon,
        quad_weights,
        sky_alm,
        phases,
        beam_norm,
        Tgnd,
    ):
        # 1. Drive Wigner-D rotation at full beam resolution.
        dl_drive = _generate_rotate_dls(beam_L, euler_drive[1])
        alm_rot = jax.vmap(
            partial(
                _rotate_flms,
                L=beam_L,
                rotation=euler_drive,
                dl_array=dl_drive,
            )
        )(beam_alm)

        # 2. Inverse SHT to pixel space (needed for horizon masking).
        pixel = jax.vmap(
            partial(
                s2fft.inverse,
                L=beam_L,
                spin=0,
                nside=nside,
                sampling=sampling,
                method="jax",
                reality=True,
            )
        )(alm_rot)

        # 3. Horizon mask and ground fraction.
        pixel_masked = pixel * horizon[None]
        norm_above = jnp.einsum("ft...,t->f", pixel_masked, quad_weights)
        fgnd = 1.0 - norm_above / beam_norm

        # 4. Forward SHT back to harmonic space.
        alm_topo = jax.vmap(
            partial(
                s2fft.forward,
                L=beam_L,
                spin=0,
                nside=nside,
                sampling=sampling,
                method="jax",
                reality=True,
            )
        )(pixel_masked)

        # 5. Truncate to simulation lmax, then topo -> equatorial.
        #    Wigner-D is ell-by-ell so truncation commutes with rotation;
        #    doing it first is much cheaper when sim_L << beam_L.
        alm_trunc = cro.utils.reduce_lmax(alm_topo, sim_L - 1)
        beam_eq_alm = jax.vmap(
            partial(
                _rotate_flms,
                L=sim_L,
                rotation=eul_topo,
                dl_array=dl_topo,
            )
        )(alm_trunc)

        # 6. Convolve, normalize, add ground contribution.
        vis_sky = convolve(beam_eq_alm, sky_alm, phases)
        vis_sky /= beam_norm[None, :]
        vis = vis_sky + fgnd * Tgnd

        return vis.real

    return _run


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


def precompute_sky_alm(sky, config=None):
    """Compute sky ALM for reuse across multiple simulate() calls.

    Returns the sky spherical harmonic coefficients in FK5 equatorial
    coordinates.  Pass the result to ``simulate(sky_alm=...)`` to
    skip redundant sky transforms in each batch.

    Parameters
    ----------
    sky : croissant.Sky
        Sky model.
    config : str, Path, or None
        Path to EIGSEP config YAML.  ``None`` uses the default.

    Returns
    -------
    sky_alm : jax.Array
        Sky ALM in the equatorial frame, shape
        ``(N_freqs, lmax+1, 2*lmax+1)``.

    """
    cfg = load_config(config)
    return sky.compute_alm_eq(world=cfg["world"])


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
    sky_alm=None,
    verbose=False,
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
    sky_alm : jax.Array or None
        Pre-computed sky ALM from :func:`precompute_sky_alm`.  When
        ``None`` the sky ALM is computed internally.
    verbose : bool
        Print per-orientation progress.
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
    beam_L = lmax + 1
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(beam_data.shape[1])
    alm = beam_to_alm(beam_data, lmax, sampling, nside=nside)

    # Build a reference Simulator to obtain frame-rotation parameters.
    ref_beam = cro.Beam(beam_data, freqs, sampling=sampling, niter=0, **beam_kw)
    ref_sim = cro.Simulator(ref_beam, sky, times_jd, freqs, **defaults)

    if sky_alm is None:
        sky_alm = ref_sim.precompute_sky_alm()

    # --- Extract precomputed quantities from the reference objects ---
    sim_lmax = ref_sim.lmax
    sim_L = sim_lmax + 1
    eul_topo = ref_sim.eul_topo  # tuple of 3 floats, fixed for all orientations
    phases = ref_sim.phases  # (N_times, 2*sim_lmax+1)

    # Truncate dl_topo and sky_alm to the simulation resolution.
    d = beam_L - sim_L
    end = d + 2 * sim_L - 1
    dl_topo_sim = ref_sim.dl_topo[:sim_L, d:end, d:end]
    sky_alm_sim = cro.utils.reduce_lmax(sky_alm, sim_lmax)

    # Beam norm (rotation-invariant) and quadrature weights.
    beam_norm = ref_beam.compute_norm()
    horizon = ref_beam.horizon
    if sampling == "healpix":
        npix = 12 * nside**2
        quad_weights = jnp.ones(npix) * (4 * jnp.pi / npix)
    else:
        quad_weights = s2fft.utils.quadrature_jax.quad_weights(
            L=beam_L, sampling=sampling, nside=nside
        )
    Tgnd = jnp.asarray(defaults["Tgnd"], dtype=jnp.float64)

    # Build the JIT'd per-orientation pipeline (compiles once).
    sim_one = _build_orientation_fn(beam_L, sim_L, sampling, nside, eul_topo)

    n_ori = len(elevations)
    results = []
    for i, (elev, az) in enumerate(zip(elevations, azimuths)):
        if verbose:
            print(f"    orientation {i + 1}/{n_ori}    ", end="\r", flush=True)

        R = drive_rotation_matrix(float(elev), float(az))
        euler = rotmat_to_eulerZYZ(R)
        euler_jax = jnp.asarray(euler, dtype=jnp.float64)

        vis = sim_one(
            alm,
            euler_jax,
            dl_topo_sim,
            horizon,
            quad_weights,
            sky_alm_sim,
            phases,
            beam_norm,
            Tgnd,
        )
        results.append(vis)

    if verbose:
        print()

    t_ant = jnp.stack(results)
    return t_ant + t_rcvr
