"""Thin simulation wrapper around croissant-sim."""

from functools import partial
from typing import NamedTuple

import croissant as cro
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from astropy.time import Time
from croissant.rotations import rotmat_to_eulerZYZ
from croissant.simulator import convolve
from croissant.simulator import correct_ground_loss as _cro_correct_ground_loss
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


def _build_orientation_fn(beam_L, sim_L, sampling, nside, eul_topo, beam_norm):
    """Return a JIT-compiled function for the time-independent part of one
    beam orientation.

    Compile-time constants (resolutions, frame-rotation angles, the beam
    normalization) are captured in the closure.  None of the returned
    function's arguments depend on the number of time samples, so it
    compiles **once** per call and handles every drive orientation
    without re-tracing -- regardless of how many groups
    :func:`simulate_path` makes or how large each group is.  The
    time-dependent sky convolution happens afterwards, in
    :func:`_run_orientation`, using croissant's own JIT-compiled
    ``convolve``, which is cheap to recompile per group size.
    """

    @jax.jit
    def _orient(beam_alm, euler_drive, dl_topo, horizon, quad_weights):
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

        return beam_eq_alm, fgnd

    return _orient


def _build_fgnd_fn(beam_L, sampling, nside):
    """Return a JIT-compiled ground-fraction function for one orientation.

    Mirrors steps 1-3 of :func:`_build_orientation_fn` exactly, so the
    returned values match the ``fgnd`` used internally by
    :func:`simulate`.
    """

    @jax.jit
    def _run(beam_alm, euler_drive, horizon, quad_weights, beam_norm):
        dl_drive = _generate_rotate_dls(beam_L, euler_drive[1])
        alm_rot = jax.vmap(
            partial(
                _rotate_flms,
                L=beam_L,
                rotation=euler_drive,
                dl_array=dl_drive,
            )
        )(beam_alm)

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

        pixel_masked = pixel * horizon[None]
        norm_above = jnp.einsum("ft...,t->f", pixel_masked, quad_weights)
        return 1.0 - norm_above / beam_norm

    return _run


def compute_fgnd(
    beam_data,
    freqs,
    elevations,
    azimuths,
    sampling="mwss",
    beam_kw=None,
    verbose=False,
):
    """Compute the ground fraction for multiple beam orientations.

    For each ``(elevation, azimuth)`` pair the beam is rotated by the
    drive model, masked by the horizon, and integrated; the ground
    fraction is one minus the above-horizon beam integral divided by
    the full-sphere beam integral (``croissant.Beam.compute_fgnd``
    generalized to rotated beams).  The pipeline replicates the masking
    steps inside :func:`simulate`, so the result matches the ground
    fraction that entered the simulated spectra.

    Use with :func:`correct_ground_loss` to recover the beam-weighted
    sky temperature from the output of :func:`simulate`.

    Parameters
    ----------
    beam_data : array_like
        Unrotated beam power pattern.
    freqs : array_like
        Frequencies in MHz.
    elevations : array_like
        Elevation angles in degrees, one per orientation.
    azimuths : array_like
        Azimuth angles in degrees, one per orientation.
    sampling : str
        Beam sampling scheme.
    beam_kw : dict or None
        Extra kwargs for ``croissant.Beam`` (e.g. *horizon*).
    verbose : bool
        Print per-orientation progress.

    Returns
    -------
    fgnd : jax.Array
        Ground fraction, shape ``(N_orientations, N_freqs)``.

    """
    beam_kw = beam_kw or {}

    beam_data = np.asarray(beam_data)
    lmax = cro.utils.lmax_from_ntheta(beam_data.shape[1], sampling)
    beam_L = lmax + 1
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(beam_data.shape[1])
    alm = beam_to_alm(beam_data, lmax, sampling, nside=nside)

    ref_beam = cro.Beam(beam_data, freqs, sampling=sampling, niter=0, **beam_kw)
    beam_norm = ref_beam.compute_norm()
    horizon = ref_beam.horizon
    if sampling == "healpix":
        npix = 12 * nside**2
        quad_weights = jnp.ones(npix) * (4 * jnp.pi / npix)
    else:
        quad_weights = s2fft.utils.quadrature_jax.quad_weights(
            L=beam_L, sampling=sampling, nside=nside
        )

    fgnd_one = _build_fgnd_fn(beam_L, sampling, nside)

    n_ori = len(elevations)
    results = []
    for i, (elev, az) in enumerate(zip(elevations, azimuths)):
        if verbose:
            print(f"    orientation {i + 1}/{n_ori}    ", end="\r", flush=True)

        R = drive_rotation_matrix(float(elev), float(az))
        euler = rotmat_to_eulerZYZ(R)
        euler_jax = jnp.asarray(euler, dtype=jnp.float64)

        results.append(fgnd_one(alm, euler_jax, horizon, quad_weights, beam_norm))

    if verbose:
        print()

    return jnp.stack(results)


def correct_ground_loss(t_sys, fgnd, Tgnd=None, t_rcvr=None, config=None):
    """Recover the sky temperature from a simulated system temperature.

    Subtracts the receiver temperature that :func:`simulate` adds, then
    applies ``croissant.simulator.correct_ground_loss``:
    ``t_sky = (t_ant - fgnd * Tgnd) / (1 - fgnd)``.

    Parameters
    ----------
    t_sys : array_like
        System temperature from :func:`simulate`, shape
        ``(N_orientations, N_times, N_freqs)``.
    fgnd : array_like
        Ground fraction from :func:`compute_fgnd`, shape
        ``(N_orientations, N_freqs)``.
    Tgnd : float or None
        Ground temperature in K.  ``None`` reads it from the config.
    t_rcvr : float or None
        Receiver temperature in K.  ``None`` reads it from the config.
        Pass ``0.0`` if *t_sys* is already an antenna temperature, as
        it is from :func:`simulate_path`.
    config : str, Path, or None
        Path to EIGSEP config YAML.  ``None`` uses the default.

    Returns
    -------
    t_sky : jax.Array
        Ground-loss-corrected sky temperature, same shape as *t_sys*.

    """
    cfg = load_config(config)
    if Tgnd is None:
        Tgnd = cfg["ground"]["temperature"]
    if t_rcvr is None:
        t_rcvr = cfg["receiver"]["temperature"]

    t_sys = jnp.asarray(t_sys)
    fgnd = jnp.asarray(fgnd)
    if fgnd.ndim == t_sys.ndim - 1:
        fgnd = jnp.expand_dims(fgnd, axis=-2)  # broadcast over the time axis

    return _cro_correct_ground_loss(t_sys - t_rcvr, fgnd, Tgnd)


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


def precompute_sky_alm(sky, times_jd, config=None):
    """Compute sky ALM for reuse across multiple simulate() calls.

    Returns the sky spherical harmonic coefficients in croissant's
    simulation frame for calls that start at ``times_jd[0]``: CIRS at
    that epoch on Earth, whose z axis is the Earth's rotation axis.
    Pass the result to ``simulate(sky_alm=...)`` to skip redundant sky
    transforms in each batch, but only for calls with the same first
    time.

    Parameters
    ----------
    sky : croissant.Sky
        Sky model.
    times_jd : array_like
        Times in Julian day of the calls the result is passed to.  Only
        ``times_jd[0]`` is used: like ``croissant.Simulator``, it fixes
        the frame's reference epoch.
    config : str, Path, or None
        Path to EIGSEP config YAML.  ``None`` uses the default.

    Returns
    -------
    sky_alm : jax.Array
        Sky ALM in the simulation frame, shape
        ``(N_freqs, lmax+1, 2*lmax+1)``.

    """
    cfg = load_config(config)
    # The epoch croissant.Simulator uses (its et_ref) for these times.
    t0 = Time(np.ravel(times_jd)[0], format="jd")
    et = cro.rotations.jd_to_et(t0.tdb.jd)
    return sky.compute_alm_eq(world=cfg["world"], et=et)


class _Setup(NamedTuple):
    """Everything one simulation shares across orientations."""

    alm: object
    orient: object
    dl_topo: object
    horizon: object
    quad_weights: object
    sky_alm: object
    phases: object
    beam_norm: object
    Tgnd: object
    t_rcvr: float


def _setup(beam_data, freqs, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw):
    """Precompute what every orientation of one simulation shares."""
    cfg = load_config(config)
    beam_kw = beam_kw or {}

    loc = cfg["location"]
    defaults = dict(
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )
    defaults.update(sim_kw)

    # Pre-compute the forward SHT of the unrotated beam once.
    beam_data = np.asarray(beam_data)
    lmax = cro.utils.lmax_from_ntheta(beam_data.shape[1], sampling)
    beam_L = lmax + 1
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(beam_data.shape[1])
    alm = beam_to_alm(beam_data, lmax, sampling, nside=nside)

    # Reference Simulator for the frame-rotation parameters.
    ref_beam = cro.Beam(beam_data, freqs, sampling=sampling, niter=0, **beam_kw)
    beam_norm = ref_beam.compute_norm()
    ref_sim = cro.Simulator(ref_beam, sky, times_jd, freqs, **defaults)
    if sky_alm is None:
        sky_alm = ref_sim.precompute_sky_alm()

    # Truncate dl_topo and sky_alm to the simulation resolution.
    sim_L = ref_sim.lmax + 1
    d = beam_L - sim_L
    end = d + 2 * sim_L - 1

    if sampling == "healpix":
        npix = 12 * nside**2
        quad_weights = jnp.ones(npix) * (4 * jnp.pi / npix)
    else:
        quad_weights = s2fft.utils.quadrature_jax.quad_weights(
            L=beam_L, sampling=sampling, nside=nside
        )

    return _Setup(
        alm=alm,
        orient=_build_orientation_fn(
            beam_L, sim_L, sampling, nside, ref_sim.eul_topo, beam_norm
        ),
        dl_topo=ref_sim.dl_topo[:sim_L, d:end, d:end],
        horizon=ref_beam.horizon,
        quad_weights=quad_weights,
        sky_alm=cro.utils.reduce_lmax(sky_alm, ref_sim.lmax),
        phases=ref_sim.phases,
        beam_norm=beam_norm,
        Tgnd=jnp.asarray(defaults["Tgnd"], dtype=jnp.float64),
        t_rcvr=cfg["receiver"]["temperature"],
    )


def _run_orientation(setup, elevation_deg, azimuth_deg, phases):
    """Antenna temperature for one orientation at the times of *phases*."""
    R = drive_rotation_matrix(float(elevation_deg), float(azimuth_deg))
    euler = jnp.asarray(rotmat_to_eulerZYZ(R), dtype=jnp.float64)
    beam_eq_alm, fgnd = setup.orient(
        setup.alm,
        euler,
        setup.dl_topo,
        setup.horizon,
        setup.quad_weights,
    )

    # Sky convolution happens outside the orientation graph: croissant's
    # convolve() is itself JIT-compiled and specialises on the number of
    # times, which is cheap, so it does not force the (expensive)
    # orientation graph above to retrace per group size.
    vis_sky = convolve(beam_eq_alm, setup.sky_alm, phases)
    vis_sky = vis_sky / setup.beam_norm[None, :]
    vis = vis_sky + fgnd * setup.Tgnd
    return vis.real


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
    setup = _setup(
        beam_data, freqs, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw
    )

    n_ori = len(elevations)
    results = []
    for i, (elev, az) in enumerate(zip(elevations, azimuths)):
        if verbose:
            print(f"    orientation {i + 1}/{n_ori}    ", end="\r", flush=True)
        results.append(_run_orientation(setup, elev, az, setup.phases))

    if verbose:
        print()

    return jnp.stack(results) + setup.t_rcvr


def simulate_path(
    beam_data,
    freqs_mhz,
    sky,
    times_jd,
    elevations_deg,
    azimuths_deg,
    config=None,
    sampling="mwss",
    beam_kw=None,
    sky_alm=None,
    verbose=False,
    **sim_kw,
):
    """Simulate one antenna orientation per time sample (D5 path mode).

    Motor telemetry gives one (elevation, azimuth) per sample. Samples
    are grouped by unique orientation and each group is simulated once,
    at that group's times only; D5 orientations repeat (static at night,
    a raster on Jul 17), so this is much cheaper than a full grid.

    Grouping is by exact equality of (elevation, azimuth): samples whose
    angles differ by encoder jitter or floating-point rounding land in
    separate groups and lose the speed-up. Pass block-median or
    quantised angles, not raw per-sample telemetry.

    Unlike :func:`simulate`, no receiver temperature is added: the
    result is the free-space antenna temperature, ``t_ant_k`` of
    ``SkyTemperature`` in the eigsep_cal interface spec (§ 5.1). It
    includes sky, horizon and the configured ground model, but no balun
    or coax.

    The orientation graph (beam rotation, horizon masking, ground
    fraction) compiles once per call, independent of group sizes. Only
    croissant's sky convolution specialises on the number of times in
    each group, and recompiling that per group size is cheap.

    Passing the result to :func:`correct_ground_loss` needs
    ``t_rcvr=0.0``: that function subtracts the config's receiver
    temperature by default, and path mode never added it.

    Parameters
    ----------
    beam_data : array_like
        Unrotated beam power pattern.
    freqs_mhz : array_like
        Frequencies in MHz, matching the beam's frequency axis.
    sky : croissant.Sky
        Sky model.
    times_jd : array_like
        Sample times in Julian day, shape ``(n_time,)``.
    elevations_deg, azimuths_deg : array_like
        Drive angles in degrees, one per sample, shape ``(n_time,)``.
    config : str, Path, or None
        Config for :func:`~eigsim.config.load_config`.
    sampling : str
        Beam sampling scheme.
    beam_kw : dict or None
        Extra kwargs for ``croissant.Beam`` (e.g. *horizon*).
    sky_alm : jax.Array or None
        Pre-computed sky ALM from :func:`precompute_sky_alm`.
    verbose : bool
        Print per-group progress.
    **sim_kw
        Override Simulator kwargs (lon, lat, alt, world, Tgnd, lmax).

    Returns
    -------
    t_ant : jax.Array
        Noiseless antenna temperature, shape ``(n_time, n_freq)``.

    """
    times_jd = np.asarray(times_jd, dtype=np.float64)
    elevations_deg = np.asarray(elevations_deg, dtype=np.float64)
    azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64)
    shapes = {times_jd.shape, elevations_deg.shape, azimuths_deg.shape}
    if len(shapes) != 1 or times_jd.ndim != 1:
        raise ValueError(
            "need one orientation per time: times_jd, elevations_deg and "
            f"azimuths_deg must be 1-D with equal length, got {sorted(shapes)}"
        )

    if times_jd.size == 0:
        raise ValueError("need at least one sample: times_jd is empty")

    if not (
        np.all(np.isfinite(times_jd))
        and np.all(np.isfinite(elevations_deg))
        and np.all(np.isfinite(azimuths_deg))
    ):
        raise ValueError("times_jd, elevations_deg and azimuths_deg must be finite")

    setup = _setup(
        beam_data, freqs_mhz, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw
    )

    orientations, group = np.unique(
        np.column_stack([elevations_deg, azimuths_deg]), axis=0, return_inverse=True
    )
    group = group.reshape(-1)

    perm = np.argsort(group, kind="stable")
    idx_per_group = np.split(
        perm, np.cumsum(np.bincount(group, minlength=len(orientations)))[:-1]
    )

    pieces = []
    for g, ((elev, az), idx) in enumerate(zip(orientations, idx_per_group)):
        if verbose:
            print(
                f"    orientation {g + 1}/{len(orientations)}    ", end="\r", flush=True
            )
        pieces.append(_run_orientation(setup, elev, az, setup.phases[idx]))

    if verbose:
        print()

    order = np.concatenate(idx_per_group)
    return jnp.concatenate(pieces)[np.argsort(order)]
