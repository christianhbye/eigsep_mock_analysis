"""Pointing-only lag/play/transient injections on declared drive paths.

Drive and play window moments are analytic. Transient, harmonic and windup
terms use microbin resolution and require a 32/64-bin sensitivity check.
The generator does not fit data or establish a telemetry sampling model.
"""

import numpy as np

from .drive import _moves, uniform_micro_means, uniform_window_means


def play_model(moves, *, y0=0.0, zeta_p_deg=0.0, initial_offset=None):
    """Return the exact play-path moves for monotone constant-rate legs.

    Gear play holds the box until the drive traverses the gap, then moves
    at the drive rate. Short moves may never take up the play. The default
    initial offset assumes arrival from the direction opposite the first
    move, so it starts at ``y0 + sign(first rate)*zeta``. This is a declared
    synthetic initial condition, not an inference about any hardware.
    """
    table = _moves(moves)
    if not np.isfinite([y0, zeta_p_deg]).all() or zeta_p_deg < 0:
        raise ValueError("finite initial angle and nonnegative play required")
    if initial_offset is None:
        initial_offset = np.sign(table[0, 2]) * zeta_p_deg if len(table) else 0.0
    if not np.isfinite(initial_offset) or abs(initial_offset) > zeta_p_deg:
        raise ValueError("initial offset must lie inside the play interval")
    box_initial = y0 + initial_offset
    drive, box = float(y0), float(box_initial)
    box_moves = []
    for start, stop, rate in table:
        sign = np.sign(rate)
        takeup = max(0.0, (sign * (box - drive) + zeta_p_deg) / abs(rate))
        if start + takeup < stop:
            box_moves.append((start + takeup, stop, rate))
        drive += rate * (stop - start)
        box = float(np.clip(box, drive - zeta_p_deg, drive + zeta_p_deg))
    return dict(
        moves=np.asarray(box_moves, float).reshape(-1, 3),
        y0=float(box_initial),
        zeta_p_deg=float(zeta_p_deg),
        initial_offset=float(initial_offset),
    )


def ar1_noise(n, sd, rho, *, rng):
    """Draw stationary per-row AR(1) noise with marginal sd, not innovation sd."""
    if int(n) != n or n < 0 or not np.isfinite(sd) or sd < 0:
        raise ValueError("nonnegative integer length and noise sd required")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("stationary AR(1) requires abs(rho) < 1")
    if n == 0:
        return np.empty(0)
    normals = rng.normal(size=int(n))
    result = np.empty(int(n))
    result[0] = sd * normals[0]
    innovation_sd = sd * np.sqrt(1 - rho**2)
    for i in range(1, int(n)):
        result[i] = rho * result[i - 1] + innovation_sd * normals[i]
    return result


def simulate_windows(
    edges,
    moves,
    *,
    y0=0.0,
    tau_s=0.0,
    zeta_p_deg=0.0,
    T0_deg=0.0,
    d_rev_deg=30.0,
    eps=0.0,
    phi=None,
    windup_deg=0.0,
    ar1=None,
    rng=None,
    n_sub=64,
    ramp_deg=100 * 180 / 11300,
    initial_offset=None,
    macro_moves=None,
):
    """Inject declared pointing effects, then average on the supplied windows.

    Positive ``tau_s`` means the reported drive is late: the physical drive
    is evaluated at report time + tau. Play moments are computed from the
    exact shortened moves. The transient is signed T0*exp(-advance/d_rev),
    rising from zero over ``ramp_deg`` and present only during motion.
    Windup is -sign*W only in cruise, outside both ramp-length edge regions.

    ``phi`` has rows (cosine, sine) coefficients in degrees for harmonics
    1..K of the physical attitude before this angular transfer. Apply
    ``(1-eps)*attitude + phi(attitude)``. Microbin transformation error is
    distinct from the exact drive/play moments. AR(1) noise is added after
    averaging. Returned angles are unwrapped degrees; the analysis caller
    owns any sensor-angle wrapping. No receiver, sky, file I/O or fitted
    estimator is involved.
    """
    table = _moves(moves).copy()
    macro = None if macro_moves is None else _moves(macro_moves).copy()
    if macro is not None and (T0_deg != 0 or windup_deg != 0):
        raise ValueError("batch transient and windup response are not defined")
    scalars = [tau_s, T0_deg, d_rev_deg, eps, windup_deg, ramp_deg]
    if not np.isfinite(scalars).all() or d_rev_deg <= 0 or ramp_deg <= 0:
        raise ValueError("finite parameters and positive decay/ramp scales required")
    coeff = np.empty((0, 2)) if phi is None else np.asarray(phi, float)
    if coeff.size == 0:
        coeff = np.empty((0, 2))
    if coeff.ndim != 2 or coeff.shape[1] != 2 or not np.isfinite(coeff).all():
        raise ValueError("phi needs finite (cosine, sine) harmonic rows")
    fine = uniform_micro_means(edges, table, y0=y0, n_sub=n_sub)
    table[:, :2] -= fine["time_origin"]
    box_model = play_model(
        table, y0=y0, zeta_p_deg=zeta_p_deg, initial_offset=initial_offset
    )
    box_mean = uniform_window_means(
        fine["edges_relative"] + tau_s, box_model["moves"], y0=box_model["y0"]
    )
    true_time = fine["time_relative"] + tau_s
    sign = np.zeros(len(true_time))
    advance = np.zeros(len(true_time))
    transient = np.zeros(len(true_time))
    windup = np.zeros(len(true_time))
    for start, stop, rate in table:
        latest = true_time >= start
        distance = abs(rate) * np.clip(true_time - start, 0, stop - start)
        sign[latest] = np.sign(rate)
        advance[latest] = distance[latest]
        moving = (true_time >= start) & (true_time < stop)
        transient[moving] = (
            np.sign(rate)
            * T0_deg
            * np.exp(-distance[moving] / d_rev_deg)
            * np.clip(distance[moving] / ramp_deg, 0, 1)
        )
        cruise = moving & (distance >= ramp_deg)
        cruise &= (abs(rate) * (stop - start) - distance) >= ramp_deg
        windup[cruise] = -np.sign(rate) * windup_deg
    if macro is not None:
        macro[:, :2] -= fine["time_origin"]
        true_drive = uniform_window_means(fine["edges_relative"] + tau_s, table, y0=y0)
        start_angle = float(y0)
        for start, stop, rate in macro:
            latest = true_time >= start
            advance[latest] = np.maximum(
                np.sign(rate) * (true_drive[latest] - start_angle), 0
            )
            start_angle += rate * (stop - start)
    attitude = box_mean + transient + windup
    transformed = (1 - eps) * attitude
    for order, (cosine, sine) in enumerate(coeff, 1):
        phase = np.deg2rad(attitude) * order
        transformed += cosine * np.cos(phase) + sine * np.sin(phase)
    n_sub = fine["n_sub"]
    el = transformed.reshape(-1, n_sub).mean(axis=1)
    noise = np.zeros(len(el))
    if ar1 is not None:
        rng = np.random.default_rng() if rng is None else rng
        noise = ar1_noise(len(el), *ar1, rng=rng)
        el += noise
    return dict(
        el=el,
        drive_mean=fine["y_mean"].reshape(-1, n_sub).mean(axis=1),
        advance=advance.reshape(-1, n_sub).mean(axis=1),
        noise=noise,
        fine=dict(
            time=fine["time"],
            drive_mean=fine["y_mean"],
            box_mean=box_mean,
            attitude_mean=transformed,
            direction=sign,
            transient=transient,
            windup=windup,
        ),
        n_sub=n_sub,
        assumption="declared continuous idealized drive; not sparse telemetry sampling",
        initial_offset=box_model["initial_offset"],
        params=dict(
            tau_s=float(tau_s),
            zeta_p_deg=float(zeta_p_deg),
            T0_deg=float(T0_deg),
            d_rev_deg=float(d_rev_deg),
            eps=float(eps),
            phi=coeff.tolist(),
            windup_deg=float(windup_deg),
            ramp_deg=float(ramp_deg),
            ar1=None if ar1 is None else list(map(float, ar1)),
            initial_offset=box_model["initial_offset"],
            n_sub=n_sub,
            macro_moves=None
            if macro_moves is None
            else np.asarray(macro_moves).tolist(),
        ),
    )
