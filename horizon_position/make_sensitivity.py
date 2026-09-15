"""dT_ant/d(position) and dT_ant/d(misalignment) for the D5 forward model.

Position (E, N, U): ``jax.jvp`` pushes the total horizon tangents
``dalpha_d{E,N,U}`` from ``horizons_position.npz`` through
``eigsim.open_sky_weight`` and ``eigsim.simulate`` at the nominal horizon,
one call per axis. ``t_sys`` is affine in the open-sky weight, so this is
exact up to the linearization of ``alpha_h`` itself. The E and N tangents
include the azimuthal parallax term (see make_horizons.py).

Misalignment (eps_y, eps_z): central differences at +/-0.25 deg of
``eigsim.rotations.misalignment_matrix``, four simulations. The Euler-angle
conversion inside ``simulate`` is NumPy and gimbal-locked at zero tilt, so
it cannot be differentiated (task-3 spike); at 0.25 deg it takes its
regular branch and represents the rotation exactly.

Cross-check: the jvp is compared with central differences of the
19-position re-run (``position_sims.npz``) at +/-0.1 m, asserting agreement
to 5 per cent RMS over LST and frequency; +/-1 m is printed as a
diagnostic. The step is the displacement ``calc_horizon`` actually saw:
E and N exact, U rounded to float32 (``calc_horizon`` subtracts it from the
float32 DEM). The output is written before the assertion, so a failed
check can be investigated from the stored agreement numbers; the run
still exits non-zero.

The receiver temperature ``simulate`` adds is constant and cancels in
every derivative.

Usage (from the monorepo root, after run_sims.py):
    uv run python horizon_position/make_sensitivity.py
Smoke test (grids differ from position_sims.npz, so no cross-check):
    uv run python horizon_position/make_sensitivity.py \
        --n-times 4 --freq-stride 40 --output-tag _smoke
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np

import eigsim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sims import OUTPUT_DIR, load_inputs  # noqa: E402

# Position axis -> (prefix in the position names, column of enu).
AXES = {"E": ("x", 0), "N": ("y", 1), "U": ("z", 2)}
# Misalignment parameter -> misalignment_matrix keyword.
TILTS = {"eps_y": "tilt_y_deg", "eps_z": "tilt_z_deg"}
EPS_DEG = 0.25
RMS_TOL = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-times",
        type=int,
        default=1436,  # run_sims.py's default; the cross-check needs the same grid
        help="time samples over one sidereal day",
    )
    p.add_argument(
        "--freq-stride",
        type=int,
        default=1,
        help="use every Nth config frequency (smoke tests only)",
    )
    p.add_argument(
        "--output-tag",
        default="",
        help="suffix for the output filename (required when grids differ)",
    )
    return p.parse_args()


def zenith_t_sys(inp, W, misalignment=None):
    """(n_times, n_freqs) zenith t_sys for open-sky weight ``W``."""
    return eigsim.simulate(
        inp.beam_data,
        inp.freqs_mhz,
        inp.sky,
        inp.times_jd,
        [0.0],
        [0.0],
        beam_kw={"horizon": W},
        sky_alm=inp.sky_alm,
        misalignment=misalignment,
    )[0]


def position_derivatives(inp, hz, i_nom):
    """jvp of t_sys along each total horizon tangent [K/m].

    Also returns the primal, which is the nominal t_sys.
    """

    def t_sys_of_alpha(alpha):
        return zenith_t_sys(inp, eigsim.open_sky_weight(alpha, inp.az_grid, inp.lmax))

    out = {}
    for axis in AXES:
        t0 = time.time()
        t_nom, dT = jax.jvp(
            t_sys_of_alpha, (inp.alpha_h[i_nom],), (hz[f"dalpha_d{axis}"],)
        )
        out[f"dT_d{axis}"] = np.asarray(dT)
        print(f"  jvp {axis}: {time.time() - t0:.0f}s")
    return out, np.asarray(t_nom)


def misalignment_derivatives(inp, W_nom):
    """Central differences of t_sys in each tilt at +/-EPS_DEG [K/deg]."""
    out = {}
    for param, kw in TILTS.items():
        t_pm = []
        for sign in (+1, -1):
            t0 = time.time()
            R = eigsim.rotations.misalignment_matrix(**{kw: sign * EPS_DEG})
            t_pm.append(np.asarray(zenith_t_sys(inp, W_nom, misalignment=R)))
            print(f"  {param} = {sign * EPS_DEG:+.2f} deg: {time.time() - t0:.0f}s")
        out[f"dT_d{param}"] = (t_pm[0] - t_pm[1]) / (2 * EPS_DEG)
    return out


def effective_delta(enu_p, enu_m, col):
    """Displacement calc_horizon saw between two positions [m].

    Only U meets float32 data (the DEM), so it is rounded to float32 before
    differencing; E and N are exact. See test_jacobian._calc_horizon_delta.
    """
    if col == 2:
        return float(np.float32(enu_p[col])) - float(np.float32(enu_m[col]))
    return float(enu_p[col]) - float(enu_m[col])


def finite_differences(sims, step):
    """Central differences of the re-run t_sys at one step ('0p1', '1').

    Returns ({'fd_dT_dE': ..., ...} in K/m, (3,) effective deltas in m).
    """
    names = [str(n) for n in sims["names"]]
    fd, deltas = {}, []
    for axis, (prefix, col) in AXES.items():
        i_p = names.index(f"{prefix}_p_{step}")
        i_m = names.index(f"{prefix}_m_{step}")
        delta = effective_delta(sims["enu"][i_p], sims["enu"][i_m], col)
        fd[f"fd_dT_d{axis}"] = (sims["t_sys"][i_p] - sims["t_sys"][i_m]) / delta
        deltas.append(delta)
    return fd, np.array(deltas)


def rms_rel_err(approx, ref):
    return float(np.sqrt(np.mean((approx - ref) ** 2) / np.mean(ref**2)))


def main():
    args = parse_args()
    wall0 = time.time()
    inp = load_inputs(args.n_times, freq_stride=args.freq_stride)
    i_nom = inp.names.index("nominal")

    hz = np.load(OUTPUT_DIR / "horizons_position.npz", allow_pickle=True)
    assert str(hz["pos_sha"]) == inp.pos_sha

    # Decide on the cross-check before the expensive part, so a grid
    # mismatch cannot silently produce an unvalidated position_sensitivity.
    sims = np.load(OUTPUT_DIR / "position_sims.npz", allow_pickle=True)
    same_grid = np.array_equal(sims["freqs_mhz"], inp.freqs_mhz) and np.array_equal(
        sims["times_jd"], inp.times_jd
    )
    if same_grid:
        assert str(sims["pos_sha"]) == inp.pos_sha
    else:
        print(
            "NOTE: freqs_mhz/times_jd differ from position_sims.npz "
            "(smoke mode) -- skipping the finite-difference cross-check."
        )
        if not args.output_tag:
            raise SystemExit("grids differ: pass --output-tag for a smoke run")

    print("Position derivatives (jax.jvp)...")
    out, t_nom = position_derivatives(inp, hz, i_nom)
    print("Misalignment derivatives (central differences)...")
    W_nom = eigsim.open_sky_weight(inp.alpha_h[i_nom], inp.az_grid, inp.lmax)
    out.update(misalignment_derivatives(inp, W_nom))

    meta = dict(
        freqs_mhz=inp.freqs_mhz,
        times_jd=inp.times_jd,
        pos_sha=inp.pos_sha,
        eigsim_version=eigsim.__version__,
        misalignment_step_deg=EPS_DEG,
        axes=np.array(list(AXES)),
    )
    agreement = {}
    if same_grid:
        # The jvp primal is the nominal t_sys; it must reproduce the re-run.
        t_nom_sims = sims["t_sys"][inp.names.index("nominal")]
        primal_err = float(np.max(np.abs(t_nom - t_nom_sims) / t_nom_sims))
        print(f"jvp primal vs position_sims nominal: max rel diff {primal_err:.2e}")
        assert primal_err < 1e-9, "jvp primal does not reproduce position_sims"

        for step in ("0p1", "1"):
            fd, deltas = finite_differences(sims, step)
            errs = np.array(
                [rms_rel_err(out[f"dT_d{a}"], fd[f"fd_dT_d{a}"]) for a in AXES]
            )
            agreement[step] = errs
            meta[f"fd_delta_{step}_m"] = deltas
            meta[f"rms_rel_err_{step}"] = errs
            print(f"jvp vs central difference at +/-{step.replace('p', '.')} m:")
            for a, d, e in zip(AXES, deltas, errs):
                print(f"  {a}: delta = {d:.10f} m  RMS rel err = {100 * e:.3f}%")
            if step == "0p1":
                out.update(fd)

    out_file = OUTPUT_DIR / f"position_sensitivity{args.output_tag}.npz"
    np.savez_compressed(out_file, **out, **meta)
    print(f"Saved {out_file}  ({time.time() - wall0:.0f}s total)")

    if same_grid:
        worst = float(agreement["0p1"].max())
        assert worst < RMS_TOL, (
            f"jvp vs +/-0.1 m central difference: worst RMS rel err "
            f"{100 * worst:.2f}% exceeds {100 * RMS_TOL:.0f}%"
        )
        print(f"Cross-check passed: worst axis {100 * worst:.3f}% < 5%")


if __name__ == "__main__":
    main()
