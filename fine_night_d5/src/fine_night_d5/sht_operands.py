"""Guarded full-L129 transform operands from the pinned S2FFT library.

Raw north-pole recurrence singularities are checked against the diagnosed
support BEFORE the library's analytic pole assignments. Unknown nonfinite
coordinates stop; no angular truncation, interpolation or generic fill.
"""

import importlib
import importlib.metadata as metadata
import json
from functools import partial
from pathlib import Path

import numpy as np

from . import assets
from .drive_operands import implementation as drive_implementation
from .shared_frame import array_identity


def raw_support(raw, theta):
    """Only diagnosed valid north-pole slots may have undefined recursion."""
    raw, theta = np.asarray(raw), np.asarray(theta)
    L = raw.shape[-1]
    if raw.shape != (2 * L - 1, len(theta), L) or theta[0] != 0:
        raise ValueError("raw Wigner/grid axes differ")
    bad = ~np.isfinite(raw)
    valid = (
        abs(np.arange(2 * L - 1) - (L - 1))[:, None, None]
        <= np.arange(L)[None, None, :]
    )
    allowed = valid & (np.arange(len(theta))[None, :, None] == 0)
    unexpected = np.argwhere(bad & ~allowed)
    if len(unexpected):
        i, t, ell = map(int, unexpected[0])
        raise ValueError(
            f"unexpected raw Wigner nonfinite at theta[{t}]={theta[t]!r}, "
            f"ell={ell}, m={L - 1 - i}"
        )
    return bad, dict(
        nonfinite_total=int(bad.sum()),
        nan=int(np.isnan(raw).sum()),
        infinity=int(np.isinf(raw).sum()),
        valid_interior_nonfinite=0,
        invalid_triangle_nonfinite=0,
        nonfinite_theta_radians=0.0,
    )


def checked_poles(kernel, weights):
    """Full finite payload and exact spin0 analytic boundary convention."""
    kernel, weights = np.asarray(kernel), np.asarray(weights)
    L = kernel.shape[1]
    if kernel.shape != (len(weights), L, L) or kernel.dtype != np.dtype("float64"):
        raise ValueError("full float64 MWSS reality kernel axes differ")
    bad = np.argwhere(~np.isfinite(kernel))
    if len(bad):
        raise ValueError(
            "unexpected completed transform kernel nonfinite at theta/ell/m="
            + repr(bad[0].tolist())
        )
    norm = np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    for row, sign in [(0, np.ones(L)), (-1, (-1.0) ** np.arange(L))]:
        np.testing.assert_array_equal(kernel[row, :, 1:], 0)
        np.testing.assert_allclose(
            kernel[row, :, 0], sign * norm * weights[row], rtol=1e-14, atol=1e-16
        )


def implementation():
    names = [
        "s2fft.precompute_transforms.construct",
        "s2fft.precompute_transforms.spherical",
        "s2fft.recursions.price_mcewen",
        "s2fft.transforms.spherical",
        "s2fft.transforms.otf_recursions",
        "s2fft.utils.resampling_jax",
        "s2fft.utils.quadrature_jax",
        "s2fft.sampling.s2_samples",
    ]
    return dict(
        drive_runtime=drive_implementation(),
        numpy=np.__version__,
        numpy_path=np.__file__,
        numpy_module_sha256=assets.sha256(np.__file__),
        s2fft=metadata.version("s2fft"),
        consumer_sha256=assets.sha256(__file__),
        modules={
            n: dict(
                path=importlib.import_module(n).__file__,
                sha256=assets.sha256(importlib.import_module(n).__file__),
            )
            for n in names
        },
    )


def build(raw_diagnostic, context, out):
    """Recheck raw support, construct original-library kernels, verify poles."""
    import jax.numpy as jnp
    from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
    from s2fft.recursions.price_mcewen import compute_all_slices_jax
    from s2fft.sampling.s2_samples import thetas
    from s2fft.utils.quadrature_jax import quad_weights_transform

    wrapper = json.loads(assets.checked_file(raw_diagnostic).read_text())
    if not wrapper["source_runtime_equal_before_after"]:
        raise ValueError("closed raw diagnostic source/runtime proof required")
    diagnosis = json.loads(assets.checked_file(wrapper["conformance"]).read_text())
    runtime = implementation()
    if (
        diagnosis["L"] != 129
        or diagnosis["spin"] != 0
        or diagnosis["sampling"] != "mwss"
        or not diagnosis["no_cleaning"]
        or len(diagnosis["records"]) != 2
    ):
        raise ValueError("actual L129 spin0 MWSS raw diagnostic required")
    for name, spec in diagnosis["modules"].items():
        if runtime["modules"][name] != spec:
            raise ValueError("diagnosed recursion/constructor implementation changed")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    kernels, supports = {}, []
    for forward, old in zip([False, True], diagnosis["records"], strict=True):
        if old["forward"] is not forward:
            raise ValueError("raw diagnostic transform direction differs")
        theta = np.asarray(thetas(258 if forward else 129, "mwss"))
        raw = np.asarray(
            compute_all_slices_jax(jnp.asarray(theta), 129, 0, "mwss", forward, None)
        )
        bad, support = raw_support(raw, theta)
        with np.load(assets.checked_file(old["payload"]), allow_pickle=False) as z:
            np.testing.assert_array_equal(theta, z["thetas"])
            np.testing.assert_array_equal(bad, z["nonfinite_mask"])
        supports.append(dict(forward=forward, **support))
        assets.write_json(out / "RAW_SUPPORT_CHECKED.json", supports)
        kernel = np.asarray(
            spin_spherical_kernel_jax(
                129,
                spin=0,
                reality=True,
                sampling="mwss",
                nside=None,
                forward=forward,
                recursion="price-mcewen",
            )
        )
        weights = (
            np.asarray(quad_weights_transform(129, "mwss", None))
            if forward
            else np.ones(len(theta))
        )
        checked_poles(kernel, weights)
        name = "forward" if forward else "inverse"
        path = out / (name + ".npy")
        np.save(path, kernel, allow_pickle=False)
        kernels[name] = dict(
            path=str(path), sha256=assets.sha256(path), array=array_identity(kernel)
        )
    assets.write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic guarded fullL129 S2FFT transform operands",
            L=129,
            spin=0,
            sampling="mwss",
            reality=True,
            nside=None,
            iter=0,
            recursion="price-mcewen",
            context=context,
            implementation=runtime,
            raw_diagnostic=raw_diagnostic,
            raw_support=supports,
            kernels=kernels,
            pole_treatment=(
                "verified analytic spin0 north/south boundary values; "
                "no interior or invalid nonfinite allowed"
            ),
        ),
    )
    path = out / "RUN_DONE.json"
    return dict(path=str(path), sha256=assets.sha256(path))


class SHTOperands:
    """Read-only checked full transform kernels, passed dynamically to JIT."""

    def __init__(self, pin, context):
        self.pin = pin
        self.record = json.loads(assets.checked_file(pin).read_text())
        r = self.record
        if any(
            r[k] != v
            for k, v in dict(
                L=129,
                spin=0,
                sampling="mwss",
                reality=True,
                nside=None,
                iter=0,
                recursion="price-mcewen",
                context=context,
                implementation=implementation(),
            ).items()
        ):
            raise ValueError(
                "full transform operand runtime/context/convention changed"
            )
        self.kernels = {}
        for name, shape in [("inverse", (130, 129, 129)), ("forward", (259, 129, 129))]:
            spec = r["kernels"][name]
            a = np.load(assets.checked_file(spec), allow_pickle=False, mmap_mode="r")
            if a.shape != shape or array_identity(a) != spec["array"]:
                raise ValueError("full transform kernel axes/payload changed")
            self.kernels[name] = a

    def orientation(self, drive, beam_norm, sim_L):
        """Same full rotation/horizon/SHT/frame, existing precomputed API."""
        import croissant as cro
        import jax
        import jax.numpy as jnp
        from s2fft.precompute_transforms import spherical

        @jax.jit
        def kernel(
            beam_alm,
            euler,
            sim,
            horizon,
            quad_weights,
            dl,
            inverse_kernel,
            forward_kernel,
        ):
            alm = jnp.asarray(beam_alm, dtype=complex)
            rotated = cro.rotations.rotate_alm_batched(alm, euler[None], dl[None])[0]
            flat = rotated.reshape((-1, 129, 257))
            inverse = partial(
                spherical.inverse,
                L=129,
                spin=0,
                kernel=inverse_kernel,
                sampling="mwss",
                reality=True,
                method="jax",
                nside=None,
            )
            maps = jnp.real(jax.vmap(inverse)(flat))
            pixel = maps.reshape(alm.shape[:-2] + maps.shape[1:])
            masked = pixel * horizon[None]
            above = jnp.einsum("ft...,t->f", masked, quad_weights)
            fg = 1.0 - above / beam_norm
            forward = partial(
                spherical.forward,
                L=129,
                spin=0,
                kernel=forward_kernel,
                sampling="mwss",
                reality=True,
                method="jax",
                nside=None,
                iter=0,
            )
            alm_topo = jax.vmap(forward)(masked)
            return sim.topo_to_sim(cro.utils.reduce_lmax(alm_topo, sim_L - 1)), fg

        def orient(beam_alm, euler, sim, horizon, quad_weights):
            if np.shape(beam_alm)[-2:] != (129, 257):
                raise ValueError("full L129 beam required")
            return kernel(
                beam_alm,
                euler,
                sim,
                horizon,
                quad_weights,
                drive.table(euler),
                self.kernels["inverse"],
                self.kernels["forward"],
            )

        return orient
