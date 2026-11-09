"""Exact full-resolution drive operands; no angular approximation or fallback."""

import importlib
import json
from functools import partial
from pathlib import Path

import numpy as np

from . import assets
from .shared_frame import array_identity
from .streaming import orientation_groups


def angle_key(euler):
    """Preserve every F64 bit, including signed zero; never round angles."""
    a = np.asarray(euler)
    if a.shape != (3,) or a.dtype != np.dtype("float64") or not np.isfinite(a).all():
        raise ValueError("finite exact float64 Euler triple required")
    return np.ascontiguousarray(a).tobytes().hex()


def drive_eulers(pointing):
    """Use precisely the native streaming provider and ZYZ conversion."""
    from croissant.rotations import rotmat_to_eulerZYZ
    from eigsim.rotations import drive_rotation_matrix

    groups = orientation_groups(
        pointing["el_drive_deg"],
        pointing["az_drive_deg"],
        pointing["rotations"],
        pointing["misalignment"],
    )
    return np.asarray(
        [
            rotmat_to_eulerZYZ(
                drive_rotation_matrix(el, az, misalignment=pointing["misalignment"])
            )
            for el, az, _ in groups
        ],
        dtype=np.float64,
    )


def implementation():
    """Bind the functions generating and consuming the numerical operands."""
    import jax
    import jaxlib
    from s2fft.recursions import risbo_jax

    names = ["croissant.rotations", "s2fft", "eigsim.simulate", "eigsim.rotations"]
    return dict(
        modules={n: assets.sha256(importlib.import_module(n).__file__) for n in names},
        risbo_jax_sha256=assets.sha256(risbo_jax.__file__),
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        jaxlib_so={
            str(p): assets.sha256(p)
            for p in sorted(Path(jaxlib.__file__).parent.rglob("*.so"))
        },
        jax_enable_x64=bool(jax.config.jax_enable_x64),
    )


def build(eulers, context, out, L=129):
    """Compute original full D tables once per exact beta, with no cutoff."""
    import jax.numpy as jnp
    from croissant.rotations import generate_rotate_dls

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    eulers = np.asarray(eulers)
    keys = [angle_key(e) for e in eulers]
    if L != 129 or not len(keys) or len(set(keys)) != len(keys):
        raise ValueError("unique exact Euler triples and full L129 required")
    runtime = implementation()
    if not runtime["jax_enable_x64"]:
        raise ValueError("full float64 drive operands require x64")
    tables, entries = {}, {}
    for e, key in zip(eulers, keys, strict=True):
        beta = np.ascontiguousarray(e[1:2]).tobytes().hex()
        if beta not in tables:
            table = np.asarray(
                generate_rotate_dls(L, jnp.asarray(e[1], dtype=jnp.float64))
            )
            if (
                table.shape != (L, 2 * L - 1, 2 * L - 1)
                or table.dtype != np.dtype("float64")
                or not np.isfinite(table).all()
            ):
                raise ValueError("original full D generation returned invalid operands")
            path = out / f"dl_{len(tables):03}.npy"
            np.save(path, table, allow_pickle=False)
            tables[beta] = dict(
                path=str(path), sha256=assets.sha256(path), array=array_identity(table)
            )
        entries[key] = dict(euler=e.tolist(), beta_bits=beta)
    assets.write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic exact full-resolution drive operands",
            L=L,
            context=context,
            implementation=runtime,
            eulers=array_identity(eulers),
            entries=entries,
            tables=tables,
        ),
    )
    path = out / "RUN_DONE.json"
    return dict(path=str(path), sha256=assets.sha256(path))


class DriveOperands:
    """Strict context/hash checked read-only tables, dynamic JIT operands."""

    def __init__(self, pin, eulers, context):
        self.pin = pin
        record = json.loads(assets.checked_file(pin).read_text())
        if (
            record["L"] != 129
            or record["context"] != context
            or record["implementation"] != implementation()
            or record["eulers"] != array_identity(eulers)
        ):
            raise ValueError(
                "exact drive operand physical/runtime/coordinate context changed"
            )
        if set(record["entries"]) != {angle_key(e) for e in eulers}:
            raise ValueError("exact drive operand coverage changed")
        self.entries, self.tables = record["entries"], {}
        for beta, spec in record["tables"].items():
            table = np.load(
                assets.checked_file(spec), allow_pickle=False, mmap_mode="r"
            )
            if (
                table.shape != (129, 257, 257)
                or table.dtype != np.dtype("float64")
                or array_identity(table) != spec["array"]
            ):
                raise ValueError("full drive table shape/dtype/payload changed")
            self.tables[beta] = table
        for key, entry in self.entries.items():
            a = np.asarray(entry["euler"], dtype=np.float64)
            if (
                angle_key(a) != key
                or entry["beta_bits"] != np.ascontiguousarray(a[1:2]).tobytes().hex()
                or entry["beta_bits"] not in self.tables
            ):
                raise ValueError("drive Euler/table coordinate binding changed")

    def table(self, euler):
        key = angle_key(np.asarray(euler))
        if key not in self.entries:
            raise ValueError(
                "unexpected exact drive Euler coverage: "
                + repr(np.asarray(euler).tolist())
            )
        return self.tables[self.entries[key]["beta_bits"]]

    def orientation(self, beam_norm, sim_L):
        """One kernel per frame, same full rotation/horizon/SHT operations."""
        import croissant as cro
        import jax
        import jax.numpy as jnp
        import s2fft

        @jax.jit
        def kernel(beam_alm, euler, sim, horizon, quad_weights, dl):
            alm = jnp.asarray(beam_alm, dtype=complex)
            rotated = cro.rotations.rotate_alm_batched(alm, euler[None], dl[None])[0]
            flat = rotated.reshape((-1, 129, 257))
            inverse = partial(
                s2fft.inverse,
                L=129,
                spin=0,
                nside=None,
                sampling="mwss",
                method="jax",
                reality=True,
            )
            maps = jnp.real(jax.vmap(inverse)(flat))
            pixel = maps.reshape(alm.shape[:-2] + maps.shape[1:])
            masked = pixel * horizon[None]
            above = jnp.einsum("ft...,t->f", masked, quad_weights)
            fg = 1.0 - above / beam_norm
            forward = partial(
                s2fft.forward,
                L=129,
                spin=0,
                nside=None,
                sampling="mwss",
                method="jax",
                reality=True,
            )
            alm_topo = jax.vmap(forward)(masked)
            alm_trunc = cro.utils.reduce_lmax(alm_topo, sim_L - 1)
            return sim.topo_to_sim(alm_trunc), fg

        def orient(beam_alm, euler, sim, horizon, quad_weights):
            if np.shape(beam_alm)[-2:] != (129, 257):
                raise ValueError("full L129 beam required")
            return kernel(
                beam_alm, euler, sim, horizon, quad_weights, self.table(euler)
            )

        return orient
