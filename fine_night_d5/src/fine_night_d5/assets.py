"""Pinned034 sky/horizon/IERS and matched scalar-beam asset construction.

All generation is opt-in and run-gated. The GSM operator is the reviewed
034/C51 float32 Galactic RING1024→64→8 TCMB/no-CMB operator, offline only.
"""

import hashlib
import importlib.metadata as metadata
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .beam import frequency_operator, interpolation_qc

GSM_URL = "https://apps.datacentral.org.au/pygdsm/data/gsm2016_components.h5"
GSM_SOURCES = {
    "gsm16.py": "0ffcd755ad2b5c1f52689ba9a4d1c7d5535278809c7d539f8a6ddba879a3e91e",
    "base_skymodel.py": (
        "0f8c21dc3666d470e719b4a3001ac4502aa9286a8259426b0be883567b0ffb8b"
    ),
    "component_data.py": (
        "c0fad9bc830ee0d4014c8d24851f3dcea381a687596b86acd3a34513f0d75aef"
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(pin):
    """Require a declared exact input checksum before loading its payload."""
    path = Path(pin["path"]).resolve(strict=True)
    if sha256(path) != pin["sha256"] or (
        "bytes" in pin and path.stat().st_size != pin["bytes"]
    ):
        raise ValueError("model asset checksum/size differs")
    return path


def calibration_stage(pin):
    """Verify a historical063 stage and its actual immutable producer bytes."""
    manifest = checked_file(pin)
    record = json.loads(manifest.read_text())
    directory = manifest.parent
    if (
        record["schema"] != 1
        or record["status"] != "COMPLETE"
        or record["label"] != "diagnostic"
        or sha256(directory / "arrays.npz") != record["arrays_sha256"]
        or record["producer_sources"].get("fine_night.py")
        != record["producer_source_sha256"]
    ):
        raise ValueError("historical063 stage payload/producer identity differs")
    for name, digest in record["producer_sources"].items():
        if Path(name).name != name or sha256(directory / "sources" / name) != digest:
            raise ValueError("historical063 producer snapshot differs")
    return directory / "arrays.npz", record


def atomic_npz(path, **arrays):
    """Install a fresh complete model stage; never overwrite/relabel."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("model stage already exists")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_json(path, obj):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scalar_beam(pin, frequencies_mhz):
    """Read a full explicitly labelled primary/control and direct cubic QC."""
    path = checked_file(pin)
    with np.load(path, allow_pickle=False) as z:
        f, maps, lmax = z["freqs"] / 1e6, z["bm"], int(z["lmax"])
        if "frequency_scope" in z and "subset" in str(z["frequency_scope"]):
            raise ValueError("bounded subset control is not a full beam authority")
        if pin["role"] in ("old_symmetric", "new_symmetric"):
            if str(z["field_symmetry"]) != "body-x":
                raise ValueError(
                    "primary beam lacks declared complex body-x projection"
                )
        elif pin["role"] not in ("old_raw_control", "new_raw_control"):
            raise ValueError("explicit primary/control beam role required")
    if lmax != 128 or maps.shape != (len(f), 130, 258):
        raise ValueError("model beam is not scalar lmax128 MWSS")
    import s2fft

    weights = np.asarray(
        s2fft.utils.quadrature_jax.quad_weights(L=129, sampling="mwss")
    )
    requested, qc = interpolation_qc(f, maps, frequencies_mhz, weights)
    np.testing.assert_allclose(
        frequency_operator(f, maps, f), maps, rtol=1e-13, atol=1e-13
    )
    return requested, {"role": pin["role"], "input_sha256": pin["sha256"]}, qc


@contextmanager
def offline_gsm_parent(parent, cache):
    """Supported private Astropy cache binding; original input read-only."""
    from astropy.config import set_temp_cache
    from astropy.utils.data import conf, download_file, import_file_to_cache

    source = checked_file(parent)
    cache = Path(cache).resolve()
    if source == cache or source.is_relative_to(cache):
        raise ValueError("GSM cache must differ from immutable parent input")
    cache.mkdir(parents=True, exist_ok=False)
    with (
        set_temp_cache(str(cache), delete=False),
        conf.set_temp("allow_internet", False),
    ):
        import_file_to_cache(GSM_URL, source, remove_original=False)
        actual = Path(download_file(GSM_URL, cache=True, sources=[])).resolve()
        if not actual.is_relative_to(cache) or sha256(actual) != parent["sha256"]:
            raise ValueError("offline GSM loader did not resolve the pinned parent")
        yield actual


def gsm_maps(config, frequencies_mhz, out):
    """Generate native-frequency maps plus one mandatory saved034 anchor.

    Each frequency's arrays and closure evidence are checkpointed before
    constructing any sky SHT or model. No source correction is applied.
    """
    import healpy as hp
    from pygdsm import GlobalSkyModel16

    if metadata.version("pygdsm") != "1.7.0":
        raise ValueError("require034's exact pygdsm1.7.0 loader")
    loader = Path(metadata.distribution("pygdsm").locate_file("pygdsm"))
    if any(sha256(loader / name) != expected for name, expected in GSM_SOURCES.items()):
        raise ValueError("audited GSM loader source differs")
    f = np.asarray(frequencies_mhz, float)
    if f.ndim != 1 or not np.isfinite(f).all() or np.any(np.diff(f) <= 0):
        raise ValueError("unique increasing native model frequency axis required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    saved_path = checked_file(config["saved034_maps"])
    with np.load(saved_path, allow_pickle=False) as z:
        saved_f = z["freqs_mhz"]
        saved64, saved8 = z["M64"], z["M8"]
    anchor = float(config["closure_frequency_mhz"])
    hit = np.flatnonzero(saved_f == anchor)
    if len(hit) != 1:
        raise ValueError("missing/ambiguous frozen034 GSM closure anchor")
    frequencies = np.unique(np.r_[f, anchor])
    maps, checks = {}, []
    with offline_gsm_parent(config["parent"], out / "astropy-cache") as parent:
        gsm = GlobalSkyModel16(
            freq_unit="MHz", data_unit="TCMB", resolution="hi", include_cmb=False
        )
        try:
            if Path(gsm.h5.filename).resolve() != parent:
                raise ValueError("GSM constructor opened a different parent")
            for frequency in frequencies:
                high = gsm.generate(float(frequency))
                if (
                    high.shape != (12 * 1024**2,)
                    or high.dtype != np.float32
                    or not np.isfinite(high).all()
                ):
                    raise ValueError("034 float32 Galactic RING1024 map required")
                m64 = hp.ud_grade(
                    high, 64, order_in="RING", order_out="RING", power=None
                )
                m8 = hp.ud_grade(m64, 8, order_in="RING", order_out="RING", power=None)
                check = {"frequency_mhz": float(frequency), "anchor_closure": None}
                if frequency == anchor:
                    j = int(hit[0])
                    exact = np.array_equal(
                        m64.astype(np.float64), saved64[j]
                    ) and np.array_equal(m8.astype(np.float64), saved8[j])
                    check["anchor_closure"] = {
                        "exact64and8": bool(exact),
                        "max64_K": float(np.max(abs(m64 - saved64[j]))),
                        "max8_K": float(np.max(abs(m8 - saved8[j]))),
                    }
                    write_json(out / f"closure_{frequency:g}.json", check)
                    if not exact:
                        raise ValueError("old034 GSM map operator byte closure failed")
                atomic_npz(
                    out / f"gsm_{frequency:g}.npz",
                    frequency_mhz=frequency,
                    M64=m64.astype(np.float64),
                    M8=m8.astype(np.float64),
                )
                maps[float(frequency)] = m64.astype(np.float64)
                checks.append(check)
                write_json(
                    out / "GSM_STAGE.json",
                    {
                        "input_sha256": config["parent"]["sha256"],
                        "completed": checks,
                        "source_correction": False,
                    },
                )
        finally:
            gsm.h5.close()
    return np.stack([maps[float(v)] for v in f]), checks


def install_iers(config):
    """Use exactly the original034 Earth-orientation tables, offline."""
    from astropy.utils import iers

    paths = {name: checked_file(pin) for name, pin in config.items()}
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    iers.IERS_B.open(
        file=str(paths["eopc04.1962-now"]), readme=str(paths["ReadMe.eopc04"])
    )
    table = iers.IERS_Auto.read(
        file=str(paths["finals2000A.all"]), readme=str(paths["ReadMe.finals2000A"])
    )
    iers.earth_orientation_table.set(table)
    return {name: pin["sha256"] for name, pin in config.items()}


def horizon_weight(pin):
    """Existing v0005 87.5m fractional horizon operator, without a new shift."""
    import eigsim

    path = checked_file(pin)
    az, altitude, info = eigsim.load_horizon_curve("87.5m", path=path)
    return np.asarray(eigsim.open_sky_weight(altitude, az, 128)), dict(info)
