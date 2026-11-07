"""C51's staged, offline map-only producer (no SHT, beam or night-data fit).

Run as ``uv run --no-project --python .venv/bin/python python -m
compact_source_d5.run --inputs SNAPSHOT --out NEW_RUN --stage reproduce``.
The account stage additionally requires a verified reproduction product.
"""

import argparse
import datetime
import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .identity import (
    PACKAGE,
    finalize,
    producer_identity,
    read_product,
    sha256,
    snapshot_producer,
    verify_inputs,
    write_json,
)
from .math import (
    METHODS,
    account_patch,
    area_sr,
    catalogue_flux,
    degrade_sparse,
    neighbour_mask,
    nominal_profile,
    patch,
    sample_sparse,
)
from .plots import comparison, encircled

URL = "https://apps.datacentral.org.au/pygdsm/data/gsm2016_components.h5"
PROBE_MHZ = (109.375,)
ACCOUNT_MHZ = (70.3125, 109.375, 148.4375)
SOURCES = ("Cyg A", "Vir A", "Cas A", "Tau A", "Her A")


def coordinates(record):
    """Pinned ICRS/J2000 anchor -> Galactic unit vector; no topocentric rotation."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if record["frame"] != "ICRS" or record["catalogue_epoch_label"] != "ep=J2000":
        raise ValueError("expected cited ICRS/J2000 patch anchor")
    coord = SkyCoord(
        ra=record["ra_deg"] * u.deg, dec=record["dec_deg"] * u.deg, frame="icrs"
    ).galactic
    return coord.cartesian.xyz.value, dict(
        l_deg=float(coord.l.deg), b_deg=float(coord.b.deg), frame="Galactic"
    )


@contextmanager
def offline_parent(parent, cache, expected_sha):
    """Use supported Astropy APIs, private cache, internet disabled; no monkeypatch."""
    from astropy.config import set_temp_cache
    from astropy.utils.data import conf, download_file, import_file_to_cache

    parent, cache = Path(parent).resolve(), Path(cache).resolve()
    if parent.is_relative_to(cache) or parent == cache:
        raise ValueError("private cache must differ from immutable parent input")
    cache.mkdir(parents=True, exist_ok=False)
    with (
        set_temp_cache(str(cache), delete=False),
        conf.set_temp("allow_internet", False),
    ):
        import_file_to_cache(URL, parent, remove_original=False)
        cached = Path(download_file(URL, cache=True, sources=[])).resolve()
        if not cached.is_relative_to(cache) or sha256(cached) != expected_sha:
            raise ValueError("offline parent cache identity mismatch")
        yield cached


def verify_runtime(identity, pins):
    for name, version in pins["runtime_versions"].items():
        if identity["packages"].get(name) != version:
            raise ValueError(f"{name}: expected historical map version {version}")
    from importlib import metadata

    root = Path(metadata.distribution("pygdsm").locate_file("pygdsm"))
    sources = {name: sha256(root / name) for name in pins["loader_sources"]}
    if sources != pins["loader_sources"]:
        raise ValueError("installed pygdsm sources differ from the audited loader")
    return dict(path=str(root), sha256=sources)


def exact_degrade(high):
    """034's original float32 intermediate, before float64 archive promotion."""
    import healpy as hp

    if high.shape != (12 * 1024**2,) or high.dtype != np.dtype("float32"):
        raise ValueError("expected unrotated Galactic/RING NSIDE1024 float32 map")
    if not np.all(np.isfinite(high)):
        raise ValueError("nonfinite generated map")
    m64 = hp.ud_grade(high, 64, order_in="RING", order_out="RING", power=None)
    m8 = hp.ud_grade(m64, 8, order_in="RING", order_out="RING", power=None)
    return {1024: high, 64: m64, 8: m8}


def closure(maps, saved64, saved8, frequency_mhz):
    result = dict(
        frequency_mhz=frequency_mhz,
        intermediate_dtypes={str(n): str(m.dtype) for n, m in maps.items()},
    )
    for n, reference in [(64, saved64), (8, saved8)]:
        generated = maps[n].astype(np.float64)
        if generated.shape != reference.shape or not np.all(np.isfinite(reference)):
            raise ValueError("stored map shape/finite check failed")
        delta = generated - reference
        result[str(n)] = dict(
            exact_equal=bool(np.array_equal(generated, reference)),
            max_abs_k=float(np.max(np.abs(delta))),
            rms_k=float(np.sqrt(np.mean(delta**2))),
        )
    result["pass"] = all(result[str(n)]["exact_equal"] for n in (64, 8))
    return result


def injection_check(q, geometry, frequency, nside, method, excluded):
    """Noiseless closure of this pixel/mask operator, not real background validation."""
    from .math import kcmb_sr_per_jy

    x, y = geometry["x_deg"] / 10, geometry["y_deg"] / 10
    background = np.full(q.shape, 200.0)
    if method != METHODS[0]:
        background += 3 * x - 4 * y
    if method == METHODS[2]:
        background += 2 * x * x + 0.7 * x * y - 1.2 * y * y
    injected_jy = 10000.0
    data = background + injected_jy * kcmb_sr_per_jy(frequency * 1e6) * q
    result, _ = account_patch(data, q, geometry, frequency, nside, method, excluded)
    status = dict(
        injected_flux_jy=injected_jy,
        status=result["status"],
        fractional_error=None,
        pass_one_percent=False,
        scope="synthetic functional closure; no real background validation",
    )
    if result["status"] == "conditional":
        error = abs(result["formal_flux_jy"] / injected_jy - 1)
        status.update(fractional_error=error, pass_one_percent=bool(error <= 0.01))
        if error > 0.01:
            raise ValueError("noiseless compact injection failed1% recovery")
    else:
        status["reason"] = result["reason"]
    return status


def account_source(maps, centre, others, source, catalogue, frequency, out):
    """One fixed conditional profile through the three native map resolutions."""
    hi_pixel, hi_q = nominal_profile(1024, centre)
    q64 = degrade_sparse(hi_pixel, hi_q, 1024, 64)
    q8 = degrade_sparse(np.arange(len(q64)), q64, 64, 8)
    profile_integrals = {
        1024: float(hi_q.sum() * area_sr(1024)),
        64: float(q64.sum() * area_sr(64)),
        8: float(q8.sum() * area_sr(8)),
    }
    if any(abs(v - 1) > 0.01 for v in profile_integrals.values()):
        raise ValueError("profile flux conservation failed1% acceptance")
    try:
        flux = catalogue_flux(catalogue, frequency)
        catalogue_status = "published domain; epoch/scale caveats retained"
    except ValueError as error:
        flux, catalogue_status = None, str(error)
    radii = np.linspace(0, 5, 101)
    geom_hi = patch(1024, centre, 5)
    enc = np.array(
        [hi_q[geom_hi["radius_deg"] <= r].sum() * area_sr(1024) for r in radii]
    )
    name = source.replace(" ", "")
    summary = dict(
        source=source,
        frequency_mhz=frequency,
        catalogue_flux_jy=flux,
        catalogue_status=catalogue_status,
        catalogue_record=catalogue,
        profile_integrals=profile_integrals,
        profile="conditional nominal56arcmin Gaussian,5deg support",
        source_response_bounded=False,
        physical_compatible_interval_jy=None,
        phase2_source_replacement_allowed=False,
        resolutions={},
    )
    for nside, model_map in maps.items():
        geometry = patch(nside, centre)
        q = (
            sample_sparse(hi_pixel, hi_q, geometry["pixel"])
            if nside == 1024
            else (q64 if nside == 64 else q8)[geometry["pixel"]]
        )
        data = np.asarray(model_map[geometry["pixel"]], float)
        excluded = neighbour_mask(geometry["vector"], others)
        arrays = dict(
            geometry,
            profile_sr_inv=q,
            neighbour_excluded=excluded,
            data_k=data,
            radii_deg=radii,
            encircled_profile_fraction=enc,
        )
        methods = {}
        for method in METHODS:
            result, stages = account_patch(
                data, q, geometry, frequency, nside, method, excluded
            )
            result["delta_cat_minus_formal_jy"] = (
                None
                if flux is None or result["formal_flux_jy"] is None
                else flux - result["formal_flux_jy"]
            )
            result["synthetic_operator_check"] = injection_check(
                q, geometry, frequency, nside, method, excluded
            )
            methods[method] = result
            arrays.update({method + "__" + key: value for key, value in stages.items()})
            # Three-method primary diagnostic; other sources retain one raw check.
            if (
                nside == 1024
                and result["status"] == "conditional"
                and (source == "Cyg A" or method == METHODS[0])
            ):
                comparison(
                    geometry,
                    data,
                    stages["model_k"],
                    out / f"{name}_{frequency:g}_n{nside}_{method}",
                    f"{source}, {frequency:g} MHz: {method}",
                    "Exact034 GSM2016 reconstruction; Galactic/RING NSIDE1024, "
                    "K_CMB. Model is a fixed nominal56arcmin Gaussian plus the "
                    f"prescribed {method} background. Negative formal flux is "
                    "retained. Residual scatter is not radiometer noise or a "
                    "source uncertainty. No source replacement is adopted.",
                )
        formal = [
            v["formal_flux_jy"]
            for v in methods.values()
            if v["formal_flux_jy"] is not None
        ]
        summary["resolutions"][str(nside)] = dict(
            methods=methods,
            conditional_formal_span_jy=[min(formal), max(formal)] if formal else None,
            span_is_confidence_interval=False,
            patch_pixel_count=len(geometry["pixel"]),
            neighbour_excluded_count=int(excluded.sum()),
        )
        np.savez_compressed(out / f"{name}_{frequency:g}_n{nside}.npz", **arrays)
    return summary, radii, enc


def run(inputs_root, out, stage, reproduction=None):
    start = time.monotonic()
    inputs = verify_inputs(inputs_root)
    pins = json.loads((PACKAGE / "frozen_inputs.json").read_text())
    identity = producer_identity()
    if identity["git"]["dirty"]:
        raise ValueError("commit the producer before running scientific inputs")
    identity["pygdsm_loader"] = verify_runtime(identity, pins)
    if stage == "account":
        if reproduction is None:
            raise ValueError("account requires verified --reproduction product")
        old, proof = read_product(reproduction)
        if (
            proof.get("stage") != "reproduce"
            or not proof.get("map_closure_pass")
            or old["producer"]["source_sha256"] != identity["source_sha256"]
            or old["producer"]["packages"] != identity["packages"]
            or {k: v["sha256"] for k, v in old["inputs"].items()}
            != {k: v["sha256"] for k, v in inputs.items()}
        ):
            raise ValueError(
                "reproduction proof does not match this producer/input/runtime"
            )
    elif stage != "reproduce":
        raise ValueError("unknown phase-one stage")
    out = Path(out).resolve()
    if out.is_relative_to(Path("/mnt/data02/eigsep")):
        raise ValueError("shared tree is read-only")
    out.mkdir(parents=True, exist_ok=False)
    snapshot_producer(out, identity)
    results = dict(
        schema=1,
        label="diagnostic",
        stage=stage,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        convention=pins["map_convention"],
        map_closure=[],
        accounting=[],
        native_data_used=False,
        source_response_bounded=False,
        phase2_source_replacement_allowed=False,
        limitations=[
            "No source-specific GSM PSF or credible retained-flux "
            "interval is established by a nominal Gaussian.",
            "No independent-pixel/frequency source error model.",
            "No native data, beam, direct-column or night score.",
        ],
    )
    write_json(
        out / "RUN_STARTED.json", dict(producer=identity, inputs=inputs, stage=stage)
    )
    try:
        from pygdsm import GlobalSkyModel16

        catalogues = json.loads(Path(inputs["catalogue"]["path"]).read_text())[
            "pb_table_sources"
        ]
        anchors = json.loads(Path(inputs["positions"]["path"]).read_text())["sources"]
        anchor_by_name = {a["source"]: a for a in anchors}
        if len(anchors) != len(SOURCES) or set(anchor_by_name) != set(SOURCES):
            raise ValueError("missing/duplicate source coordinate identity")
        centres = {s: coordinates(anchor_by_name[s]) for s in SOURCES}
        results["source_anchors"] = {
            s: dict(anchor_by_name[s], galactic=centres[s][1]) for s in SOURCES
        }
        freqs = PROBE_MHZ if stage == "reproduce" else ACCOUNT_MHZ
        with np.load(inputs["saved_maps"]["path"], allow_pickle=False) as saved:
            saved_freqs, stored64, stored8 = (
                saved["freqs_mhz"],
                saved["M64"],
                saved["M8"],
            )
        if (
            saved_freqs.shape != (43,)
            or stored64.shape != (43, 49152)
            or stored8.shape != (43, 768)
            or np.any(np.diff(saved_freqs) <= 0)
        ):
            raise ValueError("saved034 map axes are not the frozen43-frequency product")
        with offline_parent(
            inputs["parent"]["path"], out / "astropy-cache", inputs["parent"]["sha256"]
        ) as cached:
            gsm = GlobalSkyModel16(
                freq_unit="MHz", data_unit="TCMB", resolution="hi", include_cmb=False
            )
            try:
                if Path(gsm.h5.filename).resolve() != cached:
                    raise ValueError("loader opened another GSM parent")
                for frequency in freqs:
                    match = np.flatnonzero(saved_freqs == frequency)
                    if len(match) != 1:
                        raise ValueError("missing/ambiguous saved frequency coordinate")
                    j = int(match[0])
                    maps = exact_degrade(gsm.generate(float(frequency)))
                    check = closure(maps, stored64[j], stored8[j], frequency)
                    results["map_closure"].append(check)
                    np.savez_compressed(
                        out / f"closure_{frequency:g}.npz",
                        generated64=maps[64],
                        saved64=stored64[j],
                        generated8=maps[8],
                        saved8=stored8[j],
                        frequency_mhz=frequency,
                    )
                    geometry = patch(64, centres["Cyg A"][0])
                    pixels = geometry["pixel"]
                    comparison(
                        geometry,
                        stored64[j][pixels],
                        maps[64][pixels],
                        out / f"closure_{frequency:g}",
                        f"034 map reproduction: {frequency:g} MHz, NSIDE64",
                        "Saved034 map / regenerated map / difference on the same "
                        "Galactic/RING pixels around CygA. Exact full-sky64 and8 "
                        "closure is checked numerically, not inferred from plot.",
                        model_label="Regenerated GSM",
                        data_label="Saved034 GSM",
                    )
                    if not check["pass"]:
                        raise ValueError(
                            "034 map reproduction differs: stop before accounting"
                        )
                    if stage == "account":
                        fractions = {}
                        for source in SOURCES:
                            summary, radii, enc = account_source(
                                maps,
                                centres[source][0],
                                [centres[s][0] for s in SOURCES if s != source],
                                source,
                                catalogues[source],
                                frequency,
                                out,
                            )
                            results["accounting"].append(summary)
                            fractions[source] = enc
                        encircled(
                            radii,
                            fractions,
                            out / f"profiles_{frequency:g}",
                            f"Conditional profile support: {frequency:g} MHz",
                        )
                    # Do not retain high-resolution maps from earlier frequencies.
                    del maps
            finally:
                gsm.h5.close()
        results.update(
            map_closure_pass=True,
            outcome="reproduction passed"
            if stage == "reproduce"
            else "conditional background accounting; "
            "unique retained-flux/correction gate unresolved",
            elapsed_s=time.monotonic() - start,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        finalize(out, identity, inputs, results)
    except BaseException as error:
        write_json(
            out / "RUN_FAILED.json",
            dict(
                error=repr(error),
                stage=stage,
                elapsed_s=time.monotonic() - start,
                partial_results=results,
            ),
        )
        raise
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["reproduce", "account"], required=True)
    parser.add_argument("--reproduction", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.out, args.stage, args.reproduction)


if __name__ == "__main__":
    main()
