"""C54 detached-producer entrypoint; dry-run first, explicit resource gate.

The probe runs one complete physical member on native AND fine grids and
saves independently completed subunits. It never scores the D5 target.
"""

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
from eigsep_analysis import ground_reflect as gr
from eigsep_analysis import night_fit
from eigsep_analysis import s11_diagnostics as sd
from eigsep_analysis import s11_hypotheses as obs
from eigsep_analysis import s11_hypothesis_io as intake
from eigsep_analysis import s11_hypothesis_pipeline as pipeline
from eigsep_analysis import s11_hypothesis_report as report
from eigsep_analysis.beammap import beam as beam_io
from scipy.interpolate import interp1d

from . import checkpoint as ck
from . import feature, leakage, terrain
from .config import DEFAULTS


def load_config(path):
    config = json.loads(Path(path).read_text())
    choices = copy.deepcopy(DEFAULTS)
    unknown = set(config.get("choices", {})) - set(choices)
    if unknown:
        raise ValueError(f"unknown scientific choices: {sorted(unknown)}")
    choices.update(config.get("choices", {}))
    config["choices"] = choices
    for name in ["code_root", "mock_root", "run_dir", "manifest"]:
        config[name] = str(Path(config[name]).resolve())
    return config


def required_sources(store):
    """Exact primary-input staging list; source Git pins stay in source closure."""
    tokens = [
        "/s11_data_deployment5/",
        "mist1_canyon_obs_",
        "20241024_BE1_FE1/",
        "/013_s11_calibration/calibrated/",
        "/013_s11_calibration/D_epoch_osl.csv",
        "/c43_followup/native_standards.h5",
        "/derived/dem/v0004/marjum_dem.npz",
        "/curation/horizon_profiles_v0005.npz",
        "/derived/height_eras/v0001/b0_features.csv",
        "/contrib_pointing_v6/pointing_table.parquet",
        "/034_night_fit/columns/columns.json",
        "/041_ground_reflectivity/echo_data.npz",
        "/_run/units/trace_",
        "/hfss_native_sep2026/bowtie_native_mwss_L180.npz",
        "/cal_materials/switch_sparams.npz",
        "/notebooks/S11_eigsep_bowtie_oct25.npz",
    ]
    return [s for s in store.entries if any(t in s for t in tokens)]


def verify_inputs(store, full_hash=False):
    rows = []
    for source in required_sources(store):
        path = store.resolve(source)
        exists = path.is_file()
        valid_size = exists and path.stat().st_size == store.entries[source]["bytes"]
        if full_hash and valid_size:
            store.get(source)
        rows.append(
            dict(
                source=source,
                path=str(path),
                exists=exists,
                valid_size=valid_size,
                bytes=store.entries[source]["bytes"],
                sha256=store.entries[source]["sha256"],
            )
        )
    return rows


def configuration_id(config):
    return f"r{config['radius_m']:g}_s{config['stride']}_b{config['bin_deg']:g}"


def physical_units(config, probe=False):
    settings = config.get(
        "terrain_configurations", [dict(radius_m=600.0, stride=4, bin_deg=1.0)]
    )
    epochs = ["night", "1310", "1917"]
    members = config["choices"]["terrain_members"]
    if probe:
        settings = [dict(radius_m=600.0, stride=4, bin_deg=1.0)]
        epochs, members = ["night"], 1
    return [
        dict(configuration=configuration_id(s), epoch=e, member=m, **s)
        for s in settings
        for e in epochs
        for m in range(members)
    ]


def gate(config, command, identity, manifest_sha):
    path = Path(config.get("resource_gate", "MISSING_RESOURCE_GATE"))
    if not path.is_file():
        raise ValueError(
            "root's measured resource gate required before producer launch"
        )
    approved = json.loads(path.read_text())
    if command not in approved["allow"]:
        raise ValueError(f"resource gate does not authorize {command}")
    if approved["source_fingerprint"] != ck.fingerprint(identity):
        raise ValueError("resource gate approved different source closure")
    if approved["manifest_sha256"] != manifest_sha:
        raise ValueError("resource gate approved different input manifest")
    if approved["configuration_fingerprint"] != ck.fingerprint(
        dict(
            choices=config["choices"], units=physical_units(config, command == "probe")
        )
    ):
        raise ValueError(
            "resource gate approved different physical/configuration scope"
        )
    return approved


def geometry(store, epoch):
    """Reproduce041's original mount, epoch mapping and exact DEM position."""
    with np.load(store.suffix("/curation/horizon_profiles_v0005.npz")) as h:
        era = {"night": "87.5m", "1310": "91m", "1917": "91m-reraise"}[epoch]
        ant = np.r_[h["antenna_enu_m"][:2], float(h[f"u_m_{era}"])]
    b0 = pd.read_csv(store.suffix("/derived/height_eras/v0001/b0_features.csv"))
    b0 = b0[(b0.n_terms == 4) & (b0.floor_kind == "F_plus") & b0.t.notna()]
    if epoch == "night":
        rows = b0[b0.is_reference.astype(bool)]
    else:
        rows = b0[
            b0.epoch_local == {"1310": "07-17 13:10", "1917": "07-17 19:17"}[epoch]
        ]
    pointing = pd.read_parquet(
        store.suffix("/contrib_pointing_v6/pointing_table.parquet"),
        columns=["t_utc_s", "az_deg", "el_deg", "tilt_deg"],
    )
    indices = [int(np.argmin(abs(pointing.t_utc_s.to_numpy() - t))) for t in rows.t]
    if not indices:
        raise ValueError("no physical epoch mapping")
    # This preserves the original041 convention for reproduction, not a new
    # estimator for scientific averaging of measured spectra.
    az = float(np.median([round(pointing.az_deg.iloc[i], 1) for i in indices]))
    el = float(np.median([round(pointing.el_deg.iloc[i], 1) for i in indices]))
    mount = json.loads(store.suffix("/034_night_fit/columns/columns.json").read_text())[
        "runs"
    ]["nominal"]["offsets"]
    rot = night_fit.mount_rotation_tilt(
        az + mount["az0_deg"], el + mount["el0_deg"], mount["tau_deg"], mount["psi_deg"]
    )[0]
    return (
        ant,
        rot,
        dict(
            era=era, az_deg=az, el_deg=el, offsets=mount, input_epochs_s=rows.t.tolist()
        ),
    )


def _unit_key(identity, manifest_sha, choices, params):
    return ck.fingerprint(
        dict(
            source=identity, input_manifest=manifest_sha, choices=choices, params=params
        )
    )


def export_physical(config, store, identity, manifest_sha, probe=False):
    """Execute complete native/fine subunits; return only verified exports."""
    root = Path(config["run_dir"])
    choices = config["choices"]
    f, _, _ = intake.load_calibrated(
        store.suffix("/013_s11_calibration/calibrated/ant_calibrated.h5")
    )
    with np.load(store.suffix("/derived/dem/v0004/marjum_dem.npz")) as z:
        dem = sd.RasterDem(z["dem"], float(z["res"]), int(z["e0_px"]), int(z["n0_px"]))
    beam = gr.native_beam(
        *beam_io.load_native(
            store.suffix("/hfss_native_sep2026/bowtie_native_mwss_L180.npz")
        )[:2]
    )
    exports = []
    for p in physical_units(config, probe):
        cid = p["configuration"]
        ant, rotation, geom_meta = geometry(store, p["epoch"])
        cell_name = f"cells_{cid}_{p['epoch']}"
        cellunit = ck.Unit(
            root,
            cell_name,
            _unit_key(identity, manifest_sha, choices, dict(p, member="geometry")),
        )
        if not cellunit.completed():
            margin = min(
                ant[0] - dem.extent_e[0],
                dem.extent_e[1] - dem.res - ant[0],
                ant[1] - dem.extent_n[0],
                dem.extent_n[1] - dem.res - ant[1],
            )
            if p["radius_m"] >= margin:
                raise ValueError("requested facet disc exceeds valid DEM domain")
            cells = gr.facet_cells(dem, ant, p["radius_m"], p["stride"])
            ck.write_h5(cellunit.path / "cells.h5", cells)
            cellunit.finish(
                dict(parameters=p, geometry=geom_meta, n_visible=len(cells["E"]))
            )
        with h5py.File(cellunit.path / "cells.h5", "r") as h:
            cells = {k: h[k][:] for k in h}
        rng = np.random.default_rng([choices["seed"], p["member"]])
        offset = rng.normal(0, choices["position_sigma_m"], 2)
        ant_member = ant + np.r_[offset, 0.0]
        sigma = float(np.exp(rng.uniform(*np.log(choices["roughness_range_m"]))))
        max_delay = (
            2 * np.max(gr.member_geometry(cells, ant_member)["range_m"]) / sd.C_LIGHT
            + 45e-9
        )
        native, fine = terrain.frequency_grids(f, choices["fine_step_hz"], max_delay)
        for grid, freq in [("native", native), ("fine", fine)]:
            name = f"facet_{cid}_{p['epoch']}_m{p['member']:03d}_{grid}"
            params = dict(
                p,
                grid=grid,
                frequency_sha256=ck.fingerprint(freq.tolist()),
                cell_result_sha256=intake.digest(cellunit.path / "cells.h5"),
            )
            unit = ck.Unit(
                root, name, _unit_key(identity, manifest_sha, choices, params)
            )
            if not unit.completed():
                result = terrain.facet_spectra(
                    freq,
                    cells,
                    ant_member,
                    beam,
                    rotation,
                    choices,
                    p["radius_m"],
                    sigma,
                    p["member"],
                    p["bin_deg"],
                    covariance=(grid == "native"),
                )
                result.update(
                    antenna_enu_m=ant_member,
                    rotation_body_true_enu=rotation,
                    position_offset_m=offset,
                    original_frequency_hz=f,
                    original_measured_mask=np.ones(len(f), bool),
                )
                ck.write_h5(unit.path / "spectra.h5", result)
                unit.finish(
                    dict(
                        parameters=params,
                        geometry=geom_meta,
                        rng="numpy PCG64",
                        seed=[choices["seed"], p["member"]],
                        input_closure=store.consumed,
                        label="diagnostic",
                        diffuse_scope="power/covariance; uncoupled grids",
                        transfer="free scale; unknown network phase",
                    )
                )
            if grid == "native":
                exports.append(dict(p, path=str(unit.path / "spectra.h5")))
            size = sum(x.stat().st_size for x in root.rglob("*") if x.is_file())
            if probe and size > choices["probe"]["product_max_bytes"]:
                raise RuntimeError(
                    "probe product budget exceeded; completed subunits retained"
                )
    return exports


def export_r0(config, store, identity, manifest_sha):
    """Use existing long first-hit traces; raw and normalized exports separate."""
    f, _, _ = intake.load_calibrated(
        store.suffix("/013_s11_calibration/calibrated/ant_calibrated.h5")
    )
    f = f[f >= 50e6]
    for epoch in ["night", "1310", "1917"]:
        ant, _, meta = geometry(store, epoch)
        for nside in config["choices"]["r0_nsides"]:
            source = store.suffix(f"/trace_{meta['era']}_ns{nside}.npz")
            params = dict(epoch=epoch, nside=nside, trace_sha=intake.digest(source))
            unit = ck.Unit(
                config["run_dir"],
                f"r0_{epoch}_ns{nside}",
                _unit_key(identity, manifest_sha, config["choices"], params),
            )
            if not unit.completed():
                with np.load(source) as z:
                    result = terrain.r0_echo(
                        f, z["points"], ant, nside, config["choices"]["cell_chunk"]
                    )
                result["frequency_hz"] = f
                with np.load(
                    store.suffix("/notebooks/S11_eigsep_bowtie_oct25.npz")
                ) as source_s11:
                    mag = interp1d(
                        source_s11["freqs"] * 1e6,
                        10 ** (source_s11["S11"] / 20),
                        kind="cubic",
                        fill_value=0,
                        bounds_error=False,
                    )(f)
                material = gr.eps_complex(
                    f,
                    config["choices"]["material_eps"],
                    config["choices"]["material_sigma_s_m"],
                )
                amplitude = (1 - mag**2) * abs(gr.r_normal(material)) * sd.C_LIGHT / f
                result["paper_gamma_historical"] = amplitude * result["raw"]
                result["paper_gamma_reference64"] = amplitude * result["reference64"]
                ck.write_h5(unit.path / "r0.h5", result)
                unit.finish(
                    dict(
                        parameters=params,
                        geometry=meta,
                        scope=(
                            "hybrid first-hit sum; historical raw retained; "
                            "common omega64 normalization"
                        ),
                    )
                )


def _load_groups(store):
    groups, excluded = intake.load_raw_groups(store)
    intake.add_calibrated_groups(store, groups)
    return groups, excluded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["intake", "controls", "probe", "export", "score", "report"]
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--members", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if (
        args.members is not None
        and args.members != config["choices"]["probe"]["members"]
    ):
        raise ValueError("--members cannot silently change frozen choices")
    store = intake.InputStore(config["manifest"], config.get("path_map"))
    manifest_sha = intake.digest(config["manifest"])
    identity = ck.source_identity(config["mock_root"], config["code_root"])
    units = physical_units(config, args.command == "probe")
    inputs = verify_inputs(store, full_hash=False)
    dry = dict(
        source_fingerprint=ck.fingerprint(identity),
        manifest_sha256=manifest_sha,
        configuration_fingerprint=ck.fingerprint(
            dict(choices=config["choices"], units=units)
        ),
        required_inputs=inputs,
        missing=[r for r in inputs if not r["valid_size"]],
        physical_units=units,
        native_working_channels=803,
        fine_channels=int(200e6 / config["choices"]["fine_step_hz"]) + 1,
        phase_cell_frequency_products_per_visible_cell=len(units)
        * (803 + int(200e6 / config["choices"]["fine_step_hz"]) + 1),
        provisional_probe=config["choices"]["probe"],
        no_measured_cost_yet=True,
    )
    if args.dry_run or args.command == "intake":
        print(json.dumps(dry, indent=2))
        return 0 if not dry["missing"] else 2
    if args.command == "report":
        report.render_completed(config["run_dir"])
        return 0
    if dry["missing"]:
        raise ValueError("stage all declared required primary inputs before execution")
    ck.require_committed(config["mock_root"], config["code_root"], identity)
    approved = gate(config, args.command, identity, manifest_sha)
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    record = dict(
        schema="c54-run-v1",
        source_identity=identity,
        source_fingerprint=ck.fingerprint(identity),
        input_manifest_sha256=manifest_sha,
        configuration=config,
        invocation=[
            sys.executable,
            "-m",
            "s11_hypotheses.run",
            args.command,
            "--config",
            str(Path(args.config).resolve()),
        ],
        host=platform.node(),
        resource_gate=approved,
        interpreter=sys.executable,
        repo_commits={
            name: subprocess.check_output(
                ["git", "-C", config[name + "_root"], "rev-parse", "HEAD"], text=True
            ).strip()
            for name in ["code", "mock"]
        },
    )
    # Different commands may share frozen controls but have separate records.
    ck.write_json(root / f"RUN_RECORD_{args.command}.json", record)
    if args.command == "probe":
        export_physical(config, store, identity, manifest_sha, probe=True)
        # Bounded actual raw-input control baseline, never final D5 target.
        raw = next(p for p in store.entries if p.endswith("ants11_20260717_190839Z.h5"))
        r = intake.read_d5_raw(store.get(raw))
        sel = dict(
            support_s=0.15e-6,
            ridge_fraction=1e-6,
            cutoff=1e-9,
            target_hz=[46e6, 48e6],
            delays_s=[],
        )
        b = obs.baseline(r["data"]["ant"], r["frequency_hz"], sel, [10e6, 12e6])
        receiver_source = next(
            p for p in store.entries if p.endswith("recs11_20260717_190839Z.h5")
        )
        receiver = intake.read_d5_raw(store.get(receiver_source))
        mistic_source = sorted(
            p for p in store.entries if "mist1_canyon_obs_" in p and p.endswith(".gz")
        )[0]
        mistic = intake.read_mistic_vna(store.get(mistic_source))
        with h5py.File(store.suffix("/c43_followup/native_standards.h5"), "r") as h:
            truth = h["capture_042/values"][:]
        osl = np.array([r["data"]["cal:" + v] for v in ["VNAO", "VNAS", "VNAL"]])
        chain = intake.osl_and_chain(r["data"]["ant"], osl, truth)
        unit = ck.Unit(
            root,
            "probe_raw_control",
            _unit_key(identity, manifest_sha, config["choices"], sel),
        )
        if not unit.completed():
            arrays = dict(
                frequency_hz=r["frequency_hz"],
                values=r["data"]["ant"],
                prediction=b.prediction,
                receiver_values=receiver["data"],
                mistic_frequency_hz=mistic[10]["values"][0],
                mistic_source_short=mistic[12]["values"],
                target_blind_mask=~obs.interval_mask(r["frequency_hz"], [46e6, 48e6]),
            )
            arrays.pop("receiver_values")
            arrays.update(
                {"receiver/" + key: value for key, value in receiver["data"].items()}
            )
            arrays.update({"chain/" + key: value for key, value in chain.items()})
            ck.write_h5(unit.path / "control.h5", arrays)
            unit.finish(dict(scope="10–12MHz raw control/chain; no D5 target score"))
        return 0
    if args.command == "controls":
        pipeline.historical_positive_control(store, ck.write_json, root)
        groups, excluded = _load_groups(store)
        lock, tuning = pipeline.freeze_design(
            store,
            groups,
            config["choices"],
            feature,
            ck.fingerprint(identity),
            manifest_sha,
        )
        lock["lock_sha256"] = ck.fingerprint(lock)
        ck.write_json(root / "design.lock.json", lock)
        ck.write_json(root / "tuning.json", tuning)
        ck.write_json(root / "excluded.json", excluded)
        return 0
    lock_path = Path(config.get("design_lock", root / "design.lock.json"))
    lock = json.loads(lock_path.read_text())
    check = dict(lock)
    digest = check.pop("lock_sha256")
    if digest != ck.fingerprint(check) or lock["source_fingerprint"] != ck.fingerprint(
        identity
    ):
        raise ValueError("design lock source/content mismatch")
    if lock["input_fingerprint"] != manifest_sha:
        raise ValueError("design lock input mismatch")
    if args.command == "export":
        export_r0(config, store, identity, manifest_sha)
        export_physical(config, store, identity, manifest_sha)
        return 0
    if args.command == "score":
        # Verify physical units BEFORE expensive repeated feature/scoring work.
        physical = []
        for p in physical_units(config):
            name = f"facet_{p['configuration']}_{p['epoch']}_m{p['member']:03d}_native"
            marker = root / "units" / name / "DONE.json"
            if not marker.exists():
                raise ValueError(f"incomplete physical export: {name}")
            row = json.loads(marker.read_text())
            unit = ck.Unit(root, name, row["key"])
            unit.completed()
            physical.append(dict(p, path=str(unit.path / "spectra.h5")))
        ck.write_json(
            root / "physical_convergence.json",
            terrain.convergence_report(physical, config["choices"]),
        )
        science = ck.Unit(
            root,
            "science",
            _unit_key(
                identity,
                manifest_sha,
                config["choices"],
                dict(lock=digest, physical=physical),
            ),
        )
        if science.completed():
            return 0
        groups, excluded = _load_groups(store)
        pipeline.feature_report(
            groups, lock, feature, ck.write_h5, ck.write_json, science.path / "feature"
        )
        pipeline.chain_report(
            store, lock, feature, ck.write_h5, ck.write_json, science.path / "chain"
        )
        pipeline.empirical_skirt(
            groups, lock, leakage, ck.write_h5, ck.write_json, science.path / "skirt"
        )
        pipeline.terrain_power_report(
            store,
            physical,
            config["choices"],
            ck.write_h5,
            ck.write_json,
            science.path / "terrain",
            r0_root=root,
        )
        science.finish(
            dict(
                label="diagnostic",
                consumed=store.consumed,
                excluded_captures=excluded,
                design_lock_sha256=intake.digest(lock_path),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
