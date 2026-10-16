"""Build private south extensions and checkpoint a bounded coverage comparison.

The source DEM frame and interior are unchanged. GeoTIFF tags, finite samples,
source hashes and overlap are checked before emitting each separately pinned
mosaic. This diagnostic compares specified south extensions at nominal +/-3 mm ENU,
all three heights, full 0.1-degree bearings; it is not the full matrix.
"""

import argparse
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import atomic_json, file_sha256
from .reference import BilinearDEM, trace_horizon

URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects/"
    "UT_WestEast_B22/UT_WestEast_7_B22/TIFF/"
)
PREFIX = "USGS_OPR_UT_WestEast_B22_12STJ"
BASE_HASH = "ea36c823b37ada86cd13e7119edea16d2469026b1cbbc37c5b85abfa199f0891"


def read_tile(path, east, north):
    with Image.open(path) as im:
        if im.size != (2000, 2000):
            raise ValueError("unexpected tile dimensions")
        if tuple(im.tag_v2[33550]) != (0.5, 0.5, 0.0):
            raise ValueError("unexpected pixel scale")
        expected = (
            0.0,
            0.0,
            0.0,
            200000.0 + east * 1000,
            4300000.0 + (north + 1) * 1000,
            0.0,
        )
        if tuple(im.tag_v2[33922]) != expected:
            raise ValueError("unexpected tile georeferencing")
        keys = tuple(im.tag_v2[34735])
        entries = {keys[i]: keys[i + 1 : i + 4] for i in range(4, len(keys), 4)}
        if entries.get(3072) != (0, 1, 6341) or entries.get(1025) != (0, 1, 1):
            raise ValueError("unexpected CRS or raster pixel convention")
        a = np.array(im, dtype=np.float32)[::-1]
        nodata = im.tag_v2.get(42113)
        if not np.isfinite(a).all() or (
            nodata is not None and np.any(a == float(nodata))
        ):
            raise ValueError("nonfinite or nodata tile")
        return a


def build(base, out, levels=(2, 4), reuse_tiles_from=None):
    levels = tuple(levels)
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(type(x) is not int or not 1 <= x <= 20 for x in levels)
    ):
        raise ValueError("extensions must be unique integer kilometres in [1,20]")
    maximum = max(levels)
    cached = {}
    if reuse_tiles_from is not None:
        cached = json.loads((reuse_tiles_from / "COVERAGE_INPUTS.json").read_text())[
            "input_tiles"
        ]
    if file_sha256(base) != BASE_HASH:
        raise ValueError("base DEM differs from frozen v0003")
    out.mkdir(parents=True, exist_ok=False)
    tiles = out / "tiles"
    tiles.mkdir()
    with np.load(base, allow_pickle=False) as d:
        saved = {k: d[k] for k in d.files}
    if saved["dem"].shape != (10000, 8000) or float(saved["res"]) != 0.5:
        raise ValueError("unexpected base geometry")
    if not np.array_equal(saved["raster_origin"], [291000.25, 4345000.25]):
        raise ValueError("unexpected fixed coordinate origin")
    if int(saved["e0_px"]) != 2000 or int(saved["n0_px"]) != 2000:
        raise ValueError("unexpected base pixel offset")
    records = {}
    # Preserve the old samples literally; append south strips in their physical
    # UTM location. No resampling, vertical correction or new survey offset.
    extension = np.empty((maximum * 2000, 8000), dtype=np.float32)
    for north in range(44 - maximum, 44):
        for east in range(90, 94):
            name = f"{PREFIX}{east}{north}.tif"
            path = tiles / name
            if name in cached:
                source = reuse_tiles_from / "tiles" / name
                if (
                    cached[name]["url"] != URL + name
                    or file_sha256(source) != cached[name]["sha256"]
                ):
                    raise ValueError("cached tile identity changed")
                shutil.copyfile(source, path)
            else:
                with (
                    urllib.request.urlopen(URL + name, timeout=90) as response,
                    path.open("wb") as f,
                ):
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
            a = read_tile(path, east, north)
            extension[
                (north - 44 + maximum) * 2000 : (north - 43 + maximum) * 2000,
                (east - 90) * 2000 : (east - 89) * 2000,
            ] = a
            records[name] = dict(
                url=URL + name, sha256=file_sha256(path), bytes=path.stat().st_size
            )
            print("tile", name, flush=True)
    products = {}
    for km in levels:
        result = dict(saved)
        result["dem"] = np.concatenate(
            [extension[(maximum - km) * 2000 :], saved["dem"]], axis=0
        )
        result["n0_px"] = np.asarray(int(saved["n0_px"]) + km * 2000)
        if not np.array_equal(result["dem"][km * 2000 :], saved["dem"]):
            raise ValueError("interior changed")
        names = [
            [str(tiles / f"{PREFIX}{e}{n}.tif") for n in range(44 - km, 44)]
            for e in range(90, 94)
        ]
        result["files"] = np.concatenate([np.asarray(names), saved["files"]], axis=1)
        path = out / f"dem-south-{km}km.npz"
        np.savez(path, **result)
        products[str(km)] = dict(
            path=str(path),
            sha256=file_sha256(path),
            shape=list(result["dem"].shape),
            n0_px=int(result["n0_px"]),
            overlap_identical=True,
        )
        del result
    manifest = dict(
        label="diagnostic private coverage candidates",
        base_path=str(base),
        base_sha256=BASE_HASH,
        unchanged_frame=True,
        input_tiles=records,
        products=products,
        builder_sha256=file_sha256(Path(__file__)),
    )
    atomic_json(out / "COVERAGE_INPUTS.json", manifest)
    print(json.dumps(products), flush=True)


def trace_one(task):
    dem_path, output, position, offsets, km, label = task
    start = time.monotonic()
    dem = BilinearDEM.from_npz(dem_path)
    bearings = np.arange(3600) * 0.1
    trace = trace_horizon(
        dem,
        position,
        bearings,
        height_offsets_m=offsets,
        method="continuous",
        inner_cutoff_m=0.25,
        earth_radius_m=6371000.0,
    )
    arrays = dict(
        asdict(trace), bearings_deg=bearings, position_enu_m=np.asarray(position)
    )
    path = Path(output) / f"trace-{km}km-{label}.npz"
    np.savez(path, **arrays)
    record = dict(
        km=km,
        label=label,
        position_enu_m=position,
        seconds=time.monotonic() - start,
        dem_sha256=file_sha256(Path(dem_path)),
        data_sha256=file_sha256(path),
        edge_counts=trace.edge_limited.sum(axis=1).tolist(),
    )
    atomic_json(path.with_suffix(".json"), record)
    return record


def probe(out, inventory_path):
    manifest = json.loads((out / "COVERAGE_INPUTS.json").read_text())
    inv = json.loads(inventory_path.read_text())
    for item in manifest["products"].values():
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("candidate DEM changed")
    nominal = np.array(inv["nominal_enu_m"])
    offsets = inv["trace_identity"]["height_offsets_m"]
    positions = [("nominal", nominal.tolist())]
    for axis in range(3):
        for sign in (-1, 1):
            p = nominal.copy()
            p[axis] += sign * 0.003
            positions.append((f"{'ENU'[axis]}{sign:+d}", p.tolist()))
    tasks = [
        (manifest["products"][str(km)]["path"], str(out), p, offsets, km, label)
        for label, p in positions
        for km in sorted(map(int, manifest["products"]))
    ]
    if len(os.sched_getaffinity(0)) > 2:
        raise ValueError("coverage probe requires at most two CPUs")
    start = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trace_one, tasks):
            records.append(result)
            print(json.dumps(result), flush=True)
    atomic_json(
        out / "COVERAGE_PROBE.json",
        dict(
            records=records,
            seconds=time.monotonic() - start,
            inventory_sha256=file_sha256(inventory_path),
            source_sha256=file_sha256(Path(__file__)),
            scope=(
                "nominal +/-3mm ENU, full 0.1deg bearings, "
                "all three heights; coverage only"
            ),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["build", "probe"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", type=Path)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--extensions-km", nargs="+", type=int, default=[2, 4])
    p.add_argument("--reuse-tiles-from", type=Path)
    a = p.parse_args()
    if a.mode == "build":
        build(a.base, a.output, a.extensions_km, a.reuse_tiles_from)
    else:
        probe(a.output, a.inventory)


if __name__ == "__main__":
    main()
