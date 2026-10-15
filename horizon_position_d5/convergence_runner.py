"""Checkpointed reference-convergence execution and coordinate-rich exports."""

import fcntl
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cache import BlockCache, atomic_json, file_sha256, fingerprint
from .convergence import evaluate_stencil
from .reference import BilinearDEM, trace_horizon

_DEM = None


def _initialize_trace_worker(dem_path):
    global _DEM
    _DEM = BilinearDEM.from_npz(dem_path)


def _trace_task(task):
    """Workers compute arrays; only the parent commits checkpoints."""
    start = time.monotonic()
    params, settings = task
    bearings = (
        np.arange(params["bearing_count"], dtype=float) * params["bearing_spacing_deg"]
    )
    if fingerprint(bearings.tolist()) != params["bearings_hash"]:
        raise ValueError("bearing identity mismatch")
    result = asdict(
        trace_horizon(
            _DEM,
            params["position_enu_m"],
            bearings,
            height_offsets_m=settings["height_offsets_m"],
            method=settings["method"],
            inner_cutoff_m=settings["inner_cutoff_m"],
            earth_radius_m=settings["earth_radius_m"],
        )
    )
    result.update(
        bearings_deg=bearings, position_enu_m=np.asarray(params["position_enu_m"])
    )
    return result, {
        "seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _validate_trace(arrays, task, nheights):
    params = task["parameters"]
    shape = (nheights, params["bearing_count"])
    for name in (
        "alpha_rad",
        "distance_m",
        "surface_height_m",
        "curvature_height_m",
        "gradient_en",
        "edge_range_m",
        "edge_limited",
    ):
        expected = (
            shape + (2,)
            if name == "gradient_en"
            else (shape[1],)
            if name == "edge_range_m"
            else shape
        )
        value = np.asarray(arrays[name])
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"invalid trace {name}")
    if arrays["edge_limited"].dtype.kind != "b":
        raise ValueError("trace edge flags must be boolean")
    if np.any(abs(arrays["alpha_rad"]) > np.pi / 2):
        raise ValueError("invalid horizon elevation")
    if fingerprint(arrays["bearings_deg"].tolist()) != params[
        "bearings_hash"
    ] or not np.array_equal(arrays["position_enu_m"], params["position_enu_m"]):
        raise ValueError("trace coordinates differ from task")


class TemperatureEvaluator:
    """One compiled bearing/mask operator at a time, all heights sequentially."""

    def __init__(self, response, lmax):
        offset, kernel = (
            np.asarray(response["offset_k"]),
            np.asarray(response["kernel_k"]),
        )
        if (
            offset.ndim != 2
            or kernel.ndim != 4
            or kernel.shape[:2] != offset.shape
            or offset.dtype.kind != "f"
            or kernel.dtype.kind != "f"
            or not np.isfinite(offset).all()
            or not np.isfinite(kernel).all()
        ):
            raise ValueError("invalid imported response arrays")
        self.response = response
        self.lmax = lmax
        self.key = self.model = None

    def __call__(self, trace, subsamples):
        import jax

        from .pilot import temperature_model

        key = (fingerprint(trace["bearings_deg"].tolist()), subsamples)
        if key != self.key:
            self.model = None
            jax.clear_caches()
            self.model = temperature_model(
                self.response["offset_k"],
                self.response["kernel_k"],
                trace["bearings_deg"],
                self.lmax,
                sub=subsamples,
            )
            self.key = key
        return np.stack([np.asarray(self.model(alpha)) for alpha in trace["alpha_rad"]])


def _flatten_arrays(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(_flatten_arrays(value, name + "__"))
        else:
            result[name] = np.asarray(value)
    return result


def _outcome_counts(values):
    names, counts = np.unique(values, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts, strict=True)}


class ConvergenceRunner:
    """Single-writer run; probe and full run share one immutable identity."""

    def __init__(self, output, inventory, runtime, *, workers, dem_path, evaluator):
        self.output = Path(output)
        self.inventory, self.runtime = inventory, runtime
        self.workers, self.dem_path, self.evaluator = workers, Path(dem_path), evaluator
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if inventory["source"]["source_dirty"]:
            raise ValueError("execution requires clean source")
        if any(
            runtime[k] != inventory["source"][k]
            for k in ("source_commit", "source_tree", "lock_sha256")
        ):
            raise ValueError("runtime source differs from planned inventory")
        config = {
            "workers": workers,
            "plan_hash": inventory["plan_hash"],
            "inventory_hash": inventory["inventory_hash"],
        }
        self.identity = {
            k: runtime[k]
            for k in ("source_commit", "source_tree", "lock_sha256", "versions")
        }
        self.identity.update(input_hashes=runtime["input_hashes"], config=config)
        self.record = {
            "schema_version": 1,
            "label": "diagnostic",
            "phase": "reference-convergence",
            "identity": self.identity,
            "identity_hash": fingerprint(self.identity),
            "runtime": runtime,
        }
        self.trace_tasks = {x["id"]: x for x in inventory["traces"]}
        self.eval_tasks = {x["id"]: x for x in inventory["evaluations"]}
        coords = inventory["coordinates"]
        self.temp_shape = (
            len(coords["heights_m"]),
            len(coords["times_jd"]),
            len(coords["frequencies_mhz"]),
        )
        self.timings, self.files = [], {}

    def _entry(self, folder, name):
        data, meta = (
            self.output / folder / (name + ".npz"),
            self.output / folder / (name + ".json"),
        )
        return {
            "data": str(data.relative_to(self.output)),
            "metadata": str(meta.relative_to(self.output)),
            "data_sha256": file_sha256(data),
            "metadata_sha256": file_sha256(meta),
        }

    def _progress(self, phase, completed, planned):
        atomic_json(
            self.output / "PROGRESS.json",
            {
                "phase": phase,
                "complete": False,
                "completed": completed,
                "planned": planned,
                "identity_hash": self.record["identity_hash"],
            },
        )

    def _traces(self, tasks, trace_function=None):
        cache = BlockCache(self.output / "traces", self.identity)
        pending = []
        for task in tasks:
            arrays = cache.load(task["id"], task["parameters"])
            if arrays is None:
                pending.append(task)
            else:
                _validate_trace(arrays, task, self.temp_shape[0])
                self.timings.append({"kind": "trace", "id": task["id"], "reused": True})
        pool = None
        try:
            payloads = [
                (task["parameters"], self.inventory["trace_identity"])
                for task in pending
            ]
            if trace_function is not None:
                results = map(trace_function, payloads)
            elif self.workers == 1:
                if pending:
                    _initialize_trace_worker(self.dem_path)
                results = map(_trace_task, payloads)
            else:
                pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_trace_worker,
                    initargs=(self.dem_path,),
                )

                # At most one wave of full arrays awaits the parent; avoid storing
                # all completed horizons in an unbounded future queue.
                def waves():
                    for start in range(0, len(payloads), self.workers):
                        yield from pool.map(
                            _trace_task, payloads[start : start + self.workers]
                        )

                results = waves()
            completed = len(tasks) - len(pending)
            for task, (arrays, timing) in zip(pending, results, strict=True):
                _validate_trace(arrays, task, self.temp_shape[0])
                cache.write(task["id"], task["parameters"], arrays)
                self.timings.append(
                    {"kind": "trace", "id": task["id"], "reused": False, **timing}
                )
                completed += 1
                self._progress("traces", completed, len(tasks))
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        return cache

    def _evaluations(self, tasks, traces):
        cache = BlockCache(self.output / "blocks", self.identity)
        ordered = sorted(
            tasks,
            key=lambda t: (
                self.trace_tasks[t["parameters"]["trace_id"]]["parameters"][
                    "bearing_spacing_deg"
                ],
                t["parameters"]["mask_subsamples"],
                t["id"],
            ),
        )
        for number, task in enumerate(ordered, 1):
            trace_id = task["parameters"]["trace_id"]
            trace_task = self.trace_tasks[trace_id]
            trace = traces.load(trace_id, trace_task["parameters"])
            if trace is None:
                raise ValueError("missing or corrupt trace during evaluation")
            _validate_trace(trace, trace_task, self.temp_shape[0])
            params = {
                **task["parameters"],
                "trace_data_sha256": file_sha256(
                    self.output / "traces" / (trace_id + ".npz")
                ),
            }
            arrays = cache.load(task["id"], params)
            reused = arrays is not None
            start = time.monotonic()
            if arrays is None:
                temperature = np.asarray(
                    self.evaluator(trace, params["mask_subsamples"])
                )
                arrays = {"temperature_k": temperature}
            value = arrays["temperature_k"]
            if (
                value.shape != self.temp_shape
                or value.dtype.kind != "f"
                or not np.isfinite(value).all()
            ):
                raise ValueError("temperature shape, dtype or finiteness invalid")
            if not reused:
                cache.write(task["id"], params, arrays)
            self.timings.append(
                {
                    "kind": "evaluation",
                    "id": task["id"],
                    "seconds": time.monotonic() - start,
                    "reused": reused,
                }
            )
            self._progress("evaluations", number, len(tasks))
        return cache

    def _export(self, name, arrays):
        cache = BlockCache(self.output / "blocks", self.identity)
        cache.write(name, {"result": name}, arrays)
        self.files[name] = self._entry("blocks", name)

    def _assemble(self, traces, evaluations):
        plan = self.inventory["plan"]
        bearings, masks, steps = (
            plan[k] for k in ("bearing_spacing_deg", "mask_subsamples", "steps_m")
        )
        cells = (len(self.inventory["center_offsets_m"]), *self.temp_shape)
        center = np.full((len(bearings), len(masks), *cells), np.nan)
        shifted = np.full((len(bearings), len(masks), len(steps), 3, 2, *cells), np.nan)
        cp, sp = np.zeros(center.shape, bool), np.zeros(shifted.shape, bool)
        edge = np.zeros(cells, bool)
        for item in self.inventory["stencil"]:
            b, m, p = (
                bearings.index(item["bearing_spacing_deg"]),
                masks.index(item["mask_subsamples"]),
                item["position_id"],
            )
            trace_task = self.trace_tasks[item["trace_id"]]
            trace = traces.load(item["trace_id"], trace_task["parameters"])
            if trace is None:
                raise ValueError("trace missing during assembly")
            eval_task = self.eval_tasks[item["evaluation_id"]]
            params = {
                **eval_task["parameters"],
                "trace_data_sha256": file_sha256(
                    self.output / "traces" / (item["trace_id"] + ".npz")
                ),
            }
            evaluated = evaluations.load(item["evaluation_id"], params)
            if evaluated is None:
                raise ValueError("evaluation missing during assembly")
            if item["step_m"] is None:
                index, values, present = (b, m, p), center, cp
            else:
                index = (
                    b,
                    m,
                    steps.index(item["step_m"]),
                    ("E", "N", "U").index(item["axis"]),
                    (-1, 1).index(item["sign"]),
                    p,
                )
                values, present = shifted, sp
            if present[index].any():
                raise ValueError("duplicate stencil destination")
            values[index], present[index] = evaluated["temperature_k"], True
            if b >= len(bearings) - 3:
                edge[p] |= trace["edge_limited"].any(axis=1)[:, None, None]
        if not cp.all() or not sp.all():
            raise ValueError("incomplete stencil cannot complete a run")
        self._export(
            "stencil_temperatures",
            {
                "center_k": center,
                "displaced_k": shifted,
                "center_present": cp,
                "displaced_present": sp,
                "edge_limited": edge,
            },
        )
        from .convergence import Criteria

        coords = self.inventory["coordinates"]
        cell_coords = {
            "position_id": np.arange(cells[0]),
            "height_m": coords["heights_m"],
            "sample_id": np.arange(cells[2]),
            "frequency_mhz": coords["frequencies_mhz"],
        }
        verdict = evaluate_stencil(
            center,
            shifted,
            bearing_spacing_deg=bearings,
            mask_subsamples=masks,
            steps_m=steps,
            axes=["E", "N", "U"],
            signs=[-1, 1],
            cell_coordinates=cell_coords,
            center_present=cp,
            displaced_present=sp,
            edge_limited=edge,
            criteria=Criteria(**plan["criteria"]),
        )
        self._export("convergence", _flatten_arrays(verdict))
        return verdict

    def run(self, *, probe=False, trace_function=None):
        """Commit RESULTS only after every required block has been verified.

        ``trace_function`` is an injection point for manufactured unit controls,
        absent from the CLI. Production workers always use the independent tracer.
        """
        start = time.monotonic()
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "RUN.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another runner owns this output") from exc
            path = self.output / "RUN_RECORD.json"
            if path.exists():
                import json

                previous = json.loads(path.read_text())
                if previous["identity"] != self.identity or previous.get(
                    "identity_hash"
                ) != fingerprint(self.identity):
                    raise ValueError("run identity changed; use another output")
            if not path.exists() and any(
                list((self.output / folder).glob("*.json"))
                or list((self.output / folder).glob("*.npz"))
                for folder in ("blocks", "traces")
            ):
                raise ValueError("unowned checkpoints without a run record")
            if (self.output / "RESULTS.json").exists():
                raise ValueError("run already complete; preserve immutable results")
            atomic_json(path, self.record)
            atomic_json(self.output / "INVENTORY.json", self.inventory)
            self.timings, self.files = [], {}
            traces = list(self.trace_tasks.values())
            evaluations = list(self.eval_tasks.values())
            if probe:
                nominal = self.inventory["nominal_enu_m"]
                traces = [
                    t for t in traces if t["parameters"]["position_enu_m"] == nominal
                ]
                ids = {t["id"] for t in traces}
                evaluations = [
                    t for t in evaluations if t["parameters"]["trace_id"] in ids
                ]
            try:
                trace_cache = self._traces(traces, trace_function)
                eval_cache = self._evaluations(evaluations, trace_cache)
                report = {
                    "label": "diagnostic",
                    "probe": probe,
                    "full_run_complete": False,
                    "timings": self.timings,
                    "planned_counts": self.inventory["counts"],
                    "executed_scope": {
                        "traces": len(traces),
                        "evaluations": len(evaluations),
                    },
                    "peak_parent_rss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "runtime": self.runtime,
                }
                if not probe:
                    verdict = self._assemble(trace_cache, eval_cache)
                    coords = {
                        k: np.asarray(v)
                        for k, v in self.inventory["coordinates"].items()
                    }
                    coords.update(
                        center_offsets_m=np.asarray(self.inventory["center_offsets_m"]),
                        bearing_spacing_deg=np.asarray(
                            self.inventory["plan"]["bearing_spacing_deg"]
                        ),
                        mask_subsamples=np.asarray(
                            self.inventory["plan"]["mask_subsamples"]
                        ),
                        steps_m=np.asarray(self.inventory["plan"]["steps_m"]),
                        axes=np.array(["E", "N", "U"]),
                        signs=np.array([-1, 1]),
                    )
                    self._export("coordinates", coords)
                    for folder, tasks in (("traces", traces), ("blocks", evaluations)):
                        for task in tasks:
                            self.files[task["id"]] = self._entry(folder, task["id"])
                    report["full_run_complete"] = True
                    report["outcomes"] = {
                        k: _outcome_counts(verdict[k])
                        for k in (
                            "forward_status",
                            "one_sided_status",
                            "common_slope_status",
                            "behavior",
                        )
                    }
                report["wall_seconds"] = time.monotonic() - start
                name = "PROBE_REPORT.json" if probe else "convergence_report.json"
                if probe:
                    # Keep repeated probe evidence rather than replacing its timings.
                    name = "probe-" + str(time.time_ns()) + ".json"
                atomic_json(self.output / name, report)
                if probe:
                    atomic_json(
                        self.output / "PROBE_REPORT.json",
                        {
                            "report": name,
                            "sha256": file_sha256(self.output / name),
                            "identity_hash": self.record["identity_hash"],
                            "full_run_complete": False,
                        },
                    )
                else:
                    atomic_json(
                        self.output / "RESULTS.json",
                        {
                            **self.record,
                            "complete": True,
                            "files": self.files,
                            "reports": {
                                name: file_sha256(self.output / name),
                                "INVENTORY.json": file_sha256(
                                    self.output / "INVENTORY.json"
                                ),
                            },
                        },
                    )
                atomic_json(
                    self.output / "PROGRESS.json",
                    {
                        "phase": "probe" if probe else "reference-convergence",
                        "complete": not probe,
                        "identity_hash": self.record["identity_hash"],
                    },
                )
                return report
            except BaseException as exc:
                atomic_json(
                    self.output / ("failure-" + str(time.time_ns()) + ".json"),
                    {
                        "identity_hash": self.record["identity_hash"],
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "timings": self.timings,
                        "wall_seconds": time.monotonic() - start,
                    },
                )
                raise
