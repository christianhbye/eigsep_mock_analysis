"""Rewrite only the metadata keys of an existing Zeus21 ensemble npz.

Every data array (``freqs_MHz``, ``T21_mK``, ``T21_native_mK``, ``z_native``,
``xHI``, ``z_xHI``, ``params``, ``kept_index``, ``param_names``) is copied
through untouched -- this script never recomputes a model. Only
``provenance``, ``generator_source`` and ``regenerate_recipe`` are rewritten,
because the npz's embedded guidance had drifted from what the analysis
actually does:

* ``provenance["selection"]["recommended"]`` said ``"xHI(z=5.9) < 0.1"``, a
  single-limb cut. The figure and every quoted statistic actually use
  ``selection.reionized_across_band``, a two-limb cut, and keep a different
  number of models. The header contradicted the analysis it documents.
* ``regenerate_recipe`` told a reader to apply the cut with
  ``selection.reionized()``, which does not implement the two-limb check,
  and claimed ``env_lock`` alone was enough to rebuild the environment,
  when ``uv sync`` also needs ``pyproject.toml`` (project name,
  ``requires-python``, the ``zeus21`` git source pin) and ``.python-version``.
* ``provenance["varied"]``'s ``Mc_III`` entry carried no note that the
  parameter is inert given ``alphastar_III = betastar_III = 0`` (see
  ``README.md``, "``Mc_III`` is sampled but has no effect").

``patched_utc`` and ``patch_note`` are added to the header so the edit is
auditable -- this file must never look like it was silently altered.

    uv run --project models_21cm python models_21cm/patch_npz_header.py \
        models_21cm/output/zeus21_models.npz

Pass ``--out`` to write elsewhere instead of patching in place.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import selection as sel

HERE = Path(__file__).resolve().parent

# What must come out bit-identical to what went in. Everything else in the
# npz is metadata and is fair game for this script.
DATA_KEYS = (
    "freqs_MHz",
    "T21_mK",
    "T21_native_mK",
    "z_native",
    "xHI",
    "z_xHI",
    "params",
    "kept_index",
    "param_names",
)

# generator_source used to bundle priors.py/provenance.py/selection.py/
# generate.py. pyproject.toml and .python-version join it here because
# env_lock (the uv.lock text) is not sufficient on its own to rebuild the
# generator environment -- see the module docstring's second bullet.
BUNDLE_FILES = (
    "priors.py",
    "provenance.py",
    "selection.py",
    "generate.py",
    "pyproject.toml",
    ".python-version",
)

MC_III_NOTE = (
    "Inert given astro_fixed.alphastar_III = astro_fixed.betastar_III = 0: "
    "Zeus21's fstarofz_III denominator, (Mh/Mc_III)**-alphastar_III + "
    "(Mh/Mc_III)**-betastar_III, collapses to 1 + 1 = 2 for every halo "
    "mass, independent of Mc_III (zeus21/sfrd.py). This column was sampled "
    "across the stated range but has zero effect on any model in this "
    "ensemble. Confirmed empirically as well as algebraically: partial "
    "rank regression of Mc_III against trough depth gives t = -0.10 "
    "(p = 0.92); marginal |Spearman| between Mc_III and each of five "
    "summary statistics is <= 0.009. See models_21cm/README.md, section "
    "'Mc_III is sampled but has no effect'."
)

PATCH_NOTE = (
    "Patched post-hoc by models_21cm/patch_npz_header.py; every data array "
    "is bit-identical to the pre-patch file. Changes: (1) "
    "provenance.selection rewritten from the single-limb recommendation "
    "'xHI(z=5.9) < 0.1' to the two-limb cut actually used downstream by "
    "make_paper_signal_loss_figure.load_t21 (selection.reionized_across_"
    "band), with the z_ref/x_max/z_top/x_max_top fields read from "
    "selection.py at patch time rather than hardcoded here; (2) a 'note' "
    "field added to the Mc_III entry in provenance.varied, documenting "
    "that it has no effect on any model given alphastar_III = "
    "betastar_III = 0; (3) generator_source re-embedded from the current "
    "priors.py/provenance.py/selection.py/generate.py, with pyproject.toml "
    "and .python-version appended, since env_lock alone cannot rebuild the "
    "generator environment; (4) regenerate_recipe corrected to point at "
    "reionized_across_band() and to describe the environment rebuild "
    "correctly. See models_21cm/README.md and "
    ".superpowers/sdd/2026-08-19-zeus21-model-ensemble/final-fix-report.md."
)

NEW_REGENERATE_RECIPE = """\
To regenerate this file with no access to the originating repository:

1. Read the `provenance` key (plain JSON, no pickle needed):
       import numpy as np, json
       d = np.load("zeus21_models.npz")
       header = json.loads(str(d["provenance"]))
2. Recreate the environment from the embedded `pyproject.toml`, `uv.lock`
   (the `env_lock` key) and `.python-version` -- all three are bundled
   verbatim in `generator_source` alongside the code, and all three are
   needed: `env_lock` alone is not sufficient, since `uv sync` also needs
   `pyproject.toml` for the project name, `requires-python`, and
   `[tool.uv.sources] zeus21 = { git = ..., rev = ... }`, and
   `.python-version` to pin the interpreter uv resolves against.
       # inside a directory holding the three extracted files above
       uv sync
3. Write out the `generator_source` key; it contains priors.py,
   provenance.py, selection.py, generate.py, pyproject.toml and
   .python-version verbatim as run.
4. Re-run the command in header["command_line"].

`params` holds the sampled values in TRANSFORMED units: header["varied"]
gives each column's name, transform ("log10" or "linear") and bounds, in
column order. A log10 column means Zeus21 received 10**value. Check each
entry's "note" key, if present, for known caveats (e.g. Mc_III, which is
sampled but has no effect on any model in this ensemble).

The reionization cut in header["selection"] has NOT been applied to the
stored arrays. Apply it with the stored `xHI` / `z_xHI` using
selection.reionized_across_band() from the extracted selection.py --
NOT selection.reionized(), which implements only the first of the cut's
two required limbs.
"""


def _read_text(path):
    return path.read_text() if path.exists() else ""


def build_bundle(here=HERE):
    """The generator_source bundle: BUNDLE_FILES, concatenated verbatim."""
    return "".join(
        f"\n\n# ---- {name} ----\n{_read_text(here / name)}" for name in BUNDLE_FILES
    )


def patch_header(old_header):
    """Return a new header dict; ``old_header`` is not mutated."""
    header = json.loads(json.dumps(old_header))  # deep copy, JSON round-trip

    header["selection"] = {
        "applied": False,
        "function": "selection.reionized_across_band",
        "z_ref": sel.Z_REION_REF,
        "x_max": sel.XHI_MAX,
        "z_top": sel.Z_BAND_TOP,
        "x_max_top": sel.XHI_MAX_BAND_TOP,
        "recommended": (
            f"xHI(z={sel.Z_REION_REF}) < {sel.XHI_MAX} AND "
            f"xHI(z={sel.Z_BAND_TOP}) < {sel.XHI_MAX_BAND_TOP}"
        ),
        "reference": (
            "McGreer et al. 2015 dark-pixel limit at the z="
            f"{sel.Z_REION_REF} reference redshift, plus reionization at "
            f"the band's top edge (z={sel.Z_BAND_TOP}, 250 MHz) to exclude "
            "models whose Zeus21 Q-based reionization ODE re-neutralizes "
            "below z~6"
        ),
    }

    varied = [dict(v) for v in header["varied"]]
    if not any(v["name"] == "Mc_III" for v in varied):
        raise RuntimeError("header['varied'] has no Mc_III entry to annotate")
    for v in varied:
        if v["name"] == "Mc_III":
            v["note"] = MC_III_NOTE
    header["varied"] = varied

    header["patched_utc"] = dt.datetime.now(dt.UTC).isoformat()
    header["patch_note"] = PATCH_NOTE
    return header


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path)
    p.add_argument(
        "--out", type=Path, default=None, help="write here instead of in place"
    )
    args = p.parse_args(argv)

    with np.load(args.npz, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}

    old_header = json.loads(str(data["provenance"]))
    new_header = patch_header(old_header)

    data["provenance"] = np.array(json.dumps(new_header, indent=2, sort_keys=True))
    data["generator_source"] = np.array(build_bundle())
    data["regenerate_recipe"] = np.array(NEW_REGENERATE_RECIPE)
    # env_lock is untouched -- still the real uv.lock text. It is simply no
    # longer claimed sufficient on its own; regenerate_recipe says why.

    missing = [k for k in DATA_KEYS if k not in data]
    if missing:
        raise RuntimeError(f"npz is missing expected data key(s): {missing}")

    out = args.out or args.npz
    np.savez_compressed(out, **data)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
