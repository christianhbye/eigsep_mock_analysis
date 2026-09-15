# Paper Figures Phase-1 Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the EIGSEP instrument paper's simulation figures (Fig. 1 `beam_comparison.pdf`, Fig. 13 `horizon_perturbations_1col.pdf`, Fig. 14 `horizon_shift.pdf`) and their deposited npz from the phase-1 simulation pipeline, hand CHB paste-ready text for every quoted number that moved, and re-pin provenance.

**Architecture:** Three repos, in one direction. `mock_analysis/horizon_position` simulates and writes the deposit npz into the paper notebooks directory (`paper.PAPER`). `papers/eigsep_instrument/notebooks` renders the PDFs from those npz and stages the Zenodo set. `papers/eigsep_instrument/eigsep_instrument_rasti` (Overleaf-synced) receives the PDFs and `docs/`; the `.tex` is edited by CHB in Overleaf only.

**Tech Stack:** uv, eigsim (JAX), croissant, nbconvert/nbclient, pytest, ruff.

**Spec:** None. Decision record: mock_analysis PR #18, "Two things to decide before merging", decision 1: **option A** (CHB, 2026-09-15): `output/position_sims.npz` stays the phase-1 rerun and the paper follows it, because the paper is not yet submitted. Dry run: on 2026-09-15 Figs. 13 and 14 were regenerated in a scratch directory from the phase-1 `position_sims.npz` without touching any repo; old-vs-new comparison at https://claude.ai/artifact/PaLhuehxJREvWBisgeCpsz. Its measured numbers pre-fill Task 5.

## Global Constraints

- **Never edit `rasti_template.tex` from git.** `eigsep_instrument_rasti` syncs with Overleaf on `main` only. Produce paste-ready text; CHB applies it in Overleaf. Figures and `docs/` go to `eigsep_instrument_rasti` `main` by commit, merged `--ff-only`.
- **Never delete or overwrite `mock_analysis/horizon_position/output/position_sims_rasti_round2.npz`.** `output/` is gitignored; it is the only local copy of the round-2 simulation.
- **npz and csv files are gitignored in the paper notebooks repo.** The Zenodo upload is CHB's; do not upload.
- **Strip embedded figures after every `nbconvert --execute --inplace`**: drop each `display_data` output that carries `image/png` (script in Task 3, Step 3). The repo convention is printed output, no embedded figures.
- **Every number quoted in the tex is read from the notebook cell that prints it**, never re-derived by hand from an npz and never rounded without recording the rule.
- **No push, tag push, PR, or Zenodo action without CHB's explicit go-ahead** for that step.
- Python via `uv run` (mock_analysis) or `~/Documents/research/papers/eigsep_instrument/.venv/bin/` (paper notebooks). No `python -c` for multiline scripts: write to a temp file.
- Conventional commits, each ending with the attribution line from the session's system reminder.
- `mock_analysis` is not in `eigsep_analysis/docs/branches.md` § 1; branches there are allowed.
- `$SCRATCH` means the executing session's scratchpad directory. Helper scripts go there, never into a repo.
- Stop conditions: any notebook `assert` failure, any `tests/` failure in the paper repo, or a bowtie-vs-nominal mismatch. Stop and report to CHB; do not edit `paper.py` constants (`N_ANCHOR`, `N_MODELS`, `N_SHOW_BEAM`) to make an assert pass.

---

### Task 0: Preconditions and archive

**Files:**
- Create (gitignored): `mock_analysis/horizon_position/output/rasti_round2_deposit/`

**Interfaces:**
- Produces: a complete byte-for-byte archive of the round-2 deposit set, and CHB's answers to the questions in Step 1.

- [ ] **Step 1: Ask CHB, and record the answers at the top of the Task 5 document**

1. Has PR #18 been merged into `mock_analysis` `main`? (Required: provenance strings must record a clean `main` commit.)
2. Tag name for the new figure state. Proposed: `rasti-round2-figs-v2`.
3. `eigsep_instrument_rasti` has an uncommitted `docs/figures.md` section ("Upstream data provenance") and an untracked `docs/deployment2025-rotation-data.md`. Commit them first as they are, or leave them to CHB?
4. Was the Zenodo version staged on 2026-09-11 (`notebooks/zenodo_upload/MANIFEST.txt`, "new version of record 20712533") already published? This decides whether Task 4 replaces an unpublished staging set or stages another version.
5. Should the npz `provenance` strings name a branch commit or the merged `main` commit? `horizon_shift.npz` records `mock_analysis@<sha>` when its notebook executes. If `main`, run Task 3 only after Task 6 Step 1's docs PR is merged. The arrays are identical either way; only the string differs (see Task 6 Step 2).

- [ ] **Step 2: Confirm clean trees**

```bash
git -C ~/Documents/research/eigsep/mock_analysis status -sb
git -C ~/Documents/research/papers/eigsep_instrument/notebooks status -sb
git -C ~/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti status -sb
```
Expected: mock_analysis on `main`, up to date, clean apart from untracked plan files. The paper notebooks repo is clean on `main`. `eigsep_instrument_rasti` shows only what Step 1 question 3 settled.

- [ ] **Step 3: Archive the round-2 deposit set**

```bash
MOCK=~/Documents/research/eigsep/mock_analysis/horizon_position
NB=~/Documents/research/papers/eigsep_instrument/notebooks
A=$MOCK/output/rasti_round2_deposit
mkdir -p $A
cp -p $NB/*.npz $NB/sparams.csv $A/
cp -p $MOCK/output/beam_sims.npz $MOCK/output/beam_bowtie.npz \
      $MOCK/output/beam_vivaldi.npz $MOCK/output/beam_isotropic.npz $A/
md5sum $A/foreground_svd.npz $A/horizon_shift.npz $A/horizon_perturbations.npz $A/beam_comparison.npz
grep -E "foreground_svd|horizon_shift|horizon_perturbations|beam_comparison" $NB/zenodo_upload/MANIFEST.txt
```
Expected: the four md5 sums match the MANIFEST rows. If one does not, stop: the live paper directory already diverged from the staged deposit.

---

### Task 1: Port `run_beam_sims.py` to the phase-1 mask

`run_beam_sims.py` still builds its mask with the local `masks.reduce_azimuth` + `masks.open_sky_weight` and never passes `config=`. The paper's `tests/test_npz_contract.py::test_beam_comparison_beams_are_in_the_figure_order` requires its bowtie row to match `foreground_svd.npz` to 1e-5, so once the deposit follows the phase-1 run, Fig. 1 cannot be regenerated without this port. The port takes every input from `run_sims.load_inputs`, which makes the bowtie row identical by construction. It also removes per-beam checkpoint resume, as PR #18 did for `run_sims.py`. A stale `beam_<tag>.npz` from the old pipeline would otherwise be silently reused.

**Files:**
- Modify: `horizon_position/run_beam_sims.py` (full rewrite below)
- Modify: `horizon_position/test_smoke.py` (add one test)
- Modify: `horizon_position/README.md` (step 2b and the checkpoint paragraph, around lines 84-100)
- Modify: `horizon_position/CLAUDE.md` (the `run_beam_sims.py` bullet and the "Do not modify `masks.py` or `run_beam_sims.py` while the paper is pinned" sentence)

**Interfaces:**
- Consumes: `run_sims.load_inputs(n_times, freq_stride=1) -> SimpleNamespace(cfg, names, enu, pos_sha, alpha_h, az_grid, beam_data, freqs_mhz, lmax, sky, times_jd, sky_alm)`; `run_sims.EIGSIM_CONFIG`, `run_sims.OUTPUT_DIR`, `run_sims.T_START`; `beams.band_limited_power_fraction`, `beams.healpix_to_mwss`, `beams.isotropic_beam`; `eigsim.open_sky_weight(alpha_h, az_grid, lmax)`.
- Produces: CLI `run_beam_sims.py [--n-times N] [--freq-stride K] [--beams TAG ...] [--output-tag TAG] [--vivaldi PATH]` → `output/beam_sims<tag>.npz`. Same keys as today (`t_sys, fgnd, beams, freqs_mhz, times_jd, t_start, n_times, t_ground, t_receiver, lon, lat, alt, sky_model, beam_lmax, vivaldi_source, band_limit_note, eigsim_version`) plus `pos_sha`.

- [ ] **Step 0: Branch**

```bash
cd ~/Documents/research/eigsep/mock_analysis
git switch -c fix/beam-sims-phase1-mask
```

- [ ] **Step 1: Write the failing test** (append to `horizon_position/test_smoke.py`)

```python
def test_run_beam_sims_bowtie_matches_run_sims_nominal():
    """beam_sims.npz's bowtie row is position_sims.npz's nominal row.

    Same beam, sky, horizon mask, times and config, so they must agree to
    float precision. The paper's test_npz_contract requires 1e-5 between
    beam_comparison.npz's bowtie and foreground_svd.npz; before the port the
    two scripts used different masks and frames and differed by kelvins.
    """
    if not (OUT / "horizons_position.npz").exists():
        pytest.skip("run make_horizons.py (eigsep_terrain env) first")
    tag = "_pytest_beams"
    common = ["--freq-stride", "40", "--n-times", "6", "--output-tag", tag]
    sims = OUT / f"position_sims{tag}.npz"
    beams = OUT / f"beam_sims{tag}.npz"
    try:
        subprocess.run(
            [sys.executable, str(HERE / "run_sims.py"), *common],
            check=True,
            cwd=HERE.parent,
        )
        subprocess.run(
            [
                sys.executable,
                str(HERE / "run_beam_sims.py"),
                *common,
                "--beams",
                "bowtie",
                "isotropic",
            ],
            check=True,
            cwd=HERE.parent,
        )
        s = np.load(sims, allow_pickle=True)
        b = np.load(beams, allow_pickle=True)
        assert [str(x) for x in b["beams"]] == ["bowtie", "isotropic"]
        i_nom = [str(n) for n in s["names"]].index("nominal")
        np.testing.assert_array_equal(b["freqs_mhz"], s["freqs_mhz"])
        np.testing.assert_array_equal(b["times_jd"], s["times_jd"])
        np.testing.assert_allclose(b["t_sys"][0], s["t_sys"][i_nom], rtol=1e-6)
        np.testing.assert_allclose(b["fgnd"][0], s["fgnd"][i_nom], rtol=1e-6)
        assert str(b["pos_sha"]) == str(s["pos_sha"])
    finally:
        sims.unlink(missing_ok=True)
        beams.unlink(missing_ok=True)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py::test_run_beam_sims_bowtie_matches_run_sims_nominal -v`
Expected: FAIL with `CalledProcessError` from `run_beam_sims.py`: `error: unrecognized arguments: --freq-stride 40 ... --output-tag _pytest_beams`.

- [ ] **Step 3: Rewrite `horizon_position/run_beam_sims.py`**

```python
"""Run the nominal-horizon sidereal day once per antenna, for the beam comparison.

Same site, same sky, same horizon, same times as `run_sims.py`'s nominal
position -- only the beam changes. Three of them:

  bowtie     the EIGSEP antenna, from eigsim's packaged MWSS beam
  vivaldi    the HERA Phase II feed used in isolation (no dish), which is what
             the October 2024 suspension flew; HEALPix, resampled by beams.py
  isotropic  a uniform beam: the chromaticity-free reference, still behind the
             real horizon

Every input comes from `run_sims.load_inputs` and the open-sky weight is built
exactly as run_sims.py builds its nominal row, so the bowtie row reproduces
position_sims.npz's nominal row (test_smoke.py pins it). The npz is written
once at the end; an interrupted run restarts from scratch.

`--vivaldi` points at the HEALPix beam file. It is not in this repo and not in
eigsim; pass the path or set EIGSEP_VIVALDI_BEAM.

Usage (from the monorepo root):
    uv run python horizon_position/run_beam_sims.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import eigsim  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beams import (  # noqa: E402
    band_limited_power_fraction,
    healpix_to_mwss,
    isotropic_beam,
)
from run_sims import EIGSIM_CONFIG, OUTPUT_DIR, T_START, load_inputs  # noqa: E402

DEFAULT_VIVALDI = "/home/christian/Documents/research/eigsep/eigsep_vivaldi.npz"
TAGS = ("bowtie", "vivaldi", "isotropic")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-times", type=int, default=1436)
    p.add_argument(
        "--freq-stride",
        type=int,
        default=1,
        help="use every Nth config frequency (smoke tests only)",
    )
    p.add_argument(
        "--beams",
        nargs="+",
        choices=TAGS,
        default=list(TAGS),
        help="antennas to simulate, in output order",
    )
    p.add_argument(
        "--output-tag",
        default="",
        help="suffix for output/beam_sims<tag>.npz (smoke tests only)",
    )
    p.add_argument(
        "--vivaldi",
        default=os.environ.get("EIGSEP_VIVALDI_BEAM", DEFAULT_VIVALDI),
        help="HEALPix Vivaldi beam npz (keys: freqs, bm, nside)",
    )
    return p.parse_args()


def load_beams(wanted, vivaldi_path, inp):
    """The requested beams on the bowtie's MWSS grid and ``inp.freqs_mhz``.

    Only builds what is asked for: the Vivaldi resample costs minutes.
    Returns ``(beams, note)``; ``note`` records the band-limit check, which is
    what licenses comparing a directive feed against a broad one on a grid
    sized for the latter.
    """
    beams, note = {}, ""
    if "bowtie" in wanted:
        beams["bowtie"] = inp.beam_data
    if "isotropic" in wanted:
        beams["isotropic"] = isotropic_beam(
            inp.freqs_mhz.size, inp.beam_data.shape[1:]
        )
    if "vivaldi" in wanted:
        vp = Path(vivaldi_path)
        if not vp.exists():
            raise SystemExit(
                f"{vp} not found -- pass --vivaldi or set EIGSEP_VIVALDI_BEAM"
            )
        viv = np.load(vp)
        viv_bm = viv["bm"][np.isin(viv["freqs"] / 1e6, inp.freqs_mhz)]
        if viv_bm.shape[0] != inp.freqs_mhz.size:
            raise SystemExit(
                f"vivaldi: {viv_bm.shape[0]} of {inp.freqs_mhz.size} config "
                "frequencies present in the beam file"
            )
        nside = int(viv["nside"])
        step = max(1, len(viv_bm) // 5)
        frac = band_limited_power_fraction(viv_bm[::step], nside, inp.lmax)
        note = (
            f"vivaldi band-limited power fraction at lmax={inp.lmax}: "
            f"{frac.min():.9f}"
        )
        print(f"  {note}")
        if frac.min() < 1 - 1e-4:
            raise SystemExit(
                "the Vivaldi is not resolved at the bowtie's band limit; the "
                "comparison would measure the grid, not the antenna"
            )
        print(f"  resampling vivaldi HEALPix (nside={nside}) -> MWSS...", flush=True)
        beams["vivaldi"] = healpix_to_mwss(viv_bm, nside, inp.lmax)
    return beams, note


def main():
    args = parse_args()
    inp = load_inputs(args.n_times, freq_stride=args.freq_stride)
    i_nom = inp.names.index("nominal")

    print(f"Loading beams for {', '.join(args.beams)}...")
    beams, note = load_beams(args.beams, args.vivaldi, inp)
    # Exactly run_sims.py's nominal mask: the native, unreduced horizon curve.
    # open_sky_weight's phi-cell integral is the band-limiting; reducing the
    # curve first would apply it twice (issue #10).
    W = eigsim.open_sky_weight(inp.alpha_h[i_nom], inp.az_grid, inp.lmax)

    OUTPUT_DIR.mkdir(exist_ok=True)
    t_sys, fgnd = [], []
    for tag in args.beams:
        print(f"  {tag:10s} simulating...", flush=True)
        t0 = time.time()
        ts = eigsim.simulate(
            beams[tag],
            inp.freqs_mhz,
            inp.sky,
            inp.times_jd,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
            sky_alm=inp.sky_alm,
            config=EIGSIM_CONFIG,
        )
        fg = eigsim.compute_fgnd(
            beams[tag],
            inp.freqs_mhz,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
            config=EIGSIM_CONFIG,
        )
        t_sys.append(np.asarray(ts)[0])
        fgnd.append(np.asarray(fg)[0])
        print(f"       done in {time.time() - t0:.0f}s")

    t_sys = np.stack(t_sys)
    fgnd = np.stack(fgnd)
    assert t_sys.shape == (len(args.beams), args.n_times, inp.freqs_mhz.size)

    out = OUTPUT_DIR / f"beam_sims{args.output_tag}.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,
        fgnd=fgnd,
        beams=np.array(args.beams),
        freqs_mhz=inp.freqs_mhz,
        times_jd=inp.times_jd,
        t_start=T_START,
        n_times=args.n_times,
        t_ground=inp.cfg["ground"]["temperature"],
        t_receiver=inp.cfg["receiver"]["temperature"],
        lon=inp.cfg["location"]["lon"],
        lat=inp.cfg["location"]["lat"],
        alt=inp.cfg["location"]["alt"],
        sky_model=inp.cfg["sky"]["model"],
        beam_lmax=inp.lmax,
        vivaldi_source=str(Path(args.vivaldi).name),
        band_limit_note=note,
        pos_sha=inp.pos_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"\nwrote {out}")
    for tag, f in zip(args.beams, fgnd):
        print(f"  {tag:10s} ground fraction {f.mean():.4f}  (eta {1 - f.mean():.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
uv run pytest horizon_position/ -v
uv run ruff check horizon_position && uv run ruff format --check horizon_position
```
Expected: both smoke tests PASS; the whole `horizon_position` suite passes (`test_masks.py` and `test_validation.py` still import `masks`, which stays); ruff clean.

- [ ] **Step 5: Update the docs**

In `horizon_position/README.md`, delete the `rm -f horizon_position/output/beam_{bowtie,isotropic,vivaldi}.npz  # step 2b` line and the paragraph beginning "`run_beam_sims.py` skips any antenna whose `beam_<tag>.npz` checkpoint". In their place under step 2b, add:

```markdown
#     Takes all inputs from run_sims.load_inputs and builds the same mask as
#     run_sims.py's nominal row, so its bowtie row reproduces position_sims.npz
#     row 0 (test_smoke.py). Written once at the end; nothing to resume.
```

In `horizon_position/CLAUDE.md`, replace the `run_beam_sims.py` bullet with:

```markdown
- `run_beam_sims.py` -> `output/beam_sims.npz` (eigsim env; one
  nominal-horizon sidereal day per beam, written once, not resumable).
  Inputs come from `run_sims.load_inputs` and the mask from
  `eigsim.open_sky_weight` exactly as run_sims.py's nominal row, so the
  bowtie row reproduces `position_sims.npz` row 0 (gated smoke test). The
  Vivaldi HEALPix beam lives outside this repo: `--vivaldi` /
  `EIGSEP_VIVALDI_BEAM`.
```
and delete the sentence "Do not modify `masks.py` or `run_beam_sims.py` while the paper is pinned at `rasti-round2-figs`."

- [ ] **Step 6: Remove the stale checkpoints** (already archived in Task 0)

```bash
rm horizon_position/output/beam_bowtie.npz horizon_position/output/beam_vivaldi.npz horizon_position/output/beam_isotropic.npz
```

- [ ] **Step 7: Commit, then ask CHB before pushing and opening the PR**

```bash
git add horizon_position/run_beam_sims.py horizon_position/test_smoke.py horizon_position/README.md horizon_position/CLAUDE.md
git commit -m "fix(horizon_position): build beam_sims from run_sims inputs and the phase-1 mask"
```
Task 2 runs only after this PR is merged into `main`.

---

### Task 2: Full simulation runs on `main`

Every product must trace to one clean `main` commit, so `position_sims.npz` is rerun too, even though a phase-1 copy exists. The existing copy doubles as a determinism check.

**Files:**
- Regenerate (gitignored): `horizon_position/output/position_sims.npz`, `horizon_position/output/beam_sims.npz`
- Create (gitignored): `horizon_position/output/position_sims_phase1_pr18.npz` (the PR #18 run, kept for comparison)

**Interfaces:**
- Consumes: Task 1 merged.
- Produces: `position_sims.npz` and `beam_sims.npz` from `main` at a recorded SHA.

- [ ] **Step 1: Sync and record the SHA**

```bash
cd ~/Documents/research/eigsep/mock_analysis
git fetch origin
# Local main held PR #18's three spec/plan docs commits (5b03bff, b651154, 77671cf)
# and lacked b2824f5 (croissant dev3, PR #17), so it cannot fast-forward. After
# PR #18 has merged, confirm those docs are on origin/main (expect no output):
git diff --stat origin/main feat/horizon-tilt-sensitivity -- \
    docs/superpowers/specs/2026-09-14-horizon-tilt-sensitivity-design.md \
    docs/superpowers/plans/2026-09-14-horizon-tilt-sensitivity-phase1.md
# then, with CHB's OK (this discards local main's copies of those commits):
git switch main && git reset --hard origin/main && uv sync --dev
git rev-parse --short HEAD   # record in the Task 5 document
```

- [ ] **Step 2: Rerun the 19 positions (~50 min)**

```bash
mv horizon_position/output/position_sims.npz horizon_position/output/position_sims_phase1_pr18.npz
uv run python horizon_position/run_sims.py
```

- [ ] **Step 3: Check determinism against the PR #18 run**

Write `$SCRATCH/cmp_sims.py`:

```python
import numpy as np

a = np.load("horizon_position/output/position_sims_phase1_pr18.npz", allow_pickle=True)
b = np.load("horizon_position/output/position_sims.npz", allow_pickle=True)
for k in ("t_sys", "fgnd", "freqs_mhz", "times_jd"):
    same = np.array_equal(a[k], b[k])
    rel = float(np.max(np.abs(a[k] - b[k]) / np.maximum(np.abs(a[k]), 1e-30)))
    print(f"{k:10s} byte-equal={same}  max rel diff={rel:.2e}")
assert str(a["pos_sha"]) == str(b["pos_sha"])
```
Run: `uv run python $SCRATCH/cmp_sims.py`
Expected: `freqs_mhz`, `times_jd` and `fgnd` byte-equal, and `t_sys` **not** byte-equal. The PR #18 run used croissant `v5.3.0.dev2`. `main` pins `v5.3.0.dev3` (PR #17), whose `rotmat_to_eulerZYZ` takes Euler angles from the nearest true rotation (croissant #152). That slightly changes the Earth-frame rotation behind the sky convolution. `fgnd` involves no sky, and the zenith drive rotation is exactly `I`, so it must not move: if it does, stop. Record `t_sys`'s max relative difference in the Task 5 document as the dev3 effect, and report it to CHB before Task 3.

- [ ] **Step 4: Run the three beams (Vivaldi at the default path, which exists)**

```bash
uv run python horizon_position/run_beam_sims.py
```
Expected: prints the Vivaldi band-limit note (≥ 0.9999) and three ground fractions. The bowtie value must equal `position_sims.npz` nominal `fgnd.mean()`.

- [ ] **Step 5: Check bowtie against nominal at full resolution**

Write `$SCRATCH/cmp_beams.py`:

```python
import numpy as np

s = np.load("horizon_position/output/position_sims.npz", allow_pickle=True)
b = np.load("horizon_position/output/beam_sims.npz", allow_pickle=True)
i_bow = [str(x) for x in b["beams"]].index("bowtie")
i_nom = [str(n) for n in s["names"]].index("nominal")
rel = np.abs(b["t_sys"][i_bow] - s["t_sys"][i_nom]) / s["t_sys"][i_nom]
print(f"bowtie vs nominal t_sys max rel diff {rel.max():.2e}")
assert rel.max() < 1e-6
```
Run: `uv run python $SCRATCH/cmp_beams.py`. Expected: PASS.

---

### Task 3: Regenerate the deposit npz in the paper directory

This overwrites the live files in `papers/eigsep_instrument/notebooks`. Task 0's archive is the rollback.

**Files:**
- Regenerate (gitignored, paper repo): `foreground_svd.npz`, `horizon_shift.npz`, `horizon_perturbations.npz`, `beam_comparison.npz`, `signal_loss.npz`
- Modify (mock_analysis, committed outputs): `horizon_position/notebooks/horizon_shift.ipynb`, `signal_loss.ipynb`, `beam_comparison.ipynb`

**Interfaces:**
- Consumes: Task 2 outputs; `EIGSEP_PAPER_NOTEBOOKS` **unset**, so `paper.PAPER` is the live paper directory.
- Produces: the new deposit npz, and executed notebooks whose printed tables are the source for Task 5.

- [ ] **Step 1: Branch and write `foreground_svd.npz`**

```bash
cd ~/Documents/research/eigsep/mock_analysis
git switch -c chore/paper-figures-phase1
env -u EIGSEP_PAPER_NOTEBOOKS uv run python horizon_position/make_foreground_svd.py --force
env -u EIGSEP_PAPER_NOTEBOOKS uv run python horizon_position/make_foreground_svd.py --check
```
Expected: `Saved .../foreground_svd.npz`, then `arrays MATCH`.

- [ ] **Step 2: Execute the notebooks in order** (each asserts against the previous one's output)

```bash
for nb in horizon_shift signal_loss beam_comparison; do
  env -u EIGSEP_PAPER_NOTEBOOKS uv run jupyter nbconvert --to notebook --execute --inplace \
      horizon_position/notebooks/$nb.ipynb || { echo "STOP: $nb failed"; break; }
done
```
Expected: all three complete. The dry run showed `horizon_shift` passes every assert on the phase-1 sims. `beam_comparison` is unverified: if `n_anchor["bowtie"] == paper.N_ANCHOR` or the "matches foreground_svd.npz" assert fails, stop and report.

- [ ] **Step 3: Strip embedded figures**

Write `$SCRATCH/strip_png.py`:

```python
import sys

import nbformat

for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = [
                o
                for o in cell.outputs
                if not (o.output_type == "display_data" and "image/png" in o.get("data", {}))
            ]
    nbformat.write(nb, path)
```
Run:
```bash
uv run python $SCRATCH/strip_png.py horizon_position/notebooks/{horizon_shift,signal_loss,beam_comparison}.ipynb
git diff --stat horizon_position/notebooks/
```
Expected: diffs of printed-output lines only (tens of lines per notebook, not thousands).

- [ ] **Step 4: Record the printed tables for Task 5**

```bash
for nb in horizon_shift beam_comparison; do
  jq -r '.cells[] | select(.cell_type=="code") | .outputs[]? | select(.output_type=="stream") | .text | join("")' \
     horizon_position/notebooks/$nb.ipynb > $SCRATCH/${nb}_printed_new.txt
  git show rasti-round2-figs:horizon_position/notebooks/$nb.ipynb | \
  jq -r '.cells[] | select(.cell_type=="code") | .outputs[]? | select(.output_type=="stream") | .text | join("")' \
     > $SCRATCH/${nb}_printed_old.txt
done
```

- [ ] **Step 5: Commit** (no push yet)

```bash
git add horizon_position/notebooks/horizon_shift.ipynb horizon_position/notebooks/signal_loss.ipynb horizon_position/notebooks/beam_comparison.ipynb
git commit -m "chore(horizon_position): regenerate paper figure data from the phase-1 pipeline"
```

---

### Task 4: Render the PDFs and stage the deposit (paper notebooks repo)

**Files:**
- Modify (committed): `notebooks/beam_comparison.pdf`, `horizon_shift.pdf`, `horizon_perturbations_1col.pdf`; the three generating `.ipynb`; `zenodo_upload/MANIFEST.txt`
- Refresh (gitignored): `zenodo_upload/*.npz`

**Interfaces:**
- Consumes: Task 3's npz in the notebooks directory.
- Produces: three committed PDFs at a recorded notebooks-repo SHA, and a staged Zenodo set whose md5 sums are in `MANIFEST.txt`.

- [ ] **Step 1: Execute the three figure notebooks**

```bash
cd ~/Documents/research/papers/eigsep_instrument/notebooks
J=../.venv/bin/jupyter
for nb in beam_comparison horizon_shift horizon_perturbations; do
  $J nbconvert --to notebook --execute --inplace $nb.ipynb || { echo "STOP: $nb failed"; break; }
done
../.venv/bin/python $SCRATCH/strip_png.py beam_comparison.ipynb horizon_shift.ipynb horizon_perturbations.ipynb
```
(`strip_png.py` needs only `nbformat`; if the paper venv lacks it, run the script with `uv run` from mock_analysis on these absolute paths.)

- [ ] **Step 2: Run the paper tests**

Run: `../.venv/bin/python -m pytest tests/ -v`
Expected: all pass, including `test_beam_comparison_beams_are_in_the_figure_order` (bowtie vs `foreground_svd.npz` < 1e-5, which only holds after Task 1) and `test_figure_box_matches_target` for all three PDFs.

- [ ] **Step 3: Visual check against round 2**

```bash
mkdir -p $SCRATCH/pdfcheck
for f in beam_comparison horizon_shift horizon_perturbations_1col; do
  git show HEAD:$f.pdf > $SCRATCH/pdfcheck/${f}_old.pdf
  pdftoppm -png -r 150 -singlefile $SCRATCH/pdfcheck/${f}_old.pdf $SCRATCH/pdfcheck/${f}_old
  pdftoppm -png -r 150 -singlefile $f.pdf $SCRATCH/pdfcheck/${f}_new
done
```
Look at each pair. Expected from the dry run: Fig. 13 unchanged, Fig. 14 changed in the North and East +1 m top panels only. Fig. 1 has no prior expectation: describe any change for CHB.

- [ ] **Step 4: Refresh the Zenodo staging set**

Run the refresh loop documented at the top of `zenodo_upload/MANIFEST.txt`, with `MOCK=~/Documents/research/eigsep/mock_analysis`, then `md5sum zenodo_upload/*.npz zenodo_upload/*.csv`. In `MANIFEST.txt`, update the BYTES and MD5 of every row that changed (expected: `position_sims.npz`, `beam_comparison.npz`, `foreground_svd.npz`, `horizon_shift.npz`, `horizon_perturbations.npz`). Add a dated section "THE 2026-09 PHASE-1 REGENERATION" stating: the fractional phi-integrated mask, the croissant frame fix `754627c`, croissant `v5.3.0.dev3` (including the #152 nearest-rotation Euler fix), the `mock_analysis` tag from Task 0 Step 1, and that the round-2 set is archived at `mock_analysis/horizon_position/output/rasti_round2_deposit/`. Word the header per CHB's answer to Task 0 question 4.

- [ ] **Step 5: Commit** (ask CHB before pushing)

```bash
git add beam_comparison.pdf horizon_shift.pdf horizon_perturbations_1col.pdf \
        beam_comparison.ipynb horizon_shift.ipynb horizon_perturbations.ipynb zenodo_upload/MANIFEST.txt
git commit -m "fix(figures): regenerate Figs 1, 13 and 14 from the phase-1 simulation pipeline"
git rev-parse --short HEAD   # record for docs/figures.md
```

---

### Task 5: Number-change table and paste-ready text

**Files:**
- Create: `eigsep_instrument_rasti/docs/phase1-number-changes.md`

**Interfaces:**
- Consumes: `$SCRATCH/*_printed_{old,new}.txt` (Task 3 Step 4); `horizon_shift.npz` `max_dT_full`; current `rasti_template.tex` pulled from `origin/main`.
- Produces: one row per quoted number (tex line, current quote, source print line, old printed value, new printed value, new quote), plus paste-ready replacement sentences. No `.tex` edit.

- [ ] **Step 1: Re-derive line numbers**

```bash
cd ~/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti && git pull --ff-only
grep -n "730\\\\,K\|1.2\\\\times10\|6 modes\|10 modes for\|0.36--0.55\|0.87\\\\,mK\|7.7 per cent\|near \$N=6\|up to 9.5\|after 7 modes\|up to 5.7\|0.70\\\\,mK\|0.53 and 0.73\|3.3\\\\,mK" rasti_template.tex
```

- [ ] **Step 2: Fill the table.** Start from the rows below. Dry-run values are measured from the phase-1 `position_sims.npz` (croissant `v5.3.0.dev2`) with the round-2 notebook code. They must be confirmed from Task 3's printed output, which runs on dev3. Rows marked *trace* are quotes whose rounding or aggregation rule is not yet known: find the cell that produced the quote before writing a new value.

| # | Quote in tex (today) | Source (notebook: printed table) | Round 2 printed | Dry run (phase-1) | New quote |
|---|---|---|---|---|---|
| 1 | Fig. 1 text: raw bowtie RMS "approximately 730 K" | horizon_shift: `foregrounds ... RMS` | 729.1 K | 729.0 K | 730 K (unchanged) |
| 2 | "one part in 1.2×10⁶" | horizon_shift: `foregrounds ... compression` | 1170236 | 1176159 | 1.2×10⁶ (unchanged) |
| 3 | "6 modes" isotropic, "10" bowtie, "18" Vivaldi to < 1 mK; excess "4" and "12" | beam_comparison: `modes filtered <1 mK` | 6 / 10 / 18 | not run | Task 3 |
| 4 | open-sky fraction "0.36--0.55" bowtie, "0.59--0.80" Vivaldi, "0.35" isotropic | beam_comparison: `eta min / eta max` | 0.357–0.549 / 0.587–0.796 / 0.347 | not run | Task 3 |
| 5 | N=10: median retained "0.87 mK", residual "0.62 mK" | horizon_shift: `N fg resid 21cm p50` row 10 | 0.623 / 0.870 | 0.620 / 0.868 | unchanged |
| 6 | retained "7.7" bowtie, "3.5" Vivaldi, "13.3" isotropic per cent | beam_comparison: `21-cm RMS retained there`, `<1 mK` column | 7.7 / 3.5 / 13.3 | not run | Task 3 |
| 7 | Fig. 1 caption: isotropic floor "near N=6" | beam_comparison: `modes filtered <0.1 mK` / figure | 7 | not run | Task 3 |
| 8 | Fig. 14 caption: "up to 9.5 K" | `horizon_shift.npz` `max_dT_full[2]` (all LSTs) | 9.462 | 9.458 | 9.5 K (unchanged) |
| 9 | Fig. 14 caption: every LST below median retained "after 7 modes" E, "2" N, "10" vertical | horizon_shift: `stays below from`, 1m rows | 7 / 2 / 10 | **10 / 4 / 10** | **10, 4, 10** |
| 10 | "up to 5.7 K to the east, 0.5 K to the north, and 9.5 K upward" | `horizon_shift.npz` `max_dT_full` | 5.707 / 0.549 / 9.462 | 6.267 / 0.556 / 9.458 | **6.3**, 0.6 (see note), 9.5 |
| 11 | N=10 floor rises "0.62 to 0.70" vertical, "0.63" eastward, "unchanged at 0.62" northward | horizon_shift: `nominal basis`, ±1 m rows. *Trace* (the quote looks like the mean of ±) | z 0.695/0.713, x 0.631/0.635, y 0.626/0.620 | z 0.692/0.711, x 0.630/0.633, y 0.623/0.617 | if mean of ±: 0.70, 0.63, 0.62 (unchanged) |
| 12 | 10 m own-basis floor "between 0.53 and 0.73 mK, against 0.62" | horizon_shift: `own basis`, all ±10 m rows | 0.530–0.728 | 0.527–0.726 | unchanged |
| 13 | 10 m floor "3.3 mK vertically, 3.8 times the retained signal, 1.4 mK eastward and 0.67 mK northward" | horizon_shift: `nominal basis`, ±10 m rows. *Trace* | z 3.486/3.207, x 1.254/1.463, y 0.694/0.654 | z 3.479/3.199, x 1.255/1.444, y 0.699/0.653 | if mean of ±: 3.3, 3.8, **1.3** (1.3495), **0.68** (0.676) |
| 14 | "induced floor overtaking the retained signal only between 1 and 10 m" | horizon_shift: `stays below from` Up rows, at N=10 | 1 m 0.35, 10 m 3.37 mK vs 0.87 | 0.35 / 3.37 | unchanged |

Note on row 10: 0.549 K rounded to "0.5" and 0.556 K rounds to "0.6". Ask CHB whether to keep one decimal (0.6) or say "about 0.55 K".

- [ ] **Step 3: Write paste-ready sentences** for every row whose new quote differs from the current one: the full sentence as it stands, then the full sentence as it should read. Quote the tex verbatim from `origin/main`. Hand the document to CHB; do not edit `rasti_template.tex`.

- [ ] **Step 4: Commit the document on `eigsep_instrument_rasti` `main`** (together with Task 6 Step 3; ask before pushing)

---

### Task 6: Re-pin provenance and retire the pinned-paper guards

**Files:**
- Modify (mock_analysis): `horizon_position/README.md` (section "The re-run and the paper's copy"), `horizon_position/CLAUDE.md` (the "`output/position_sims.npz` is the phase-1 re-run, NOT the paper's" bullet), `horizon_position/make_foreground_svd.py` (module docstring's "DO NOT RUN THIS WHILE THE PAPER IS PINNED" paragraph)
- Modify (eigsep_instrument_rasti): `docs/figures.md`; copy in `beam_comparison.pdf`, `horizon_shift.pdf`, `horizon_perturbations_1col.pdf`

**Interfaces:**
- Consumes: Task 3's mock_analysis commit, Task 4's notebooks SHA, the tag name from Task 0.

- [ ] **Step 1: mock_analysis docs, on `chore/paper-figures-phase1`**

Replace the README section "The re-run and the paper's copy (read before step 2a)" with:

```markdown
## The paper's simulation and the round-2 archive

`output/position_sims.npz` is the simulation behind the paper's figures as of
tag `<TAG>`: the phase-1 pipeline (`eigsim.open_sky_weight`'s phi-integrated
mask, croissant's fixed-pole frame fix `754627c`, croissant `v5.3.0.dev3` with the #152 Euler fix).
Steps 2a-3 regenerate the deposit from it.

- `output/position_sims_rasti_round2.npz` and `output/rasti_round2_deposit/`
  hold the round-2 simulation and the deposit set built from it (tag
  `rasti-round2-figs`). **Never delete them**: `output/` is gitignored, so
  they are the only local copies.
- `make_foreground_svd.py` still refuses to overwrite a deposit whose arrays
  differ; pass `--force` when regenerating deliberately.
```

In `CLAUDE.md`, replace the "`output/position_sims.npz` is the phase-1 re-run, NOT the paper's" bullet with:

```markdown
- `output/position_sims.npz` is the paper's simulation as of tag `<TAG>`
  (phase-1 pipeline). The round-2 run is archived as
  `output/position_sims_rasti_round2.npz` plus `output/rasti_round2_deposit/`;
  never delete either (`output/` is gitignored). `make_foreground_svd.py`
  refuses to overwrite a differing deposit without `--force`.
```

In `make_foreground_svd.py`, replace the paragraph from "DO NOT RUN THIS WHILE THE PAPER IS PINNED" through "...being regenerated deliberately." with:

```
A plain run REFUSES to overwrite a deposit whose arrays differ from what it
would write, so a stray run cannot silently move the published figures;
`--force` overrides when the figures are being regenerated deliberately. The
round-2 deposit is archived under output/rasti_round2_deposit/.
```
Replace `<TAG>` with Task 0's answer. Run `uv run pytest horizon_position/ && uv run ruff check horizon_position`, then commit: `docs(horizon_position): the paper follows the phase-1 simulation`. Push, open the PR and, after merge, tag the merge commit (`git tag -a <TAG> -m "mock_analysis state behind the phase-1 paper figures"`), each only on CHB's go-ahead.

- [ ] **Step 2: Rebuild provenance strings if the tag moved them.** `horizon_shift.npz` records `mock_analysis@<sha>` at execution time (Task 3, on the branch). If CHB wants the provenance to name the merged `main` commit, rerun Task 3 Steps 1-3 and Task 4 Steps 1-2 and 4 from `main` after the merge. The arrays come back byte-identical; only `provenance` changes. Ask CHB which they prefer before Task 3, because this decides whether Task 3 runs on the branch or after merging a docs-only PR.

- [ ] **Step 3: `eigsep_instrument_rasti` `main`**

```bash
cd ~/Documents/research/papers/eigsep_instrument/eigsep_instrument_rasti
cp -p ../notebooks/beam_comparison.pdf ../notebooks/horizon_shift.pdf ../notebooks/horizon_perturbations_1col.pdf .
```
In `docs/figures.md`: set the `NB commit` and `date` cells of the `beam_comparison.pdf`, `horizon_perturbations_1col.pdf` and `horizon_shift.pdf` rows to Task 4's SHA and date. In "Upstream data provenance", point both npz rows at tag `<TAG>`. Replace the paragraph "**`mock_analysis` `main` no longer reproduces these outputs.** ..." with:

```markdown
**Regenerated on the phase-1 pipeline (2026-09).** The fractional horizon
mask, croissant's frame fix (`754627c`) and croissant `v5.3.0.dev3` moved
Fig. 14's horizontal-displacement panels and several §4.7 numbers; see
`docs/phase1-number-changes.md`. The round-2 state stays reproducible from tag
`rasti-round2-figs` with the archived inputs in
`mock_analysis/horizon_position/output/rasti_round2_deposit/`.
```

```bash
git add beam_comparison.pdf horizon_shift.pdf horizon_perturbations_1col.pdf docs/figures.md docs/phase1-number-changes.md
git commit -m "fix(figures): phase-1 regeneration of Figs 1, 13 and 14, and the number changes"
```
Push to `main` on CHB's go-ahead (`git push origin main`; `--ff-only` for any merge). CHB pulls in Overleaf and applies Task 5's text.

- [ ] **Step 4: Hand-offs to CHB** (list them in the final report; do not act)

1. Apply Task 5's paste-ready text in Overleaf.
2. Upload the staged Zenodo set (`notebooks/zenodo_upload/`, per `MANIFEST.txt`). The concept DOI in the tex needs no edit.
3. Post the drafted issue #10 closing comment (`mock_analysis/.superpowers/sdd/2026-09-14-horizon-tilt-sensitivity-phase1/task-7-report.md`), noting that the paper now uses the fractional mask.
