"""Freeze actual-frequency field operands for an unchanged native model.

Runs in the fitted producer's runtime. The later model consumes numeric
coefficients/grids without installing a different harmonic-basis package.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .assets import atomic_npz, checked_file, sha256, write_json
from .native_field_export import grid, load_basis

DELETIONS = [9, 10, 12, 14, 15, 17, 19, 20, 22, 24, 25, 27, 29]


def role_penalties(fits, selected, strong):
    """Require the complete matched ensemble and unambiguous prior role."""
    if not 0 < selected < strong:
        raise ValueError("strong penalty must exceed selected positive penalty")
    expected = {
        "raster_hybrid": selected,
        "represented_hfss_control": selected,
        "penalty_" + str(strong): strong,
    }
    expected.update({"delete_stripe_" + str(k): selected for k in DELETIONS})
    roles = [item["role"] for item in fits]
    if len(roles) != len(expected) or set(roles) != set(expected):
        raise ValueError("exact main/prior/strong/13-deletion roles required")
    if roles[0] != "raster_hybrid":
        raise ValueError("main fit must anchor experimental identities first")
    for item in fits:
        role = item["role"]
        if item.get("represented_prior", False) != (role == "represented_hfss_control"):
            raise ValueError("only represented-HFSS prior role may use c0")
        expected_id = (
            int(role.removeprefix("delete_stripe_"))
            if role.startswith("delete_stripe_")
            else None
        )
        if item.get("deleted_training_stripe") != expected_id:
            raise ValueError("role/deletion mapping differs")
    return expected


def run(config, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    prepared_done = checked_file(config["prepared"])
    prepared = prepared_done.parent
    record = json.loads(prepared_done.read_text())
    for name, digest in record["output_hashes"].items():
        if sha256(prepared / name) != digest:
            raise ValueError("prepared native field changed")
    with np.load(prepared / "representation.npz", allow_pickle=False) as z:
        c0 = z["c0"]
        lmax = int(z["lmax"])
    basis = load_basis(prepared / "spectral_basis.npz")
    channels = np.arange(264, 984)
    members = channels.reshape(-1, 4)
    centres = members.mean(1) * 250 / 1024
    f = np.unique(np.r_[channels * 250 / 1024, centres])
    if len(f) != 900 or f.min() < basis.freqs.min() or f.max() > basis.freqs.max():
        raise ValueError("C57 frequency scope outside native representation")
    spectral = basis.evaluate(f)
    labels, coefficients, provenance = [], [], []
    selection = json.loads(checked_file(config["selection"]).read_text())
    selected = selection["selection"]["selected_penalty"]
    penalties = role_penalties(config["fits"], selected, config["strong_penalty"])
    deletion_ids = []
    reference_identities = None
    reference_sources = None
    identity_names = [
        "t",
        "channels",
        "split",
        "original_fit_channels",
        "folds",
        "stripes",
        "normalization_train",
    ]
    for item in config["fits"]:
        label = item["role"]
        if label in labels:
            raise ValueError("duplicate beam role")
        done_path = checked_file(item["done"])
        done = json.loads(done_path.read_text())
        if not done["fit_converged"]:
            raise ValueError("unconverged fit cannot supply field operands")
        directory = done_path.parent
        parent = checked_file(item["record"])
        if parent.parent != directory:
            raise ValueError("fit record path differs from fit directory")
        parent_record = json.loads(parent.read_text())
        if parent_record["prepared_done_sha256"] != config["prepared"]["sha256"]:
            raise ValueError("fit/basis parent mismatch")
        if (
            parent_record["penalty"] != penalties[label]
            or done["optimizer"]["penalty"] != penalties[label]
        ):
            raise ValueError("fit penalty differs from matched role")
        if reference_sources is None:
            reference_sources = parent_record["source_sha256"]
        elif parent_record["source_sha256"] != reference_sources:
            raise ValueError("matched fit producer source differs")
        if parent_record["withheld_frequency_fold"] is not None:
            raise ValueError("frequency holdout fit is not an adopted field")
        for name, digest in done["output_hashes"].items():
            if sha256(directory / name) != digest:
                raise ValueError("fitted payload changed")
        with np.load(directory / "identities.npz", allow_pickle=False) as z:
            identities = {key: z[key] for key in identity_names}
            np.testing.assert_array_equal(z["fit_channels"], z["original_fit_channels"])
            if reference_identities is None:
                reference_identities = identities
            else:
                for key in identity_names:
                    np.testing.assert_array_equal(
                        identities[key], reference_identities[key]
                    )
            if "deleted_training_stripe" not in item:
                np.testing.assert_array_equal(z["train"], z["normalization_train"])
        if "deleted_training_stripe" in item:
            stripe = item["deleted_training_stripe"]
            if parent_record["deleted_training_stripe"] != stripe:
                raise ValueError("deletion identity changed")
            with np.load(directory / "identities.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(
                    z["train"], z["normalization_train"] & (z["stripes"] != stripe)
                )
            deletion_ids.append(stripe)
        elif parent_record["deleted_training_stripe"] is not None:
            raise ValueError("unlabelled deletion fit")
        with np.load(directory / "fit.npz", allow_pickle=False) as z:
            c = c0 if item.get("represented_prior", False) else z["coeff"]
        if c.shape != c0.shape:
            raise ValueError("component-major spectral coefficient layout changed")
        labels.append(label)
        coefficients.append(spectral @ c)
        provenance.append(item)
    expected = DELETIONS
    if sorted(deletion_ids) != expected or len(set(deletion_ids)) != 13:
        raise ValueError("complete matched13deletion identities required")
    atomic_npz(
        out / "coefficients.npz",
        frequencies_mhz=f,
        roles=np.array(labels),
        frequency_coefficients=np.stack(coefficients),
        native_channels=channels,
        group_members=members,
        nominal_centres_mhz=centres,
        lmax=lmax,
        deletion_ids=np.array(expected),
    )
    for L in [129, 180]:
        y, r, w = grid(L, lmax)
        atomic_npz(
            out / f"angular_L{L}.npz",
            Y=y,
            directions=r,
            weights=w,
            L=L,
            field_lmax=lmax,
        )
    outputs = ["coefficients.npz", "angular_L129.npz", "angular_L180.npz"]
    write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic; shape-only field operands",
            roles=labels,
            prepared_done_sha256=config["prepared"]["sha256"],
            fits=provenance,
            role_to_deletion={
                item["role"]: item.get("deleted_training_stripe")
                for item in config["fits"]
            },
            selection_sha256=config["selection"]["sha256"],
            source_sha256=sha256(__file__),
            frequencies=900,
            native_frequencies=720,
            nominal_centres=180,
            no_independent_observations_added=True,
            output_hashes={n: sha256(out / n) for n in outputs},
            elapsed_s=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    run(json.loads(a.inputs.read_text()), a.out)
