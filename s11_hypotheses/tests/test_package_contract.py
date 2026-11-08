"""Actual installed/source contract; no substituted analysis physics."""

import ast
import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import s11_hypotheses
from s11_hypotheses import install_contract, runtime
from s11_hypotheses.run import required_sources


def test_installed_src_layout_and_true_dependencies():
    project = Path(__file__).resolve().parents[1]
    assert Path(s11_hypotheses.__file__).resolve().is_relative_to(project / "src")
    requires = metadata.requires("eigsep-s11-hypothesis-tools")
    assert all(
        any(r.lower().startswith(name) for r in requires)
        for name in [
            "numpy",
            "h5py",
            "pandas",
            "scipy",
            "eigsep_analysis",
        ]
    )
    assert (
        runtime.verify_environment()["analysis_commit"]
        == runtime.contract()["analysis_commit"]
    )


def test_correct_head_with_dirty_scientific_file_is_refused(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "uv.lock").write_text("frozen test lock\n")
    (tmp_path / "science.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    policy = dict(
        analysis_commit=head,
        analysis_lock_sha256=hashlib.sha256(
            (tmp_path / "uv.lock").read_bytes()
        ).hexdigest(),
    )
    assert runtime.verify_checkout(tmp_path, policy)[0] == head
    (tmp_path / "science.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="dirty tracked"):
        runtime.verify_checkout(tmp_path, policy)
    assert (
        subprocess.check_output(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
        ).strip()
        == head
    )


def test_installed_cli_imports_the_real_companion():
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "s11_hypotheses.run", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "physical-shard" in completed.stdout


def test_delivery_preserves_original_mathematics_and_choices():
    project = Path(__file__).resolve().parents[1]
    original = json.loads(Path(__file__).with_name("b9_math_source.json").read_text())
    package = project / "src/s11_hypotheses"
    assert original["original_producer"] == runtime.contract()["original_science_mock"]
    for name, expected in original["functions"].items():
        tree = ast.parse((package / (name + ".py")).read_text())
        actual = {
            node.name: hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert actual == expected
    assert (
        hashlib.sha256((package / "config.py").read_bytes()).hexdigest()
        == original["config_sha256"]
    )


def test_preflight_lists_the_actual_historical042_runtime_input():
    primary = "/private/MIST/mistic_analysis/products/042_s11_noise_floor/summary.json"
    store = SimpleNamespace(entries={primary: {}, "/unrelated/file": {}})
    assert required_sources(store) == [primary]


def test_local_build_source_cannot_escape_owned_prefix_via_symlink(tmp_path):
    owner = tmp_path / "owned"
    analysis = owner / "analysis"
    analysis.mkdir(parents=True)
    protected = tmp_path / "protected"
    protected.mkdir()
    (owner / "sibling").symlink_to(protected, target_is_directory=True)
    (analysis / "uv.lock").write_text(
        '[[package]]\nname = "dependency"\nsource = { editable = "../sibling" }\n'
    )
    with pytest.raises(ValueError, match="escapes owned prefix"):
        install_contract.owned_sources(analysis, owner)
    assert list(protected.iterdir()) == []


def test_retained_distribution_keeps_real_metadata_and_refuses_overwrite(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    site = source / "lib/python3.12/site-packages"
    package = site / "healjax"
    dist = site / "healjax-0.2.2.dist-info"
    package.mkdir(parents=True)
    dist.mkdir()
    (package / "__init__.py").write_text("# fixture payload, no science imported\n")
    record = b"fixture RECORD bytes are retained exactly\n"
    (dist / "RECORD").write_bytes(record)
    (dist / "METADATA").write_text("Name: healjax\nVersion: 0.2.2\n")
    result = install_contract.retain_healjax(source, destination)
    assert result["revision_claim"] is None
    copied = destination / "lib/python3.12/site-packages"
    assert (copied / dist.name / "RECORD").read_bytes() == record
    for name, digest in result["files"].items():
        assert hashlib.sha256((copied / name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="overwrite"):
        install_contract.retain_healjax(source, destination)
