"""No vendor imports or AEDT execution: real parsing and fake-desktop safety checks."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from eigsim import hfss_boundary_pilot as hp


def block(name, text="", children=()):
    return "$begin '%s'\n%s\n%s\n$end '%s'\n" % (name, text, "".join(children), name)


def project_text():
    parts = [
        block(
            "GeometryPart",
            children=[
                block("Attributes", "Name='%s'\nSolveInside=true" % name),
                block("Operation", "ObjectID=%d\nWidth='25mm'" % i),
            ],
        )
        for i, name in enumerate(hp.PARTS + ["RadiatingSurface"])
    ]
    model = block(
        "HFSSModel",
        "Name='%s'" % hp.DESIGN,
        [
            block(
                "ModelSetup",
                children=[
                    block(
                        "GeometryCore",
                        "Units='mm'",
                        [
                            block("ToplevelParts", children=parts),
                            block("OperandParts", "HistoryID=123"),
                            block("CoordinateSystems"),
                        ],
                    ),
                    block("Properties", "VariableProp('Width', 'UD', '', '25mm')"),
                    block("DesignDatasets"),
                    block(
                        "AssignedEntities",
                        children=[block("AssignedFace", "kID=54050")],
                    ),
                ],
            ),
            block(
                "BoundarySetup",
                children=[
                    block(
                        "Boundaries",
                        children=[
                            block(
                                name, "ModeNum=1\nImpedance='50ohm'\nRenormImp='75ohm'"
                            )
                            for name in ["3", "PerfE1", "PerfE2", "AutoOpen1"]
                        ],
                    )
                ],
            ),
            *[
                block(name, "setting=true")
                for name in [
                    "DrivenOptions",
                    "Solutions",
                    "RadField",
                    "MeshSetup",
                    "AnalysisSetup",
                    "Optimetrics",
                ]
            ],
        ],
    )
    return block(
        "AnsoftProject",
        "VariableOrders[0:]",
        [
            block(
                "Definitions", children=[block("Materials", "conductivity=38000000")]
            ),
            block("ProjectDatasets"),
            model,
        ],
    ) + block("AllReferencedFilesForProject", "File='dependency.dat'")


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source.aedt"
    source.write_text(project_text())
    prefix = tmp_path / "private"
    prefix.mkdir()
    target = prefix / "draft.aedt"
    script, guard = [prefix / n for n in ["prepare.py", "guard.py"]]
    script.write_text("# exact rendered vendor source")
    guard.write_text("# exact guard")
    pin = dict(
        source_sha256=hp.sha256(source),
        protected_inputs={str(source): hp.sha256(source)},
        design=hp.DESIGN,
        physical_signature_sha256=hp.digest(hp.clone_signature(source)),
    )
    gate = dict(
        schema=hp.SCHEMA,
        phase="prepare",
        source=str(source),
        target=str(target),
        private_prefix=str(prefix),
        padding_mm=400,
        approved_by="root",
        decision_record="recorded concrete source gate",
        source_sha256=pin["source_sha256"],
        source_pin_sha256=hp.digest(pin),
        script_sha256=hp.sha256(script),
        guard_sha256=hp.sha256(guard),
    )
    return source, target, prefix, script, guard, pin, gate


@pytest.mark.parametrize(
    "old,new",
    [
        ("conductivity=38000000", "conductivity=37000000"),
        ("Width='25mm'", "Width='26mm'"),
        ("'25mm')", "'26mm')"),
        ("Impedance='50ohm'", "Impedance='75ohm'"),
        ("ModeNum=1", "ModeNum=2"),
        ("kID=54050", "kID=54051"),
        ("HistoryID=123", "HistoryID=124"),
        ("File='dependency.dat'", "File='different.dat'"),
        ("VariableOrders[0:]", "VariableProp('$a', 'UD', '', '3mm')"),
        ("setting=true", "setting=false"),
    ],
)
def test_physical_material_variable_source_and_dependency_drift(case, old, new):
    source, target, *_ = case
    target.write_text(source.read_text().replace(old, new))
    with pytest.raises(ValueError, match="drift"):
        hp.compare(hp.clone_signature(source), hp.clone_signature(target))


def test_units_unknown_design_and_unbalanced_blocks(case):
    source = case[0]
    for old, new in [
        ("Units='mm'", "Units='m'"),
        (hp.DESIGN, "unknown"),
        ("$end 'Solutions'", "$end 'Other'"),
    ]:
        source.write_text(project_text().replace(old, new))
        with pytest.raises(ValueError):
            hp.clone_signature(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("phase", "solve"),
        ("source_pin_sha256", "0" * 64),
        ("padding_mm", 800),
        ("source_sha256", "0" * 64),
        ("script_sha256", "0" * 64),
        ("guard_sha256", "0" * 64),
        ("approved_by", "self"),
        ("decision_record", ""),
        ("target", "/data/source.aedt"),
    ],
)
def test_gate_rejects_wrong_phase_identity_or_paths(case, key, value):
    source, target, prefix, script, guard, pin, gate = case
    gate[key] = value
    with pytest.raises(ValueError):
        hp.validate_gate(
            gate, str(source), str(target), str(prefix), script, guard, pin
        )


def test_symlink_escape_and_existing_results_rejected(case):
    source, target, prefix, *_ = case
    escape = prefix / "escape"
    escape.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="inside private"):
        hp.private_target(str(source), str(escape / "x.aedt"), str(prefix))
    Path(str(target) + "results").mkdir()
    with pytest.raises(ValueError, match="results already"):
        hp.private_target(str(source), str(target), str(prefix))


def test_bounds_controls_and_no_nonfinite_or_unit_drift():
    legacy = [-600, -700, -800, 600, 700, 800]
    hp.bounds_check(legacy, legacy, 400)
    hp.bounds_check(legacy, [-1700, -1800, -1900, 1700, 1800, 1900], 1500)
    for fresh, padding in [
        (legacy, 1500),
        (legacy, 800),
        ([float("nan")] * 6, 400),
        ([0] * 5, 400),
    ]:
        with pytest.raises(ValueError):
            hp.bounds_check(legacy, fresh, padding)


class Project:
    def __init__(self, desktop, path=None):
        self.desktop, self.path = desktop, path

    def SetActiveDesign(self, name):
        assert name == hp.DESIGN

    def CopyDesign(self, name):
        self.desktop.events.append("CopyDesign")

    def Paste(self):
        self.desktop.events.append("Paste")

    def SaveAs(self, path, overwrite):
        assert overwrite is False
        self.desktop.events.append("SaveAs_NEW")
        self.path = Path(path)
        shutil.copyfile(self.desktop.source, path)
        if self.desktop.drift:
            self.path.write_text(self.path.read_text().replace("38000000", "1"))

    def GetPath(self):
        return str(self.path.parent)

    def GetName(self):
        return self.path.stem


class Desktop:
    def __init__(self, source, projects=(), wrong_active=False, drift=False):
        self.source, self.projects = source, projects
        self.events, self.wrong_active, self.drift = [], wrong_active, drift

    def GetProjectList(self):
        return self.projects

    def OpenProject(self, path):
        assert path == str(self.source)
        self.original = Project(self, self.source)
        self.events.append("OpenProject")
        return self.original

    def NewProject(self):
        self.clone = Project(self)
        self.events.append("NewProject")
        return self.clone

    def GetActiveProject(self):
        return self.original if self.wrong_active else self.clone


def run_clone(case, desktop):
    _, _, _, script, guard, pin, gate = case
    return hp.clone_design(desktop, gate, pin, script, guard)


def test_clone_new_save_context_validation_then_stop(case):
    desktop = Desktop(case[0])
    result = run_clone(case, desktop)
    assert desktop.events == [
        "OpenProject",
        "CopyDesign",
        "NewProject",
        "Paste",
        "SaveAs_NEW",
    ]
    assert result["phase"] == "CLONE_VALIDATED"
    assert hp.sha256(case[0]) == case[5]["source_sha256"]
    # Fake API intentionally exposes no Save/DeleteFullVariation/Analyze method.


@pytest.mark.parametrize(
    "kwargs", [dict(projects=["active"]), dict(wrong_active=True), dict(drift=True)]
)
def test_no_takeover_wrong_active_project_and_clone_drift(case, kwargs):
    desktop = Desktop(case[0], **kwargs)
    with pytest.raises(ValueError):
        run_clone(case, desktop)
    assert hp.sha256(case[0]) == case[5]["source_sha256"]
    if kwargs.get("projects"):
        assert desktop.events == []


def test_duplicate_target_and_unknown_object_stop_before_vendor(case):
    source, target, *_ = case
    target.write_text("already exists")
    desktop = Desktop(source)
    with pytest.raises(ValueError, match="exists"):
        run_clone(case, desktop)
    assert not desktop.events
    source.write_text(project_text().replace("Name='Box4'", "Name='Unrecognized'"))
    with pytest.raises(ValueError, match="inventory"):
        hp.clone_signature(source)


def test_pin_contract_has_no_scientific_adoption():
    pin = json.loads(
        (
            Path(hp.__file__).parent / "pins" / "hfss_boundary_source_v0001.json"
        ).read_text()
    )
    assert (
        pin["source_sha256"]
        == "ccd57db3c2f75307a15b522efa6f52cc03e8bb611a6e6dceaa2608dc57b011a2"
    )
    assert pin["physical_signature_sha256"] == hp.digest(pin["signature"])
    assert "no physical efficiency adoption" in pin["label"]
    altered = copy.deepcopy(pin["signature"])
    altered["objects"]["Box1"] = "wrong"
    with pytest.raises(ValueError, match="objects"):
        hp.compare(pin["signature"], altered)


def test_changed_stored_field_stops_before_vendor(case):
    source, _, prefix, _, _, pin, gate = case
    field = prefix / "original-field.evtrs_0"
    field.write_bytes(b"exact frozen coefficients")
    pin["protected_inputs"][str(field)] = hp.sha256(field)
    gate["source_pin_sha256"] = hp.digest(pin)
    field.write_bytes(b"changed coefficients")
    desktop = Desktop(source)
    with pytest.raises(ValueError, match="field identity"):
        run_clone(case, desktop)
    assert desktop.events == []


def test_render_is_code_only_handles_quoted_paths_and_refuses_overwrite(case):
    import runpy

    source, _, prefix, _, guard, _, gate = case
    planner = Path(hp.__file__).parents[2] / "scripts/hfss_boundary/plan.py"
    render = runpy.run_path(str(planner))["render"]
    config = prefix / "config's name.json"
    config.write_text(json.dumps(gate))
    pin_path = prefix / "source-pin.json"
    pin_path.write_text("{}")
    output = prefix / "rendered.py"
    result = render(config, guard, pin_path, output)
    compile(output.read_text(), str(output), "exec")
    assert result["phase"] == "RENDERED_ONLY"
    assert not Path(gate["target"]).exists()
    assert hp.sha256(source) == gate["source_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        render(config, guard, pin_path, output)
    with pytest.raises(ValueError, match="private prefix"):
        render(config, guard, pin_path, prefix.parent / "escaped.py")
