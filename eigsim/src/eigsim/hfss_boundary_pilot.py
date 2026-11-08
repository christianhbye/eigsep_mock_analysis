"""Fail-closed, stdlib-only C56 AEDT preparation contracts.

Kept Python-2.7 compatible for the installed AEDT IronPython interpreter.
No import of eigsim, numerical backend or vendor automation occurs here.
"""

from __future__ import print_function

import hashlib
import io
import json
import math
import os
import re
import sys

SCHEMA = "eigsep.hfss-boundary-prepare.v1"
DESIGN = "1_0_Exp_Taper_BowTie_Balun8_5 CharlieMod9"
PARTS = sorted(
    [
        "Box1",
        "Rectangle1",
        "Rectangle3",
        "Port3",
        "Polyline9",
        "Polyline9_1",
        "EquationCurve1",
        "EquationCurve1_2",
        "Box4",
    ]
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PY2_JSON = sys.version_info[0] == 2
# IronPython2 represents both plain and Unicode text as str; decoding such
# Unicode again as Latin1 can corrupt or reject characters above U+00FF.
_PY2_BYTE_STRING = str if _PY2_JSON and type("") is not str else ()


def _latin1_json_strings(value):
    """Preserve native Latin1 bytes as Unicode before Python2 JSON encoding.

    IronPython can return byte strings from the Latin1 text reader. Its JSON
    encoder otherwise interprets them as UTF8. Python3 strings are unchanged;
    arbitrary Python3 bytes remain invalid JSON input.
    """
    if isinstance(value, _PY2_BYTE_STRING):
        return value.decode("latin1")
    if isinstance(value, dict):
        return dict(
            (_latin1_json_strings(k), _latin1_json_strings(v)) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return [_latin1_json_strings(v) for v in value]
    return value


def _ascii_json_bytes(value):
    """ASCII-escape stdlib JSON output without redecoding unified text.

    The vendor Python2 ASCII encoder incorrectly decodes Unicode as UTF8.
    Keep stdlib key ordering, numbers, quoting and control escapes, then escape
    characters outside printable ASCII exactly as its normal ASCII path does.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    chunks = []
    for char in text:
        number = ord(char)
        if number < 127:
            chunks.append(char)
        elif number <= 65535:
            chunks.append("\\u%04x" % number)
        else:
            number -= 65536
            chunks.append(
                "\\u%04x\\u%04x" % (55296 + (number >> 10), 56320 + (number & 1023))
            )
    return "".join(chunks).encode("ascii")


def digest(value):
    normalized = _latin1_json_strings(value)
    if _PY2_JSON:
        payload = _ascii_json_bytes(normalized)
    else:
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
            "utf8"
        )
    return hashlib.sha256(payload).hexdigest()


def read_tree(path):
    """Parse balanced native blocks, retaining all direct data lines and ordering."""
    root = {"name": "ROOT", "lines": [], "children": []}
    stack = [root]
    with io.open(path, encoding="latin1") as stream:
        for line in stream.read().splitlines():
            match = re.match(r"\s*\$(begin|end) '([^']+)'\s*$", line)
            if match:
                action, name = match.groups()
                if action == "begin":
                    node = {"name": name, "lines": [], "children": []}
                    stack[-1]["children"].append(node)
                    stack.append(node)
                else:
                    require(
                        len(stack) > 1 and stack[-1]["name"] == name,
                        "Unbalanced AEDT block: " + name,
                    )
                    stack.pop()
            elif line.strip():
                stack[-1]["lines"].append(line.strip())
    require(len(stack) == 1, "Unclosed AEDT block")
    return root


def find(node, name):
    result = []
    for child in node["children"]:
        if child["name"] == name:
            result.append(child)
        result.extend(find(child, name))
    return result


def one(node, name):
    matches = find(node, name)
    require(len(matches) == 1, "Expected one " + name)
    return matches[0]


def canonical(node):
    """Only volatile material timestamps are omitted; IDs remain significant."""
    return {
        "name": node["name"],
        "lines": [s for s in node["lines"] if not s.startswith("ModTime=")],
        "children": [canonical(c) for c in node["children"]],
    }


def snapshot(path, design=DESIGN):
    """Conservative saved-source signature; unknown/missing sections fail closed.

    Top-level objects and ALL operand history are preserved, including IDs.
    Materials/variables/CS/source/PEC/port/sphere/mesh rules are independent keys.
    UI history and numerical domain/setup are deliberately outside this signature.
    """
    root = read_tree(path)
    models = [n for n in find(root, "HFSSModel") if "Name='%s'" % design in n["lines"]]
    require(len(models) == 1, "Unknown or ambiguous design")
    model = models[0]
    ms = one(model, "ModelSetup")
    top = one(ms, "ToplevelParts")
    objects = {}
    for part in top["children"]:
        require(part["name"] == "GeometryPart", "Unknown top-level geometry")
        attrs = one(part, "Attributes")
        names = [s[6:-1] for s in attrs["lines"] if s.startswith("Name='")]
        require(len(names) == 1 and names[0] not in objects, "Ambiguous object")
        objects[names[0]] = part
    require(
        sorted(objects) == sorted(PARTS + ["RadiatingSurface"]),
        "Physical object inventory changed",
    )
    boundary = one(model, "BoundarySetup")
    boundaries = one(boundary, "Boundaries")
    require(
        sorted(c["name"] for c in boundaries["children"])
        == ["3", "AutoOpen1", "PerfE1", "PerfE2"],
        "Boundary inventory changed",
    )
    signatures = {"objects": dict((n, digest(canonical(objects[n]))) for n in PARTS)}
    for name in ["OperandParts", "CoordinateSystems", "Properties", "DesignDatasets"]:
        signatures[name] = digest(canonical(one(ms, name)))
    # Assigned face/edge geometry is retained, excluding only the numerical region ID
    # in AssignedObject. Physical topology also remains in each object history.
    assigned = one(ms, "AssignedEntities")
    signatures["assigned_topology"] = digest(
        [canonical(c) for c in assigned["children"]]
    )
    for name in ["DrivenOptions", "Solutions", "RadField", "MeshSetup"]:
        signatures[name] = digest(canonical(one(model, name)))
    for name in ["3", "PerfE1", "PerfE2"]:
        signatures["boundary_" + name] = digest(canonical(one(boundaries, name)))
    # Compare all project definitions (including custom materials/dependencies),
    # rather than replacing missing project definitions with a local library.
    signatures["Definitions"] = digest(canonical(one(root, "Definitions")))
    project = [n for n in find(root, "AnsoftProject")]
    require(len(project) == 1, "Unknown project container")
    project_properties = [
        c for c in project[0]["children"] if c["name"] == "Properties"
    ]
    signatures["project_variables"] = digest([canonical(c) for c in project_properties])
    units = [
        s
        for n in find(ms, "GeometryCore")
        for s in n["lines"]
        if s.startswith("Units=")
    ]
    require(units == ["Units='mm'"], "Model units must be mm")
    signatures["units"] = units
    return signatures


def compare(source, candidate):
    require(
        source == candidate,
        "Physical/source/material/variable signature drift: "
        + ", ".join(
            sorted(
                k
                for k in set(source) | set(candidate)
                if source.get(k) != candidate.get(k)
            )
        ),
    )


def private_target(source, target, prefix, must_be_absent=True):
    paths = [source, target, prefix]
    require(all(os.path.isabs(p) for p in paths), "Absolute paths required")
    if must_be_absent:
        require(not os.path.lexists(target), "Target already exists")
        require(not os.path.lexists(target + "results"), "Target results already exist")
    source, target, prefix = [os.path.realpath(p) for p in paths]
    require(
        prefix not in ("/", "/tmp", "/home", "/data"),
        "Dedicated private prefix required",
    )
    require(
        target.startswith(prefix + os.sep) and target.endswith(".aedt"),
        "Target must be an AEDT file inside private prefix",
    )
    require(
        source != target and not source.startswith(prefix + os.sep),
        "Private prefix overlaps source",
    )
    if must_be_absent:
        require(not os.path.lexists(target), "Target already exists")
        require(not os.path.exists(target + "results"), "Target results already exist")
    return target


def active_clone(project, target, source, prefix):
    private_target(source, target, prefix, must_be_absent=False)
    actual = os.path.join(str(project.GetPath()), str(project.GetName()) + ".aedt")
    require(
        os.path.realpath(actual) == os.path.realpath(target),
        "Active project is not the approved saved clone",
    )


def bounds_check(legacy, fresh, padding_mm, tolerance_mm=1e-9):
    require(padding_mm in (400, 1500), "Only predeclared ordinary RBC cases allowed")
    require(len(legacy) == len(fresh) == 6, "Six domain bounds required")
    values = [float(x) for x in list(legacy) + list(fresh)]
    require(
        all(not math.isnan(v) and not math.isinf(v) for v in values), "Nonfinite bounds"
    )
    delta = padding_mm - 400
    expected = [float(v) + (-delta if i < 3 else delta) for i, v in enumerate(legacy)]
    require(
        all(abs(float(a) - b) <= tolerance_mm for a, b in zip(fresh, expected)),
        "Fresh numerical domain does not match declared padding/bounds",
    )


def validate_gate(gate, source, target, prefix, script_path, guard_path, pin):
    require(
        gate.get("schema") == SCHEMA and gate.get("phase") == "prepare",
        "Explicit prepare gate required; solve/export are separate gates",
    )
    require(
        gate.get("source_sha256") == pin["source_sha256"] == sha256(source),
        "Source byte identity failed",
    )
    require(
        gate.get("script_sha256") == sha256(script_path)
        and gate.get("guard_sha256") == sha256(guard_path),
        "Code pin mismatch",
    )
    require(
        gate.get("target") == target and gate.get("private_prefix") == prefix,
        "Gate path mismatch",
    )
    require(gate.get("padding_mm") in (400, 1500), "Undeclared case")
    require(
        gate.get("approved_by") == "root" and bool(gate.get("decision_record")),
        "Missing orchestrator decision record",
    )
    require(
        gate.get("source_pin_sha256") == digest(pin), "Source contract pin mismatch"
    )
    private_target(source, target, prefix)
    protected_hashes(pin)


def protected_hashes(pin):
    """Verify original project, solution index and stored-field coefficients."""
    expected = pin["protected_inputs"]
    require(bool(expected), "Protected source ledger missing")
    actual = dict((path, sha256(path)) for path in expected)
    require(actual == expected, "Protected project/solution/field identity changed")
    return actual


def clone_design(desktop, config, pin, script_path, guard_path):
    """Save a NEW project, validate it, then STOP. Never clean/solve either project.

    Validation can fail after SaveAs: preserve that draft as FAILED, never repair
    missing variables/materials, overwrite it or call Save on the original.
    """
    source, target, prefix = [config[k] for k in ("source", "target", "private_prefix")]
    validate_gate(config, source, target, prefix, script_path, guard_path, pin)
    initial = clone_signature(source, pin["design"])
    require(
        digest(initial) == pin["physical_signature_sha256"],
        "Source signature pin mismatch",
    )
    # Caller must provide a fresh isolated desktop. Never attach to an active project.
    require(
        not list(desktop.GetProjectList()), "Desktop already owns projects; no takeover"
    )
    original = desktop.OpenProject(source)
    require(original is not None, "OpenProject failed")
    original.SetActiveDesign(pin["design"])
    original.CopyDesign(pin["design"])
    clone = desktop.NewProject()
    clone.Paste()
    # The first write is a new SaveAs, before any clone cache cleanup is possible.
    private_target(source, target, prefix)
    clone.SaveAs(target, False)
    active_clone(desktop.GetActiveProject(), target, source, prefix)
    compare(initial, clone_signature(target, pin["design"]))
    protected_after = protected_hashes(pin)
    require(sha256(source) == pin["source_sha256"], "Original source changed")
    return {
        "schema": SCHEMA,
        "phase": "CLONE_VALIDATED",
        "target": target,
        "target_sha256": sha256(target),
        "protected_after": protected_after,
        "source_sha256": sha256(source),
        "physical_signature_sha256": digest(initial),
        "padding_mm": config["padding_mm"],
        "limits": "NO cleanup, region edit, setup edit, solve or export performed",
    }


def clone_signature(path, design=DESIGN):
    """At clone-only stage, numerical setup/domain are NOT allowed to change."""
    root = read_tree(path)
    models = [n for n in find(root, "HFSSModel") if "Name='%s'" % design in n["lines"]]
    require(len(models) == 1, "Unknown or ambiguous design")
    result = snapshot(path, design)
    result["design_header"] = digest(
        [s for s in models[0]["lines"] if not s.startswith("DesignID=")]
    )
    for name in ["AnalysisSetup", "ModelSetup", "BoundarySetup", "Optimetrics"]:
        result["clone_" + name] = digest(canonical(one(models[0], name)))
    project = one(root, "AnsoftProject")
    result["project_variable_lines"] = digest(
        [
            s
            for s in project["lines"]
            if s.startswith(("VariableOrders", "VariableProp("))
        ]
    )
    result["ProjectDatasets"] = digest(canonical(one(project, "ProjectDatasets")))
    # Includes embedded report files: conservative, potential safe-stop on CopyDesign.
    result["AllReferencedFilesForProject"] = digest(
        canonical(one(root, "AllReferencedFilesForProject"))
    )
    return result


def write_result(path, value):
    """Create one result without following an existing or dangling symlink."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
