"""Vendor template: rendered prefixes define CONFIG_PATH, GUARD_PATH and PIN_PATH.

Run ONLY in a fresh isolated ansysedt -ng -RunScriptAndExit process after the
orchestrator's explicit clone-prepare gate. Only private SaveAs is allowed;
no original Save, Analyze or cleanup call exists.
"""

from __future__ import print_function

import io
import json
import os
import traceback

import imp


def main(desktop, config_path, guard_path, pin_path, script_path):
    guard = imp.load_source("c56_guard", guard_path)
    with io.open(config_path, encoding="utf8") as stream:
        config = json.load(stream)
    with io.open(pin_path, encoding="utf8") as stream:
        pin = json.load(stream)
    guard.validate_gate(
        config,
        config["source"],
        config["target"],
        config["private_prefix"],
        script_path,
        guard_path,
        pin,
    )
    result_path = os.path.join(config["private_prefix"], "CLONE_RESULT.json")
    guard.require(not os.path.lexists(result_path), "Result already exists")
    try:
        result = guard.clone_design(desktop, config, pin, script_path, guard_path)
    except Exception:
        result = {
            "phase": "FAILED",
            "traceback": traceback.format_exc(),
            "source_sha256_after": guard.sha256(config["source"]),
            "protected_after": dict(
                (p, guard.sha256(p)) for p in pin["protected_inputs"]
            ),
            "source_unchanged": guard.sha256(config["source"]) == pin["source_sha256"],
        }
        guard.write_result(result_path, result)
        raise
    guard.write_result(result_path, result)
    return result


if __name__ == "__main__":
    import ScriptEnv

    ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
    main(oDesktop, CONFIG_PATH, GUARD_PATH, PIN_PATH, __file__)  # noqa: F821
