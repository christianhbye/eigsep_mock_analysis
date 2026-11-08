"""Render a clone-only vendor script; never launch AEDT or issue an approval."""

import argparse
import json
from pathlib import Path

from eigsim.hfss_boundary_pilot import private_target, sha256


def render(config_path, guard_path, pin_path, output):
    config_path, guard_path, pin_path = map(
        lambda p: Path(p).resolve(), (config_path, guard_path, pin_path)
    )
    # Preserve the final path component so exclusive create sees dangling links.
    output = Path(output).absolute()
    config = json.loads(config_path.read_text())
    private_target(config["source"], config["target"], config["private_prefix"])
    if output.exists() or output.is_symlink():
        raise ValueError("Rendered script already exists")
    if not output.resolve().is_relative_to(Path(config["private_prefix"]).resolve()):
        raise ValueError("Rendered script must be inside private prefix")
    prefix = "\n".join(
        "%s = %r" % (key, str(path))
        for key, path in [
            ("CONFIG_PATH", config_path),
            ("GUARD_PATH", guard_path),
            ("PIN_PATH", pin_path),
        ]
    )
    template = Path(__file__).with_name("prepare_aedt.py").read_text()
    # Future import must remain first in the rendered file.
    template = template.replace("from __future__ import print_function\n", "")
    with output.open("x", encoding="utf8") as stream:
        stream.write(
            "from __future__ import print_function\n" + prefix + "\n" + template
        )
    return {
        "rendered_script": str(output),
        "script_sha256": sha256(output),
        "guard_sha256": sha256(guard_path),
        "phase": "RENDERED_ONLY",
        "next": "Root must pin these hashes and authorize a separate isolated launch",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["config", "guard", "pin", "output"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.config, args.guard, args.pin, args.output), indent=2))
