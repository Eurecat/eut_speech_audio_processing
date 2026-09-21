"""Work around a pkg_resources/PEP 503 naming mismatch for dotted project names.

Some packages (e.g. pyannote.audio, ruamel.yaml) have a literal dot in their
PyPI project name. Modern wheel builds normalize that to an underscore in the
on-disk ``*.dist-info`` directory name (e.g. ``pyannote_audio-3.4.0.dist-info``),
but the system's setuptools ``pkg_resources`` (used by nemo/hydra at runtime,
see asr.launch.py's PYTHONPATH injection) does not apply PEP 503 normalization
when parsing a ``Requirement`` string: ``Requirement.parse("pyannote.audio>=2.1.1").key``
stays ``"pyannote.audio"`` (dot preserved), which then fails to match the
installed distribution's key ``"pyannote-audio"`` (derived from the
underscored directory name). Anything that depends on such a package
declaring the dotted spelling (diart -> pyannote.audio, speechbrain's
hyperpyyaml -> ruamel.yaml, ...) then raises
``pkg_resources.DistributionNotFound`` even though the package is installed
and importable.

Fix: for every ``*.dist-info`` whose METADATA "Name:" field contains a dot
that isn't reflected in the directory name, duplicate that directory using
the literal, dotted name. This only adds metadata pkg_resources can find; it
never touches the actual importable package files.
"""

import os
import re
import shutil
import sys


def fix_dotted_distinfo(site_packages: str) -> list[str]:
    created = []
    for entry in sorted(os.listdir(site_packages)):
        if not entry.endswith(".dist-info"):
            continue
        path = os.path.join(site_packages, entry)
        metadata_path = os.path.join(path, "METADATA")
        if not os.path.isfile(metadata_path):
            continue
        name = None
        with open(metadata_path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                    break
        if not name or "." not in name:
            continue
        m = re.match(r"^(.*)-([^-]+)\.dist-info$", entry)
        if not m:
            continue
        _, version = m.group(1), m.group(2)
        dotted_dirname = f"{name}-{version}.dist-info"
        if dotted_dirname == entry:
            continue
        dest = os.path.join(site_packages, dotted_dirname)
        if os.path.exists(dest):
            continue
        shutil.copytree(path, dest)
        created.append(dotted_dirname)
    return created


if __name__ == "__main__":
    created = fix_dotted_distinfo(sys.argv[1])
    print(f"Created {len(created)} dot-preserving dist-info aliases:")
    for c in created:
        print(" ", c)
