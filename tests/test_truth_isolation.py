"""Sensors, perception and control must never read oracle (*_truth.npz) data."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tar_swarm"
GUARDED = ("sensors", "perception", "control")
# evaluation.logging writes the truth files; importing it is a path to them.
FORBIDDEN_MODULES = ("tar_swarm.evaluation",)


def truth_violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "_truth" in node.value or "truth.npz" in node.value:
                found.append(f"string {node.value!r}")
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
            )
            names = [alias.name for alias in node.names]
            for module in modules:
                if module.startswith(FORBIDDEN_MODULES) or "truth" in module:
                    found.append(f"import {module}")
            found += [f"import {name}" for name in names if "truth" in name.lower()]
    return found


def test_detector_flags_truth_access():
    assert truth_violations('np.load(run / "lidar_truth.npz")')
    assert truth_violations("from tar_swarm.evaluation.logging import save_lidar")
    assert truth_violations("from tar_swarm.io import load_truth")
    assert not truth_violations('np.load(run / "lidar_scans.npz")')


def test_guarded_modules_do_not_read_truth():
    files = [
        path
        for package in GUARDED
        if (PACKAGE / package).is_dir()
        for path in (PACKAGE / package).rglob("*.py")
    ]
    assert files
    violations = {
        str(path.relative_to(PACKAGE)): found
        for path in files
        if (found := truth_violations(path.read_text(encoding="utf-8")))
    }
    assert not violations, violations
