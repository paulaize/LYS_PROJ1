from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = {
    "nibabel": "nibabel",
    "SimpleITK": "SimpleITK",
    "pandas": "pandas",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "napari": "napari",
}

OPTIONAL = {
    "ants": "antspyx",
    "brkraw": "brkraw",
    "paquo": "paquo",
}


def check_module(module_name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - env audit should show import errors directly
        return False, repr(exc)
    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()} / machine={platform.machine()}")

    failed_required: list[str] = []

    print("\nRequired packages:")
    for module_name, package_name in REQUIRED.items():
        ok, detail = check_module(module_name)
        status = "OK" if ok else "MISSING"
        print(f"  {status:8} {package_name:14} import {module_name} -> {detail}")
        if not ok:
            failed_required.append(package_name)

    print("\nOptional packages:")
    for module_name, package_name in OPTIONAL.items():
        ok, detail = check_module(module_name)
        status = "OK" if ok else "not installed"
        print(f"  {status:12} {package_name:14} import {module_name} -> {detail}")

    if failed_required:
        print("\nMissing required packages:", ", ".join(failed_required))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
