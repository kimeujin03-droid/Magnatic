import importlib
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES_TO_COMPILE = [
    "path_utils.py",
    "analyze_mms.py",
    "analyze_whistler_burst.py",
    "analyze_whistler_baseline.py",
    "analyze_event_coupling.py",
    "build_ml_dataset.py",
]


def check_py_compile() -> None:
    """Compile key scripts to catch syntax errors without requiring data files."""
    for relative_path in FILES_TO_COMPILE:
        py_compile.compile(str(ROOT / relative_path), doraise=True)


def check_path_utils() -> None:
    """Verify the shared path helper resolves repository-local defaults."""
    path_utils = importlib.import_module("path_utils")
    repo_dir = path_utils.repo_base_dir()
    if repo_dir != ROOT:
        raise AssertionError(f"repo_base_dir mismatch: {repo_dir} != {ROOT}")

    case_dir = path_utils.resolve_case_dir(default_case="demo_case")
    expected = ROOT / "cases" / "demo_case"
    if case_dir != expected:
        raise AssertionError(f"resolve_case_dir mismatch: {case_dir} != {expected}")


def main() -> None:
    check_py_compile()
    check_path_utils()
    print("smoke test passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        raise
