import os
from pathlib import Path


def repo_base_dir() -> Path:
    """Return the repository base directory used by standalone scripts.

    Resolution order is:
    1. `MAGNETIC_BASE_DIR`
    2. The directory containing this file
    """
    env_value = os.environ.get("MAGNETIC_BASE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parent


def resolve_case_dir(case_dir: str | None = None, default_case: str | None = None) -> Path:
    """Resolve a case directory with a shared fallback policy.

    Resolution order is:
    1. explicit function argument
    2. `MMS_CASE_DIR`
    3. `<repo>/cases/<default_case>` when provided
    4. repository root
    """
    if case_dir:
        return Path(case_dir).expanduser().resolve()

    env_value = os.environ.get("MMS_CASE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    base_dir = repo_base_dir()
    if default_case:
        return (base_dir / "cases" / default_case).resolve()
    return base_dir
