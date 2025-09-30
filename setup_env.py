"""Setup script for creating and managing Python virtual environment.

This educational script demonstrates:
- Virtual environment creation and management
- Cross-platform compatibility (Windows/Unix)
- Dependency installation from requirements.in
- Automatic requirements.txt generation via pip freeze
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Windows-focused but works cross-platform

PROJECT_ROOT: Path = Path(__file__).parent.resolve()
VENV_DIR: Path = PROJECT_ROOT / ".venv"
PIP_PATH: Path = VENV_DIR / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
PYTHON_PATH: Path = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ_IN: Path = PROJECT_ROOT / "requirements.in"
REQ_TXT: Path = PROJECT_ROOT / "requirements.txt"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute a subprocess command with error handling.

    Args:
        cmd: Command and arguments as a list.
        check: Whether to raise CalledProcessError on non-zero exit.

    Returns:
        CompletedProcess instance with command results.

    Raises:
        subprocess.CalledProcessError: If command fails and check=True.
    """
    print(f"[setup_env] Running: {' '.join(map(str, cmd))}")
    try:
        return subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[setup_env] Command failed with exit code {e.returncode}: {e.cmd}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        raise


def ensure_venv() -> None:
    """Create virtual environment if it doesn't exist or is incomplete.

    Checks for existence of pip/python executables. If .venv exists but
    is incomplete, recreates it from scratch.
    """
    if VENV_DIR.exists() and (PIP_PATH.exists() or PYTHON_PATH.exists()):
        print(f"[setup_env] Found existing virtual environment at {VENV_DIR}")
        return
    if VENV_DIR.exists() and not (PIP_PATH.exists() or PYTHON_PATH.exists()):
        print("[setup_env] .venv directory exists but appears incomplete. Recreating it...")
        shutil.rmtree(VENV_DIR)
    print(f"[setup_env] Creating virtual environment at {VENV_DIR} using {sys.executable}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def ensure_pip_and_build_tools() -> None:
    """Upgrade pip, setuptools, and wheel to latest versions.

    Skipped if SETUP_ENV_OFFLINE environment variable is set.
    """
    if os.getenv("SETUP_ENV_OFFLINE"):
        print("[setup_env] Offline mode: skipping upgrade of pip/setuptools/wheel")
        return
    run([str(PIP_PATH), "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_requirements() -> None:
    """Install dependencies from requirements.in or requirements.txt.

    Prefers requirements.in if it exists, otherwise falls back to requirements.txt.
    Skipped if SETUP_ENV_OFFLINE or SETUP_ENV_NO_INSTALL environment variables are set.
    """
    if os.getenv("SETUP_ENV_OFFLINE") or os.getenv("SETUP_ENV_NO_INSTALL"):
        print("[setup_env] Offline/No-Install mode: skipping installation of requirements")
        return
    if REQ_IN.exists():
        print(f"[setup_env] Installing dependencies from {REQ_IN.name}")
        run([str(PIP_PATH), "install", "-r", str(REQ_IN)])
    elif REQ_TXT.exists():
        print(f"[setup_env] Installing dependencies from {REQ_TXT.name}")
        run([str(PIP_PATH), "install", "-r", str(REQ_TXT)])
    else:
        print("[setup_env] No requirements.in or requirements.txt found. Nothing to install.")


def freeze_requirements() -> None:
    """Generate fully pinned requirements.txt from installed packages.

    Uses pip freeze to capture exact versions of all installed packages.
    Skipped if SETUP_ENV_OFFLINE or SETUP_ENV_NO_INSTALL environment variables are set.
    """
    if os.getenv("SETUP_ENV_OFFLINE") or os.getenv("SETUP_ENV_NO_INSTALL"):
        print("[setup_env] Offline/No-Install mode: skipping freeze to requirements.txt")
        return
    print(f"[setup_env] Freezing installed packages to {REQ_TXT.name}")
    result = subprocess.run([str(PIP_PATH), "freeze"], capture_output=True, text=True, check=True)
    pinned = result.stdout.strip() + "\n"
    REQ_TXT.write_text(pinned, encoding="utf-8")
    print(f"[setup_env] Wrote {REQ_TXT} with {len(pinned.splitlines())} entries")


def verify_core_requirements_installed() -> None:
    """Verify that all requirements from requirements.in are installed.

    Parses requirements.in to extract package names (handling extras and version
    specifiers), checks if they're installed, and attempts to install any missing packages.
    """
    if not REQ_IN.exists():
        return
    print("[setup_env] Verifying that top-level requirements from requirements.in are installed...")
    result = subprocess.run([str(PIP_PATH), "list", "--format=freeze"], capture_output=True, text=True, check=True)
    installed = {line.split("==")[0].lower() for line in result.stdout.splitlines() if line and not line.startswith("-")}
    missing = []
    for line in REQ_IN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line
        for sep in ["[", ">", "<", "=", "~", "!", ";", " "]:
            if sep in name:
                name = name.split(sep, 1)[0]
        pkg = name.strip().lower()
        if pkg and pkg not in installed:
            missing.append(line)
    if missing:
        print("[setup_env] Warning: Some requirements from requirements.in appear missing in the venv:")
        for m in missing:
            print(f"  - {m}")
        print("[setup_env] Attempting to install missing packages...")
        run([str(PIP_PATH), "install", *missing])
    else:
        print("[setup_env] All top-level requirements from requirements.in are present.")


def main() -> None:
    """Main setup workflow: create venv, install dependencies, and freeze versions."""
    print("[setup_env] Project root:", PROJECT_ROOT)
    ensure_venv()
    ensure_pip_and_build_tools()
    install_requirements()
    verify_core_requirements_installed()
    freeze_requirements()
    print("[setup_env] Done. To use the environment:")
    if os.name == "nt":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
