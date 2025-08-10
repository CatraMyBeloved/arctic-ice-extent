import os
import sys
import subprocess
import shutil
from pathlib import Path

# Windows-focused but works cross-platform

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
PIP_PATH = VENV_DIR / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
PYTHON_PATH = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ_IN = PROJECT_ROOT / "requirements.in"
REQ_TXT = PROJECT_ROOT / "requirements.txt"


def run(cmd, check=True):
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


def ensure_venv():
    if VENV_DIR.exists() and (PIP_PATH.exists() or PYTHON_PATH.exists()):
        print(f"[setup_env] Found existing virtual environment at {VENV_DIR}")
        return
    if VENV_DIR.exists() and not (PIP_PATH.exists() or PYTHON_PATH.exists()):
        print("[setup_env] .venv directory exists but appears incomplete. Recreating it...")
        shutil.rmtree(VENV_DIR)
    print(f"[setup_env] Creating virtual environment at {VENV_DIR} using {sys.executable}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def ensure_pip_and_build_tools():
    # Upgrade pip, setuptools, wheel in venv, unless offline mode is requested
    if os.getenv("SETUP_ENV_OFFLINE"):
        print("[setup_env] Offline mode: skipping upgrade of pip/setuptools/wheel")
        return
    run([str(PIP_PATH), "install", "--upgrade", "pip", "setuptools", "wheel"]) 


def install_requirements():
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


def freeze_requirements():
    if os.getenv("SETUP_ENV_OFFLINE") or os.getenv("SETUP_ENV_NO_INSTALL"):
        print("[setup_env] Offline/No-Install mode: skipping freeze to requirements.txt")
        return
    print(f"[setup_env] Freezing installed packages to {REQ_TXT.name}")
    # Use pip freeze to produce a fully pinned requirements.txt
    result = subprocess.run([str(PIP_PATH), "freeze"], capture_output=True, text=True, check=True)
    pinned = result.stdout.strip() + "\n"
    REQ_TXT.write_text(pinned, encoding="utf-8")
    print(f"[setup_env] Wrote {REQ_TXT} with {len(pinned.splitlines())} entries")


def verify_core_requirements_installed():
    # Optionally verify that all top-level requirements from requirements.in are satisfied
    if not REQ_IN.exists():
        return
    print("[setup_env] Verifying that top-level requirements from requirements.in are installed...")
    # Build a set of installed distributions
    result = subprocess.run([str(PIP_PATH), "list", "--format=freeze"], capture_output=True, text=True, check=True)
    installed = {line.split("==")[0].lower() for line in result.stdout.splitlines() if line and not line.startswith("-")}
    missing = []
    for line in REQ_IN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support extras and version specifiers; extract name before any bracket or comparator
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


def main():
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
