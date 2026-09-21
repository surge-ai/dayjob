"""Bake the OpenHands runner into a task image.

``stage_openhands_runner()`` copies the runner sources and lockfile into a
Docker build context and appends a ``uv sync --frozen`` build step. This
installs the runner at ``/app/openhands-runner/.venv`` with pinned dependencies.
For images without a baked runner, ``HarborInstalledAgent`` installs it at
setup time instead.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import agent_harness

_AGENT_HARNESS_DIR = Path(agent_harness.__file__).resolve().parent


def openhands_runner_path() -> Path:
    """Path to the runner script staged into task images."""
    return _AGENT_HARNESS_DIR / "runner.py"


def package_root_path() -> Path:
    """Root of the agent-harness package checkout (pyproject + uv.lock).

    Staging requires agent-harness to be importable from a source checkout
    (editable install or PYTHONPATH), not a built wheel.
    """
    root = _AGENT_HARNESS_DIR.parents[1]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise FileNotFoundError(
            f"agent-harness source checkout not found at {root}; staging needs "
            "pyproject.toml and uv.lock (install the package editable)"
        )
    return root


def dockerfile_snippet() -> str:
    """Dockerfile steps that install the runner venv from the staged package.

    Assumes the build context contains ``openhands-runner/`` with
    ``pyproject.toml``, ``uv.lock`` and ``src/agent_harness/``.
    """
    return "\n".join([
        "",
        "# Install the OpenHands runner in a separate virtual environment.",
        "# Install uv if the base image does not provide it.",
        "RUN command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv",
        "# The lockfile pins the SDK, local tools, and their dependencies.",
        "COPY openhands-runner/ /app/openhands-runner/",
        "# Ignore container-local uv configuration when syncing the lockfile.",
        "RUN uv sync --no-config --frozen --no-dev --python 3.13 \\",
        "    --project /app/openhands-runner",
        "",
    ])


def stage_openhands_runner(build_context_dir: Path) -> None:
    """Copy the package into *build_context_dir* and extend its Dockerfile."""
    dockerfile = build_context_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"no Dockerfile in build context: {build_context_dir}")

    root = package_root_path()
    runner_dir = build_context_dir / "openhands-runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "pyproject.toml", runner_dir / "pyproject.toml")
    shutil.copy2(root / "uv.lock", runner_dir / "uv.lock")
    package_dir = runner_dir / "src" / "agent_harness"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copytree(
        root / "src" / "agent_harness",
        package_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    content = dockerfile.read_text()
    if "openhands-runner/" not in content:
        dockerfile.write_text(content.rstrip("\n") + "\n" + dockerfile_snippet())
