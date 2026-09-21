"""Tests for the Docker build-context staging helpers."""

from pathlib import Path

import pytest

from harbor_agents.docker import (
    dockerfile_snippet,
    openhands_runner_path,
    package_root_path,
    stage_openhands_runner,
)


def test_runner_path_points_at_shipped_script():
    path = openhands_runner_path()
    assert path.is_file()
    assert path.name == "runner.py"


def test_package_root_carries_project_and_lockfile():
    root = package_root_path()
    assert "openhands-sdk==" in (root / "pyproject.toml").read_text()
    assert (root / "uv.lock").is_file()


def test_dockerfile_snippet_syncs_runner_venv():
    snippet = dockerfile_snippet()
    assert "COPY openhands-runner/ /app/openhands-runner/" in snippet
    assert "uv sync --no-config --frozen" in snippet
    assert "--project /app/openhands-runner" in snippet


def test_stage_requires_a_dockerfile(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        stage_openhands_runner(tmp_path)


def test_stage_copies_package_and_extends_dockerfile(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.13-slim\n")

    stage_openhands_runner(tmp_path)

    runner_dir = tmp_path / "openhands-runner"
    root = package_root_path()
    assert (
        runner_dir / "pyproject.toml"
    ).read_text() == (root / "pyproject.toml").read_text()
    assert (runner_dir / "uv.lock").read_text() == (root / "uv.lock").read_text()
    staged_runner = runner_dir / "src" / "agent_harness" / "runner.py"
    assert staged_runner.read_text() == openhands_runner_path().read_text()
    assert not list(runner_dir.rglob("__pycache__"))

    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM python:3.13-slim")
    assert dockerfile_snippet() in dockerfile


def test_stage_is_idempotent(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.13-slim\n")

    stage_openhands_runner(tmp_path)
    once = (tmp_path / "Dockerfile").read_text()
    stage_openhands_runner(tmp_path)
    twice = (tmp_path / "Dockerfile").read_text()

    assert once == twice
    assert twice.count("COPY openhands-runner/") == 1
