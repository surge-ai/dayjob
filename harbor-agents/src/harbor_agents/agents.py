"""Harbor agents that run the OpenHands SDK agent loop.

- ``HarborInstalledAgent`` — runs the loop inside the task container.
  Images delivered with the runner baked in
  (``harbor_agents.docker.stage_openhands_runner``) are used as-is;
  otherwise the agent self-installs at setup time by uploading the sibling
  ``agent-harness`` source checkout into the container and ``uv sync``-ing
  it there (the container needs network access to PyPI).

Delivery decisions arrive as ``OH_*`` env vars baked into the image:
``OH_MCP_URL`` (proxy location) and ``OH_RUNNER_USER`` (non-root user for
the loop); both are optional. If the task directory has a
``system_prompt.md``, it is used verbatim as the system prompt.

Usage:
    harbor run -p ./dist/tasks/some_task \
        --agent-import-path harbor_agents:HarborInstalledAgent \
        -m anthropic/claude-opus-4-8
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tarfile
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from types import SimpleNamespace

from agent_harness import LLM_CREDENTIAL_ENV_VARS
from harbor.agents.installed.base import (
    ApiInternalServerError,
    ApiOverloadedError,
    ApiRateLimitError,
    BaseInstalledAgent,
    ContextWindowExceededError,
    ErrorPattern,
    NetworkConnectionError,
    UnknownApiError,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

MAX_TURNS = 1000

# Optional runtime settings supplied by the task image.
FACT_MCP_URL = "OH_MCP_URL"
FACT_RUNNER_USER = "OH_RUNNER_USER"
FACT_VARS = (
    FACT_MCP_URL,
    FACT_RUNNER_USER,
)

# Where the bash tool operates: the agent's own deliverable space.
BASH_WORKSPACE_DIR = "/workdir"

# In-container locations (installed agent). The agent-harness package is
# staged as a uv project and synced into an isolated venv; the runner is
# exec'd as an installed module.
RUNNER_DIR = "/app/openhands-runner"
RUNNER_PATH = f"{RUNNER_DIR}/src/agent_harness/runner.py"
RUNNER_PYTHON = f"{RUNNER_DIR}/.venv/bin/python"
RUNNER_MODULE = "agent_harness.runner"
# Fallback proxy launch when nothing else in the container is serving it.
PROXY_START_CMD = (
    "/app/scripts/start.sh --method http --port 8000"
    " ${DAYJOB_TOOL_SETS:+--tool-sets $DAYJOB_TOOL_SETS}"
)
PROXY_LAUNCHER_PATH = "/app/scripts/start.sh"
MCP_URL = "http://localhost:8000/mcp"
LOCAL_WORKSPACE_DIR = "/app"

HEALTH_URL = "http://localhost:8000/health"


def _health_url_for(mcp_url: str) -> str:
    """The proxy serves /health next to /mcp on the same host/port."""
    if mcp_url.endswith("/mcp"):
        return mcp_url[: -len("/mcp")] + "/health"
    return HEALTH_URL


def _is_local_url(url: str) -> bool:
    return "//localhost" in url or "//127.0.0.1" in url


REMOTE_LOG_DIR = "/logs/agent"
REMOTE_CONFIG_PATH = "/tmp/openhands_run_config.json"
REMOTE_TRAJECTORY_PATH = f"{REMOTE_LOG_DIR}/trajectory.json"
REMOTE_RUNNER_LOG_PATH = f"{REMOTE_LOG_DIR}/run-openhands.log"
REMOTE_PROXY_LOG_PATH = f"{REMOTE_LOG_DIR}/mcp-proxy.log"

REMOTE_RUNNER_TARBALL = "/tmp/openhands-runner.tgz"

# Timeouts (seconds). AGENT_EXEC_TIMEOUT is a generous ceiling; the
# effective limit is harbor's [agent] timeout_sec.
PROBE_TIMEOUT = 15
HEALTH_TIMEOUT = 180
AGENT_EXEC_TIMEOUT = 24 * 60 * 60
# Self-install downloads openhands-sdk and its dependency tree from PyPI.
INSTALL_TIMEOUT = 1200

# LLM credentials forwarded from the host into the container for the runner.
FORWARDED_ENV_VARS = LLM_CREDENTIAL_ENV_VARS

# --ak kwargs forwarded to the runner's LLM config (see runner.py for how
# each is applied).
FORWARDED_LLM_KWARGS = (
    "api_mode",
    "max_output_tokens",
    "reasoning_effort",
    "reasoning_summary",
    "model_canonical_name",  # capability-lookup override for proxied model ids
    "thought_display",  # MAI: encrypted reasoning in message content
    "force_string_serializer",  # MAI: plain-string message serialization
    "log_completions",  # dump per-call request payloads into the trial logs
)


def _forwarded_env() -> dict[str, str]:
    return {k: v for k in FORWARDED_ENV_VARS if (v := os.environ.get(k))}


async def _has_installed_runner(environment: BaseEnvironment) -> bool:
    """Whether the image carries the delivery-time runner + venv."""
    probe = await environment.exec(
        f"test -f {shlex.quote(RUNNER_PATH)} && test -x {shlex.quote(RUNNER_PYTHON)}",
        timeout_sec=PROBE_TIMEOUT,
    )
    return probe.return_code == 0


def _pack_runner_project(tarball_path: Path) -> None:
    """Tar the agent-harness source checkout for upload into the container.

    Contents match what ``stage_openhands_runner`` copies into a build
    context: pyproject.toml, uv.lock and src/agent_harness/.
    """
    from harbor_agents.docker import package_root_path

    root = package_root_path()

    def _skip_caches(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(info.name).name
        if name == "__pycache__" or name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(root / "pyproject.toml", arcname="pyproject.toml")
        tar.add(root / "uv.lock", arcname="uv.lock")
        tar.add(
            root / "src" / "agent_harness",
            arcname="src/agent_harness",
            filter=_skip_caches,
        )


async def _read_image_facts(environment: BaseEnvironment) -> dict[str, str]:
    """Read the OH_* image facts baked in at delivery time."""
    result = await environment.exec(
        "printenv | grep '^OH_' || true", timeout_sec=PROBE_TIMEOUT
    )
    env = dict(
        line.split("=", 1)
        for line in (result.stdout or "").splitlines()
        if "=" in line
    )
    return {k: v for k, v in env.items() if k in FACT_VARS}


def _pop_llm_kwargs(kwargs: dict) -> dict:
    """Extract LLM kwargs and discard the obsolete ``bash_tool`` switch."""
    llm_kwargs = {k: kwargs.pop(k) for k in FORWARDED_LLM_KWARGS if k in kwargs}
    kwargs.pop("bash_tool", None)
    return llm_kwargs


def _read_system_prompt(environment: BaseEnvironment) -> str | None:
    """A system_prompt.md next to task.toml is used verbatim."""
    sp_path = Path(environment.environment_dir).parent / "system_prompt.md"
    return sp_path.read_text() if sp_path.is_file() else None


def _run_config(
    *,
    instruction: str,
    system_prompt: str | None,
    model: str | None,
    mcp_url: str | None,
    workspace_dir: str | None,
    llm_kwargs: dict,
    log_dir: str,
) -> dict:
    """The config JSON contract with runner.py."""
    return {
        "instruction": instruction,
        "systemPrompt": system_prompt,
        "model": model,
        "mcpUrl": mcp_url,
        "workspaceDir": workspace_dir,
        "maxTurns": MAX_TURNS,
        "llmKwargs": llm_kwargs,
        "logDir": log_dir,
    }


def _run_summary(traj: dict) -> dict:
    """The runner's verdict (``TrajectoryExtras``), carried under the ATIF
    trajectory's ``extra`` field."""
    extra = traj.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError(
            "trajectory.json has no run summary under 'extra'; "
            "the container's agent-harness runner predates the ATIF format"
        )
    return extra


def _n_tool_calls(traj: dict) -> int:
    return sum(1 for s in traj.get("steps") or [] if s.get("tool_calls"))


def _apply_trajectory(traj: dict, context: AgentContext) -> None:
    summary = _run_summary(traj)
    metrics = traj.get("final_metrics") or {}
    context.n_input_tokens = metrics.get("total_prompt_tokens")
    context.n_output_tokens = metrics.get("total_completion_tokens")
    context.n_cache_tokens = metrics.get("total_cached_tokens")
    # The runner reports 0.0 when litellm has no pricing for the model;
    # surface that as unknown so cost reporting falls back to dashboards.
    context.cost_usd = metrics.get("total_cost_usd") or None
    agent_block = traj.get("agent") or {}
    context.metadata = {
        "agent_id": agent_block.get("name"),
        "model": agent_block.get("model_name"),
        "reasoning_tokens": summary.get("reasoning_tokens"),
        "n_tool_calls": _n_tool_calls(traj),
        "stopped_reason": summary.get("stopped_reason"),
        "final_output_chars": len(summary.get("final_output") or ""),
        "error_message": summary.get("error_message"),
    }


def _loop_error_detail(traj: dict, summary: dict) -> str:
    return (
        "OpenHands agent loop ended with an error after "
        f"{_n_tool_calls(traj)} tool call(s): "
        f"{summary.get('error_message') or '<no error message>'}"
    )


class HarborInstalledAgent(BaseInstalledAgent):
    """Runs the OpenHands SDK loop inside the task container.

    Failed runs are classified into typed errors so harbor retry policy can
    target them (e.g. ``--retry-include ApiRateLimitError``).
    """

    # The runner logs raw litellm exception names; classify those too.
    ERROR_PATTERNS = BaseInstalledAgent.ERROR_PATTERNS + [
        ErrorPattern(r"litellm\.RateLimitError", ApiRateLimitError),
        ErrorPattern(r"litellm\.InternalServerError", ApiInternalServerError),
        ErrorPattern(r"litellm\.ServiceUnavailableError", ApiOverloadedError),
        ErrorPattern(
            r"litellm\.(Timeout|APIConnectionError)", NetworkConnectionError
        ),
        ErrorPattern(
            r"ContextWindowExceededError", ContextWindowExceededError
        ),
        ErrorPattern(r"litellm\.\w*Error", UnknownApiError),
    ]

    def __init__(self, *args, **kwargs) -> None:
        self.llm_kwargs = _pop_llm_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "harbor-installed-agent"

    def version(self) -> str | None:
        if self._version:
            return self._version
        try:
            return _pkg_version("harbor-agents")
        except PackageNotFoundError:
            return None

    async def install(self, environment: BaseEnvironment) -> None:
        """Use the delivery-time runner when baked in; otherwise self-install.

        Self-install uploads the sibling agent-harness source checkout into
        the container and syncs its pinned lockfile into an isolated venv —
        the runtime equivalent of ``stage_openhands_runner``'s build step.
        """
        if await _has_installed_runner(environment):
            return

        self.logger.info(
            "no baked runner at %s; self-installing agent-harness", RUNNER_DIR
        )
        tarball = (self.logs_dir / "openhands-runner.tgz").resolve()
        tarball.parent.mkdir(parents=True, exist_ok=True)
        _pack_runner_project(tarball)
        await environment.upload_file(str(tarball), REMOTE_RUNNER_TARBALL)

        install_cmd = " && ".join([
            f"rm -rf {RUNNER_DIR}",
            f"mkdir -p {RUNNER_DIR}",
            f"tar -xzf {REMOTE_RUNNER_TARBALL} -C {RUNNER_DIR}",
            f"rm -f {REMOTE_RUNNER_TARBALL}",
            # Bootstrap uv when the base image doesn't ship it.
            "(command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv)",
            # --no-config keeps any container-local uv configuration from
            # rejecting the runner's pinned lockfile.
            f"uv sync --no-config --frozen --no-dev --project {RUNNER_DIR}",
        ])
        result = await environment.exec(install_cmd, timeout_sec=INSTALL_TIMEOUT)
        if result.return_code != 0:
            raise RuntimeError(
                "failed to self-install the OpenHands runner "
                f"(rc={result.return_code}):\n"
                f"{(result.stderr or result.stdout or '').strip()[-2000:]}"
            )
        if not await _has_installed_runner(environment):
            raise RuntimeError(
                f"self-install completed but no runner venv at {RUNNER_PYTHON}"
            )

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)  # -> install()

        facts = await _read_image_facts(environment)
        self.logger.info("image facts: %s", facts)
        self._run_as_user = facts.get(FACT_RUNNER_USER) or None
        # Without OH_* settings, use local tools in /app. Skip launcher
        # detection: the workspace may contain an unrelated scripts/start.sh.
        if not facts:
            self._use_mcp = False
            self.logger.info("no image facts; runner will use local tools")
            return

        self._mcp_url = facts.get(FACT_MCP_URL) or MCP_URL
        self._health_url = _health_url_for(self._mcp_url)

        if not _is_local_url(self._mcp_url):
            # The proxy lives in another container (compose sidecar); just
            # wait for it.
            self._use_mcp = True
            await self._wait_for_proxy(environment)
            return

        # No proxy launcher -> plain filesystem task; the runner uses local
        # tools over /app instead of MCP tools.
        launcher = await environment.exec(
            f"test -x {shlex.quote(PROXY_LAUNCHER_PATH)}", timeout_sec=PROBE_TIMEOUT
        )
        self._use_mcp = launcher.return_code == 0
        if not self._use_mcp:
            self.logger.info(
                "no MCP proxy launcher at %s; runner will use local tools",
                PROXY_LAUNCHER_PATH,
            )
            return

        if not await self._proxy_healthy(environment):
            # If the proxy is the container command (pid 1) it may just not
            # be up yet; launching a second one would race it for the port.
            # Only launch when nothing else is starting it.
            pid1 = await environment.exec(
                "tr '\\0' ' ' </proc/1/cmdline", timeout_sec=PROBE_TIMEOUT
            )
            if "start.sh" in (pid1.stdout or "") or "mcp" in (pid1.stdout or ""):
                self.logger.info(
                    "MCP proxy is the container command; waiting for it to come up"
                )
                await self._wait_for_proxy(environment)
                return
            self.logger.info("Launching MCP proxy in-container")
            launch = (
                f"mkdir -p {REMOTE_LOG_DIR} && "
                f"nohup {PROXY_START_CMD} </dev/null "
                f">>{REMOTE_PROXY_LOG_PATH} 2>&1 &"
            )
            result = await environment.exec(launch, timeout_sec=PROBE_TIMEOUT)
            if result.return_code != 0:
                raise RuntimeError(
                    f"failed to launch MCP proxy (rc={result.return_code}): "
                    f"{(result.stderr or '').strip()}"
                )
        await self._wait_for_proxy(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        use_mcp = getattr(self, "_use_mcp", True)
        if use_mcp:
            # The bash tool's cwd is the agent's deliverable space.
            workspace_dir = BASH_WORKSPACE_DIR
        else:
            workspace_dir = LOCAL_WORKSPACE_DIR
        config = _run_config(
            instruction=instruction,
            system_prompt=_read_system_prompt(environment),
            model=self.model_name,
            mcp_url=getattr(self, "_mcp_url", MCP_URL) if use_mcp else None,
            workspace_dir=workspace_dir,
            llm_kwargs=self.llm_kwargs,
            log_dir=REMOTE_LOG_DIR,
        )

        config_path = (self.logs_dir / "openhands_run_config.json").resolve()
        config_path.write_text(json.dumps(config, indent=2))
        await environment.upload_file(str(config_path), REMOTE_CONFIG_PATH)

        self.logger.info(
            "Running OpenHands runner in-container (model=%s, user=%s)",
            self.model_name,
            self._run_as_user or "<default>",
        )

        # Root pre-step: make the log dir writable (and the uploaded config
        # readable) by the runner user.
        prep = f"mkdir -p {REMOTE_LOG_DIR}"
        if self._run_as_user:
            prep += (
                f" && chown -R {self._run_as_user}: {REMOTE_LOG_DIR}"
                f" && chmod 644 {shlex.quote(REMOTE_CONFIG_PATH)}"
            )
        await self.exec_as_root(environment, prep, timeout_sec=PROBE_TIMEOUT)

        # tee keeps the runner log on disk while the captured output feeds
        # error classification. The trajectory goes to its own file — stdout
        # carries the SDK transcript.
        runner_cmd = (
            f"{shlex.quote(RUNNER_PYTHON)} -m {RUNNER_MODULE} "
            f"{shlex.quote(REMOTE_CONFIG_PATH)} {shlex.quote(REMOTE_TRAJECTORY_PATH)} "
            f"2>&1 | tee {REMOTE_RUNNER_LOG_PATH}"
        )
        try:
            await self._exec(
                environment,
                runner_cmd,
                user=self._run_as_user,
                env=_forwarded_env(),
                timeout_sec=AGENT_EXEC_TIMEOUT,
            )
        finally:
            runner_log = await self._download_quiet(
                environment, REMOTE_RUNNER_LOG_PATH, "run-openhands.log"
            )
            trajectory = await self._download_quiet(
                environment, REMOTE_TRAJECTORY_PATH, "trajectory.json"
            )

        if trajectory is None or trajectory.stat().st_size == 0:
            raise RuntimeError(
                "OpenHands runner produced no trajectory output; "
                f"see {self.logs_dir} for the runner log"
            )

        traj = json.loads(trajectory.read_text())
        _apply_trajectory(traj, context)

        # The runner exits 0 and records loop errors in the trajectory;
        # classify the log here to raise a typed, retry-targetable error.
        summary = _run_summary(traj)
        if summary.get("stopped_reason") == "error":
            log_text = ""
            if runner_log is not None:
                log_text = runner_log.read_text(errors="replace")
            shim = SimpleNamespace(
                return_code=1,
                stdout=f"{summary.get('error_message') or ''}\n{log_text}",
                stderr=_loop_error_detail(traj, summary),
            )
            raise self._classify_exec_error("openhands agent loop", shim)

    # --- proxy health (exec'd curl inside the container) ---

    async def _proxy_healthy(self, environment: BaseEnvironment) -> bool:
        health_url = getattr(self, "_health_url", HEALTH_URL)
        result = await environment.exec(
            f"curl -fsS -o /dev/null --max-time 5 {shlex.quote(health_url)}",
            timeout_sec=PROBE_TIMEOUT,
        )
        return result.return_code == 0

    async def _wait_for_proxy(self, environment: BaseEnvironment) -> None:
        health_url = getattr(self, "_health_url", HEALTH_URL)
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if await self._proxy_healthy(environment):
                self.logger.info("MCP proxy healthy")
                return
            await asyncio.sleep(2)
        tail = await environment.exec(
            f"tail -c 2000 {REMOTE_PROXY_LOG_PATH} 2>/dev/null || true",
            timeout_sec=PROBE_TIMEOUT,
        )
        raise RuntimeError(
            f"MCP proxy at {health_url} never became healthy within {HEALTH_TIMEOUT}s"
            f"\n--- proxy log tail ---\n{(tail.stdout or '').strip() or '<no log>'}"
        )

    async def _download_quiet(
        self, environment: BaseEnvironment, remote_path: str, filename: str
    ) -> Path | None:
        """Download a container file into logs_dir; None if unavailable."""
        target = self.logs_dir / filename
        try:
            await environment.download_file(remote_path, target)
        except Exception as e:
            self.logger.warning("could not download %s: %s", remote_path, e)
            return None
        return target if target.exists() else None
