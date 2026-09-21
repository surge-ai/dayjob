# harbor-agents

The Harbor adapter for DAYJOB. `HarborInstalledAgent` runs the
[OpenHands SDK](https://github.com/OpenHands/software-agent-sdk) agent loop
from the sibling `agent-harness` package inside the task container. It
classifies API failures so Harbor can retry specific errors, for example
with `--max-retries 3 --retry-include ApiRateLimitError`.

## Installation and tools

The adapter uses the runner at `/app/openhands-runner` when it is already
installed in the image. Otherwise, it uploads the local `agent-harness`
source checkout and installs its locked dependencies during setup. This
requires PyPI access and adds setup time to each trial.

Every run includes terminal, file-editor, and task-tracker tools. Images
without `OH_*` settings use `/app` as the workspace and the image's default
user. The adapter does not attempt to start an MCP server in this mode.

Images can supply these optional settings:

- `OH_MCP_URL` — MCP server address, such as `http://localhost:8000/mcp`
  or `http://mcp:8000/mcp` for a separate container.
- `OH_RUNNER_USER` — user account to run the agent under. The image must
  provide the account and set the appropriate file permissions.

With `OH_*` settings present, the MCP URL defaults to
`http://localhost:8000/mcp`. The adapter waits for a remote server. For a
local server, it checks for `/app/scripts/start.sh` and starts the proxy if
needed; without that launcher, it uses local tools only. MCP mode adds the
server's tools and uses `/workdir` as the workspace.

`DAYJOB_TOOL_SETS` optionally selects tool sets when the adapter starts a
local MCP proxy. It passes the space-separated names to `--tool-sets`.

To bake the DAYJOB runner into a dataset, use
`harbor_agents.docker.stage_openhands_runner(build_context_dir)` on a docker
build context that already has a Dockerfile.

If a task directory contains a `system_prompt.md` next to `task.toml`, it is
used verbatim as the system prompt; otherwise the OpenHands SDK default
prompt is used.

## Usage

```bash
harbor run -p ./dist/tasks/some_task \
  --agent-import-path harbor_agents:HarborInstalledAgent \
  -m anthropic/claude-opus-4-8
```

LLM credentials are forwarded from the host environment
(`ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, etc.). Extra
LLM kwargs can be passed with harbor's `--ak`, e.g.
`--ak reasoning_effort=high` or `--ak log_completions=true`.

The runner filters the forwarded LLM credential variables from tool subprocess
environments while retaining them for model requests. This filtering does not
cover the SDK's tmux terminal backend. The DAYJOB task images do not install tmux;
custom images with tmux do not receive this protection.
