# DAYJOB Benchmarks, by Surge AI

DAYJOB is Surge AI's benchmark family for long-horizon professional agents. DAYJOB is designed around rich knowledge work from everyday enterprise: short, realistic workplace requests that require professional judgment, large working environments that require exploration, and much more for agents to figure out.

This repository contains the cross-domain evaluation harness from [Surge AI](https://surgehq.ai/). Find the tasks on [Hugging Face](https://huggingface.co/surgeai).

Tasks use [Harbor](https://harborframework.com/) format. The harness runs models through the [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk).

## Layout

This repository contains:

- `agent-harness/` — the agent loop used to complete trials.
- `harbor-agents/` — the Harbor adapter that installs and runs the agent in a sandbox.
- `pyproject.toml` + `uv.lock` — host-side environment. Pins `harbor` and pulls in the two packages above as path dependencies.

## Getting the tasks

Find DAYJOB task datasets on [Surge AI on Hugging Face](https://huggingface.co/surgeai).

Follow the dataset's download instructions to obtain the Harbor task files. Each task directory contains `task.toml`, `instruction.md`, `environment/`, and `tests/`. The commands below use `TASKS_DIR` for the local directory containing the task directories you want to run.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.
- Docker with Linux containers, or another Harbor environment that supports agent-phase network allowlists. See the network requirements below.
- API credentials for the model under test and the autograder.

## Running the benchmark

Run the following commands from this harness directory. Install the pinned Harbor version and local harness packages:

```bash
uv sync
```

### 1. Identify the API hosts your agent needs

During task execution, network access is restricted to explicitly allowed model API hosts. Browsing and package downloads are blocked. Harness setup and grading retain network access.

Tasks start with an empty `[agent].allowed_hosts` allowlist in `task.toml`. Set the hostname of your model endpoint, without a URL scheme or path:

```bash
LLM_HOST=api.provider.com
```

The run command below adds this host with `--allow-agent-host`. Repeat the flag for multiple API hosts, or set `[agent].allowed_hosts` in `task.toml`.

Use an environment that supports agent-phase allowlists. Local Docker requires Linux containers and Docker-host support for nftables.

### 2. Configure credentials

Set the path to your downloaded task directories and the autograder's credentials:

```bash
TASKS_DIR=/path/to/downloaded/tasks
export ANTHROPIC_API_KEY=...
```

Also export the credentials for the model under test if it uses a different provider (for example, `OPENAI_API_KEY` or `GEMINI_API_KEY`).

### 3. Run the tasks

Run tasks on a model with `harbor run`, replacing `<litellm-model>` with its model ID. This example uses the leaderboard's **5 attempts per task** and the **Claude Opus 4.8 autograder**.

```bash
uv run harbor run \
  -p "$TASKS_DIR" \
  -a harbor_agents:HarborInstalledAgent \
  -m "<litellm-model>" \
  --allow-agent-host "$LLM_HOST" \
  -k 5 \
  --max-retries 3 --retry-include ApiRateLimitError \
  --ve REWARDKIT_MODEL=anthropic/claude-opus-4-8 \
  --ve ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
```

Pass the judge's credentials explicitly with `--ve`. See Grading below.

## Results

By default, Harbor saves each run to `jobs/<job-name>/` relative to the directory where you run the command. Each trial has its own subdirectory:

- `<trial>/result.json` — the trial's reward, status, and any exception details.
- `<trial>/agent/trajectory.json` — the agent's recorded trajectory.
- `<trial>/agent/run-openhands.log` — the agent runner's output.
- `<trial>/verifier/` — grading output and verifier logs.

## Configuration options

- **Model** — `-m` takes a litellm model id, e.g. `anthropic/claude-opus-4-8`. Its provider key is read from your environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`) and forwarded into the agent container. Supported endpoint overrides are `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, and `GEMINI_API_BASE`.
- **Judge model** — leaderboard runs use `anthropic/claude-opus-4-8`. Set it explicitly with `--ve REWARDKIT_MODEL=anthropic/claude-opus-4-8` as above; this overrides each task's `tests/judge.toml` defaults. Its credentials come from the `--ve` flags.
- **Reasoning effort** — `--ak reasoning_effort=<level>` (e.g. `high`); levels are model-dependent.
- **Attempts per task** — `-k`. Leaderboard runs use 5 attempts per task.
- **Using an LLM proxy** — point the base-URL vars at the proxy: `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` etc. for the model under test, and `--ve ANTHROPIC_BASE_URL=...` for the Opus judge when using an Anthropic-compatible endpoint. **Important**: if the proxy serves the model under an alias, add `--ak model_canonical_name=<real-model-id>` so capability lookup still works.
- **Endpoint mode (Chat Completions vs Responses)** — `--ak api_mode=chat` or `--ak api_mode=responses`. The default (`auto`) resolves from model metadata; override it for proxy aliases or newly released models.

## Grading

Grading uses Claude Opus 4.8 as an agentic LLM judge (OpenCode) running inside the task container. Use the explicit model override in the run command above to match the leaderboard autograder.

Grading dependencies are pinned to OpenCode `1.18.31` and `harbor-rewardkit` `0.1.7`.

The scoring convention:

1. A trial's reward is `verifier_result.rewards.reward` in its `result.json` (0 to 1).
2. A task's score is the mean reward over its 5 attempts. Trials that errored (`exception_info` set) are excluded from the mean.
3. A domain's score is the unweighted mean over that domain's task scores.
