# agent-harness

The DAYJOB agent loop, built on the OpenHands SDK. It runs the model with
terminal, file-editor, and task-tracker tools, plus any configured MCP tools,
and records the trial as an ATIF trajectory.

Responses requests default to `reasoning_summary=auto`; pass
`--ak reasoning_summary=none` to disable summaries.

## Development

```bash
uv run --extra test pytest tests/ -q
```
