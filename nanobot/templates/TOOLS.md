# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace
- `allowedPaths` can grant narrow access to extra project directories while keeping the workspace sandbox enabled

## coding_agent — Direct Code Delegation

- Use `coding_agent` when the task needs real workspace inspection, code edits, or local verification.
- The coding worker prefers a named agent route called `coder`, then `codex`, then falls back to the default model.
- The final tool result should be a short summary of changes, touched files, and verification status.

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
