# NHL Goal Probability Research Team — V2

A code-first CrewAI project designed as a **real multi-agent research team**, not seven copies of the same prompt.

## What V2 fixes

- Removes embedding-dependent `CSVSearchTool`, `JSONSearchTool` and `GithubSearchTool` from the first runnable path.
- Uses a **hierarchical CrewAI process** with a dedicated manager that can delegate and request rework.
- Makes each specialist's model configurable independently, so GPT / Claude / Gemini (or other providers) can be mixed in one crew.
- Adds `PythonWorkspaceTool`, allowing the Data Engineer, ML Scientist and Backtester to **execute Python** and inspect real results on an ephemeral GitHub Actions runner.
- Never assumes fake fixed odds for NHL goal-scorer props. Economic results are calculated only from genuine timestamp-valid historical prices.
- Adds GitHub Actions workflows and persistent run artifacts (`reports/`, `workspace/`).

## Architecture

```text
                           MANAGER
                              |
       +----------+-----------+-----------+----------+
       |          |           |           |          |
   Researcher  Data Eng.   ML Scientist  Critic  Backtester
                              |           ^          |
                              +-----------+----------+
                                      evidence
                                          |
                                      Developer
                                          |
                                     Synthesizer
```

The manager coordinates the team. The Critic is deliberately adversarial. Python execution is used to settle empirical disagreements whenever possible.

## Important limitation

`PythonWorkspaceTool` is **not a hardened security sandbox**. It is intended to run inside an ephemeral CI machine such as GitHub Actions. The child process receives a reduced environment so provider/API keys are not intentionally forwarded to generated Python. For production-grade isolated execution, migrate this tool to E2B, Daytona, Modal or another dedicated sandbox service.

## GitHub setup

1. In GitHub: **Settings → Secrets and variables → Actions**.
2. Add only the API secrets you actually use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `EXA_API_KEY`).
3. Under **Variables**, set model identifiers for the roles you want to override.
4. Open **Actions → Run NHL multi-agent crew → Run workflow**.
5. When the run ends, download the `nhl-crew-output` artifact containing `reports/` and `workspace/`.

> ChatGPT/Claude consumer subscriptions do not automatically provide API credits. The crew needs API access for whichever models you configure.

## Suggested multi-model mapping

- **Manager**: strongest reasoning model available
- **Researcher**: fast, web/tool-capable model
- **ML Scientist**: strong coding/data model
- **Critic**: a *different model family* from the ML Scientist
- **Backtester**: strong coding/reasoning model
- **Developer**: strongest coding model you can afford
- **Synthesizer**: strong long-context model

The point is diversity of reasoning, not merely seven calls to the same model.

## Local run

```bash
cp .env.example .env
# fill only the keys/models you actually use
uv sync
uv run run_crew
```

## Evidence rules

The crew is instructed to:

- use only pre-game information for historical predictions;
- use temporal/walk-forward validation;
- distinguish executed experiments from proposed code;
- label synthetic fixtures explicitly;
- refuse ROI/CLV claims without real historical market prices;
- reject models with leakage or insufficient evidence;
- keep failed/rejected approaches visible in the final report.
