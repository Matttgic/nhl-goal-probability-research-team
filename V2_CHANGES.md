# V2 changes

1. Embedding-dependent RAG tools removed from the initial run path.
2. GitHub token is no longer required merely to start the crew.
3. Crew process changed from sequential to hierarchical with a custom manager.
4. Seven worker roles can use different model families through environment variables.
5. PythonWorkspaceTool added for real code execution on ephemeral runners.
6. GitHub Actions runner + artifact collection added.
7. Backtest methodology corrected: no invented fixed odds for NHL goal-scorer props.
8. Research outputs are written to `reports/`; executed scripts/data are stored in `workspace/`.
