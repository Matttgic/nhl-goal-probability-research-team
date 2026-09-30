from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class PythonWorkspaceInput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Complete Python code to execute. The final lines should print the metrics, "
            "validation results, or file paths that the agent needs to inspect."
        ),
    )
    filename: str = Field(
        default="analysis.py",
        description="Safe filename for the script, e.g. train_models.py or backtest.py.",
    )
    timeout_seconds: int = Field(
        default=180,
        ge=5,
        le=900,
        description="Maximum execution time in seconds.",
    )


class PythonWorkspaceTool(BaseTool):
    """Execute Python in a dedicated project workspace.

    Intended for an EPHEMERAL CI runner such as GitHub Actions. It is not a
    hardened security sandbox. The child process receives a deliberately reduced
    environment so model/API secrets are not forwarded to generated Python.
    """

    name: str = "Python ML Workspace"
    description: str = (
        "Write and execute Python 3 code for data engineering, statistics, machine learning, "
        "calibration and backtesting. Use it to test claims with real computation instead of "
        "only describing code. Files are persisted under ./workspace for later agents."
    )
    args_schema: Type[BaseModel] = PythonWorkspaceInput

    def _run(self, code: str, filename: str = "analysis.py", timeout_seconds: int = 180) -> str:
        safe_name = Path(filename).name
        if not safe_name.endswith(".py"):
            safe_name += ".py"

        root = Path.cwd() / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        script = root / safe_name
        script.write_text(code, encoding="utf-8")

        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(Path.home())),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
        }

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=child_env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return (
                f"TIMEOUT after {timeout_seconds}s. Partial stdout:\n{exc.stdout or ''}\n"
                f"Partial stderr:\n{exc.stderr or ''}"
            )

        return (
            f"exit_code={result.returncode}\n"
            f"script={script}\n"
            f"--- stdout ---\n{result.stdout[-12000:]}\n"
            f"--- stderr ---\n{result.stderr[-12000:]}"
        )
