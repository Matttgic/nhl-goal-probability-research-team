from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ArxivPaperTool, ExaSearchTool, FileReadTool, JinaScrapeWebsiteTool, ScrapeWebsiteTool

from nhl_goal_probability_research_team.tools import PythonWorkspaceTool


FREE_MODEL = "openrouter/openrouter/free"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cheap_test() -> bool:
    return _truthy(os.getenv("CHEAP_TEST"))


def _project_file_reader() -> FileReadTool:
    """Allow reads only inside the checked-out repository.

    CrewAI's FileReadTool otherwise chooses its own allowed base directory, which
    can reject legitimate paths such as ./workspace on GitHub Actions runners.
    """
    return FileReadTool(base_dir=str(Path.cwd()))


def model_from_env(name: str, fallback: str = FREE_MODEL) -> LLM:
    """Create an LLM from an environment variable.

    Blank GitHub variables are treated as unset. The zero-cost default routes through
    OpenRouter's free-model router; production runs can override each role separately.
    """
    model = (os.getenv(name) or "").strip() or fallback
    return LLM(model=model, temperature=0.1)


def worker_max_iter() -> int:
    if _cheap_test():
        return 3
    return int(os.getenv("WORKER_MAX_ITER", "20"))


def manager_max_iter() -> int:
    if _cheap_test():
        return 6
    return int(os.getenv("MANAGER_MAX_ITER", "35"))


@CrewBase
class NhlGoalProbabilityResearchTeamCrew:
    """Evidence-first, multi-agent NHL goal-probability research crew."""

    @agent
    def sports_data_intelligence_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["sports_data_intelligence_researcher"],
            tools=[ExaSearchTool(), ScrapeWebsiteTool(), JinaScrapeWebsiteTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("RESEARCHER_MODEL"),
        )

    @agent
    def sports_data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["sports_data_engineer"],
            tools=[_project_file_reader(), ExaSearchTool(), PythonWorkspaceTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("DATA_ENGINEER_MODEL"),
        )

    @agent
    def sports_betting_ml_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["sports_betting_ml_engineer"],
            tools=[_project_file_reader(), ArxivPaperTool(), PythonWorkspaceTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("ML_SCIENTIST_MODEL"),
        )

    @agent
    def ml_methodology_critic_and_bias_detective(self) -> Agent:
        tools = [ExaSearchTool(), ArxivPaperTool()] if _cheap_test() else [
            ExaSearchTool(),
            ArxivPaperTool(),
            _project_file_reader(),
        ]
        return Agent(
            config=self.agents_config["ml_methodology_critic_and_bias_detective"],
            tools=tools,
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("CRITIC_MODEL"),
        )

    @agent
    def quantitative_backtesting_specialist(self) -> Agent:
        # The free router is intentionally kept tool-free here. If the critic does not
        # PASS the experiment, the backtester should only report that validation is
        # blocked. Full runs retain file/Python tools for real walk-forward testing.
        tools = [] if _cheap_test() else [
            _project_file_reader(),
            ArxivPaperTool(),
            PythonWorkspaceTool(),
        ]
        return Agent(
            config=self.agents_config["quantitative_backtesting_specialist"],
            tools=tools,
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("BACKTESTER_MODEL"),
        )

    @agent
    def ml_platform_developer(self) -> Agent:
        tools = [] if _cheap_test() else [_project_file_reader()]
        return Agent(
            config=self.agents_config["ml_platform_developer"],
            tools=tools,
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("DEVELOPER_MODEL"),
        )

    @agent
    def evidence_synthesizer(self) -> Agent:
        tools = [] if _cheap_test() else [_project_file_reader(), ExaSearchTool()]
        return Agent(
            config=self.agents_config["evidence_synthesizer"],
            tools=tools,
            inject_date=True,
            allow_delegation=False,
            max_iter=worker_max_iter(),
            llm=model_from_env("SYNTHESIZER_MODEL"),
        )

    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["research_supervisor_and_final_arbiter"],
            allow_delegation=True,
            max_iter=manager_max_iter(),
            llm=model_from_env("MANAGER_MODEL"),
        )

    @task
    def research_nhl_data_sources(self) -> Task:
        return Task(config=self.tasks_config["research_nhl_data_sources"])

    @task
    def engineer_nhl_datasets(self) -> Task:
        return Task(config=self.tasks_config["engineer_nhl_datasets"])

    @task
    def train_goal_probability_models(self) -> Task:
        return Task(config=self.tasks_config["train_goal_probability_models"])

    @task
    def critique_model_results(self) -> Task:
        return Task(config=self.tasks_config["critique_model_results"])

    @task
    def run_walk_forward_backtests(self) -> Task:
        return Task(config=self.tasks_config["run_walk_forward_backtests"])

    @task
    def build_developer_project_scaffold(self) -> Task:
        return Task(config=self.tasks_config["build_developer_project_scaffold"])

    @task
    def final_synthesis_report(self) -> Task:
        return Task(config=self.tasks_config["final_synthesis_report"])

    @crew
    def crew(self) -> Crew:
        workers = [
            self.sports_data_intelligence_researcher(),
            self.sports_data_engineer(),
            self.sports_betting_ml_engineer(),
            self.ml_methodology_critic_and_bias_detective(),
            self.quantitative_backtesting_specialist(),
            self.ml_platform_developer(),
            self.evidence_synthesizer(),
        ]

        if _cheap_test():
            # OpenRouter's free router can choose models with inconsistent native
            # tool-calling behavior. Sequential smoke mode still passes every prior
            # task's evidence through explicit context while avoiding manager/delegation
            # tool calls. Production mode below remains truly hierarchical.
            return Crew(
                agents=workers,
                tasks=self.tasks,
                process=Process.sequential,
                planning=False,
                verbose=True,
            )

        return Crew(
            agents=workers,
            tasks=self.tasks,
            manager_agent=self.manager(),
            process=Process.hierarchical,
            planning=True,
            verbose=True,
        )
