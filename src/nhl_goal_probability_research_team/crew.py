from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ArxivPaperTool, ExaSearchTool, FileReadTool, JinaScrapeWebsiteTool, ScrapeWebsiteTool

from nhl_goal_probability_research_team.tools import PythonWorkspaceTool


def model_from_env(name: str, fallback: str = "openai/gpt-5.6-luna") -> LLM:
    """Create an LLM from an environment variable.

    The value must use a CrewAI/LiteLLM provider/model identifier. This keeps the
    project multi-model ready without hard-coding API-only model names that may
    differ between providers/accounts.
    """
    return LLM(model=os.getenv(name, fallback), temperature=0.1)


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
            max_iter=20,
            llm=model_from_env("RESEARCHER_MODEL"),
        )

    @agent
    def sports_data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["sports_data_engineer"],
            tools=[FileReadTool(), ExaSearchTool(), PythonWorkspaceTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=model_from_env("DATA_ENGINEER_MODEL"),
        )

    @agent
    def sports_betting_ml_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["sports_betting_ml_engineer"],
            tools=[FileReadTool(), ArxivPaperTool(), PythonWorkspaceTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=model_from_env("ML_SCIENTIST_MODEL"),
        )

    @agent
    def ml_methodology_critic_and_bias_detective(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_methodology_critic_and_bias_detective"],
            tools=[ExaSearchTool(), ArxivPaperTool(), FileReadTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=model_from_env("CRITIC_MODEL"),
        )

    @agent
    def quantitative_backtesting_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quantitative_backtesting_specialist"],
            tools=[FileReadTool(), ArxivPaperTool(), PythonWorkspaceTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=model_from_env("BACKTESTER_MODEL"),
        )

    @agent
    def ml_platform_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_platform_developer"],
            tools=[FileReadTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=model_from_env("DEVELOPER_MODEL"),
        )

    @agent
    def evidence_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["evidence_synthesizer"],
            tools=[FileReadTool(), ExaSearchTool()],
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=model_from_env("SYNTHESIZER_MODEL"),
        )

    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["research_supervisor_and_final_arbiter"],
            allow_delegation=True,
            max_iter=35,
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

        return Crew(
            agents=workers,
            tasks=self.tasks,
            manager_agent=self.manager(),
            process=Process.hierarchical,
            planning=True,
            verbose=True,
        )
