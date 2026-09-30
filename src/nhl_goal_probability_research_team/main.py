#!/usr/bin/env python
from __future__ import annotations

import os
import sys

from nhl_goal_probability_research_team.crew import NhlGoalProbabilityResearchTeamCrew


DEFAULT_MISSION = (
    "Find and verify the data needed to build an NHL player anytime-goal probability model; "
    "execute a leakage-safe ML and backtesting pipeline where the available data permits it, "
    "and prepare a GitHub-ready research prototype."
)


def inputs() -> dict[str, str]:
    return {"mission": os.getenv("MISSION", DEFAULT_MISSION)}


def run():
    return NhlGoalProbabilityResearchTeamCrew().crew().kickoff(inputs=inputs())


def train():
    try:
        return NhlGoalProbabilityResearchTeamCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs()
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    try:
        return NhlGoalProbabilityResearchTeamCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    try:
        return NhlGoalProbabilityResearchTeamCrew().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs()
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


if __name__ == "__main__":
    run()
