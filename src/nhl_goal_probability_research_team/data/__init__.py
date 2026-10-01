"""NHL data ingestion and leakage-safe feature engineering."""

from .sportsdataverse_features import build_goal_scorer_dataset, load_sportsdataverse_frames

__all__ = ["build_goal_scorer_dataset", "load_sportsdataverse_frames"]
