"""NHL data ingestion and leakage-safe feature engineering."""

from .sportsdataverse_features import build_goal_scorer_dataset, load_sportsdataverse_frames
from .shot_context import build_shot_context_features
from .pick_evaluation import evaluate_picks

__all__ = ["build_goal_scorer_dataset", "load_sportsdataverse_frames", "build_shot_context_features", "evaluate_picks"]
