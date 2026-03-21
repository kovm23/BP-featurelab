"""MachineLearningPipeline – central state object for the ML workflow."""
import logging
import os
import pickle

import pandas as pd

from config import CHECKPOINT_FOLDER
from pipeline.feature_discovery import discover_features
from pipeline.feature_extraction import extract_features_async
from pipeline.ml_training import train_model, predict_batch

logger = logging.getLogger(__name__)

_STATE_FILE = os.path.join(CHECKPOINT_FOLDER, "pipeline_state.pkl")


class MachineLearningPipeline:
    """Holds all state for the five-phase ML pipeline."""

    PIPELINE_STATE_FILE = _STATE_FILE

    def __init__(self):
        self.feature_spec: dict = {}
        self.target_variable: str = ""
        # Phase 2 outputs
        self.training_X: pd.DataFrame | None = None
        self.training_Y: pd.Series | None = None
        self.training_Y_df: pd.DataFrame | None = None
        self.training_Y_column: str = ""
        # Phase 3 outputs
        self.model = None
        self.rules: list[str] = []
        self.mse: float | None = None
        self.is_trained: bool = False
        # Phase 4 outputs
        self.testing_X: pd.DataFrame | None = None
        # Internal
        self._training_columns: list[str] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist pipeline state to disk."""
        try:
            with open(self.PIPELINE_STATE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "feature_spec": self.feature_spec,
                        "target_variable": self.target_variable,
                        "training_X": self.training_X,
                        "training_Y_df": self.training_Y_df,
                        "training_Y": self.training_Y,
                        "training_Y_column": self.training_Y_column,
                        "model": self.model,
                        "rules": self.rules,
                        "mse": self.mse,
                        "is_trained": self.is_trained,
                        "testing_X": self.testing_X,
                        "_training_columns": self._training_columns,
                    },
                    f,
                )
            logger.info("Pipeline state saved to disk.")
        except Exception as e:
            logger.warning("Cannot save pipeline state: %s", e)

    def load_state(self) -> bool:
        """Load pipeline state from disk if it exists. Returns True on success."""
        if not os.path.exists(self.PIPELINE_STATE_FILE):
            return False
        try:
            with open(self.PIPELINE_STATE_FILE, "rb") as f:
                state = pickle.load(f)
            self.feature_spec = state.get("feature_spec", {})
            self.target_variable = state.get("target_variable", "")
            self.training_X = state.get("training_X")
            self.training_Y_df = state.get("training_Y_df")
            self.training_Y = state.get("training_Y")
            self.training_Y_column = state.get("training_Y_column", "")
            self.model = state.get("model")
            self.rules = state.get("rules", [])
            self.mse = state.get("mse")
            self.is_trained = state.get("is_trained", False)
            self.testing_X = state.get("testing_X")
            self._training_columns = state.get("_training_columns", [])
            logger.info("Pipeline state loaded from disk.")
            return True
        except Exception as e:
            logger.warning("Cannot load pipeline state: %s", e)
            return False

    # ------------------------------------------------------------------
    # Phase delegates
    # ------------------------------------------------------------------

    def discover_features(self, media_paths, target_variable, model_name, labels_df=None):
        """Phase 1: Feature discovery."""
        return discover_features(self, media_paths, target_variable, model_name, labels_df)

    def extract_features_async(
        self, media_files, feature_spec, job_id, model_name, dataset_type,
        csv_path=None, labels_df=None
    ):
        """Phase 2/4: Async feature extraction (called from a background thread)."""
        return extract_features_async(
            self, media_files, feature_spec, job_id, model_name,
            dataset_type, csv_path, labels_df
        )

    def train_model(self, target_column: str) -> dict:
        """Phase 3: Train RuleKit model."""
        return train_model(self, target_column)

    def predict_batch(self, testing_Y_df=None) -> dict:
        """Phase 5: Batch prediction."""
        return predict_batch(self, testing_Y_df)
