"""MachineLearningPipeline – central state object for the ML workflow."""
import json
import logging
import os
import pickle

import pandas as pd

from config import CHECKPOINT_FOLDER
from pipeline.feature_discovery import discover_features
from pipeline.feature_extraction import extract_features_async
from pipeline.ml_training import train_model, predict_batch

logger = logging.getLogger(__name__)

_STATE_JSON = os.path.join(CHECKPOINT_FOLDER, "pipeline_state.json")
_MODEL_PKL = os.path.join(CHECKPOINT_FOLDER, "model.pkl")
_TRAINING_X_CSV = os.path.join(CHECKPOINT_FOLDER, "training_X.csv")
_TRAINING_Y_CSV = os.path.join(CHECKPOINT_FOLDER, "training_Y.csv")
_TRAINING_Y_DF_CSV = os.path.join(CHECKPOINT_FOLDER, "training_Y_df.csv")
_TESTING_X_CSV = os.path.join(CHECKPOINT_FOLDER, "testing_X.csv")

# Legacy pickle file (for migration)
_LEGACY_STATE_FILE = os.path.join(CHECKPOINT_FOLDER, "pipeline_state.pkl")


class MachineLearningPipeline:
    """Holds all state for the five-phase ML pipeline."""

    PIPELINE_STATE_FILE = _STATE_JSON

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
        """Persist pipeline state to disk (JSON + CSV + pickle for model)."""
        try:
            # Save JSON-serialisable scalars
            state = {
                "feature_spec": self.feature_spec,
                "target_variable": self.target_variable,
                "training_Y_column": self.training_Y_column,
                "rules": self.rules,
                "mse": self.mse,
                "is_trained": self.is_trained,
                "_training_columns": self._training_columns,
            }
            with open(_STATE_JSON, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            # Save DataFrames as CSV
            self._save_df(self.training_X, _TRAINING_X_CSV)
            self._save_series(self.training_Y, _TRAINING_Y_CSV)
            self._save_df(self.training_Y_df, _TRAINING_Y_DF_CSV)
            self._save_df(self.testing_X, _TESTING_X_CSV)

            # Model must stay pickle (RuleKit Java object)
            if self.model is not None:
                with open(_MODEL_PKL, "wb") as f:
                    pickle.dump(self.model, f)
            elif os.path.exists(_MODEL_PKL):
                os.remove(_MODEL_PKL)

            logger.info("Pipeline state saved to disk.")
        except Exception as e:
            logger.warning("Cannot save pipeline state: %s", e)

    def load_state(self) -> bool:
        """Load pipeline state from disk if it exists. Returns True on success."""
        # Try new JSON format first
        if os.path.exists(_STATE_JSON):
            return self._load_json_state()
        # Fallback: migrate legacy pickle
        if os.path.exists(_LEGACY_STATE_FILE):
            ok = self._load_legacy_pickle()
            if ok:
                self.save_state()  # Re-save in new format
                try:
                    os.remove(_LEGACY_STATE_FILE)
                except OSError:
                    pass
            return ok
        return False

    def _load_json_state(self) -> bool:
        try:
            with open(_STATE_JSON, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.feature_spec = state.get("feature_spec", {})
            self.target_variable = state.get("target_variable", "")
            self.training_Y_column = state.get("training_Y_column", "")
            self.rules = state.get("rules", [])
            self.mse = state.get("mse")
            self.is_trained = state.get("is_trained", False)
            self._training_columns = state.get("_training_columns", [])

            self.training_X = self._load_df(_TRAINING_X_CSV)
            self.training_Y = self._load_series(_TRAINING_Y_CSV)
            self.training_Y_df = self._load_df(_TRAINING_Y_DF_CSV)
            self.testing_X = self._load_df(_TESTING_X_CSV)

            if os.path.exists(_MODEL_PKL):
                with open(_MODEL_PKL, "rb") as f:
                    self.model = pickle.load(f)

            logger.info("Pipeline state loaded from disk.")
            return True
        except Exception as e:
            logger.warning("Cannot load pipeline state: %s", e)
            return False

    def _load_legacy_pickle(self) -> bool:
        try:
            with open(_LEGACY_STATE_FILE, "rb") as f:
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
            logger.info("Pipeline state loaded from legacy pickle (will migrate).")
            return True
        except Exception as e:
            logger.warning("Cannot load legacy pipeline state: %s", e)
            return False

    # ------------------------------------------------------------------
    # DataFrame I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_df(df: pd.DataFrame | None, path: str) -> None:
        if df is not None:
            df.to_csv(path, index=False)
        elif os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _save_series(s: pd.Series | None, path: str) -> None:
        if s is not None:
            s.to_csv(path, index=False, header=True)
        elif os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _load_df(path: str) -> pd.DataFrame | None:
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    @staticmethod
    def _load_series(path: str) -> pd.Series | None:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df.columns) == 1:
                return df.iloc[:, 0]
            return df.iloc[:, 0]
        return None

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
