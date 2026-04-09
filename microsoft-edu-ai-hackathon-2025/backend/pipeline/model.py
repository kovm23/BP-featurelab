"""MachineLearningPipeline – central state object for the ML workflow."""
import json
import logging
import os
import pickle

import pandas as pd

from config import CHECKPOINT_FOLDER
from pipeline.feature_discovery import discover_features
from pipeline.feature_extraction import extract_features_async
from pipeline.feature_schema import normalize_feature_spec
from pipeline.ml_training import train_model, predict_batch

logger = logging.getLogger(__name__)


def _json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MachineLearningPipeline:
    """Holds all state for the five-phase ML pipeline."""

    def __init__(self, checkpoint_folder: str | None = None):
        self._checkpoint_folder = checkpoint_folder or CHECKPOINT_FOLDER
        os.makedirs(self._checkpoint_folder, exist_ok=True)

        self._state_json = os.path.join(self._checkpoint_folder, "pipeline_state.json")
        self._model_pkl = os.path.join(self._checkpoint_folder, "model.pkl")
        self._xgb_model_pkl = os.path.join(self._checkpoint_folder, "xgb_model.pkl")
        self._training_x_csv = os.path.join(self._checkpoint_folder, "training_X.csv")
        self._training_y_csv = os.path.join(self._checkpoint_folder, "training_Y.csv")
        self._training_y_df_csv = os.path.join(self._checkpoint_folder, "training_Y_df.csv")
        self._testing_x_csv = os.path.join(self._checkpoint_folder, "testing_X.csv")
        self._legacy_state_file = os.path.join(self._checkpoint_folder, "pipeline_state.pkl")
        # Alias kept for compatibility with any external references
        self.PIPELINE_STATE_FILE = self._state_json
        self.feature_spec: dict = {}
        self.target_variable: str = ""
        self.target_mode: str = "regression"
        # Phase 2 outputs
        self.training_X: pd.DataFrame | None = None
        self.training_Y: pd.Series | None = None
        self.training_Y_df: pd.DataFrame | None = None
        self.training_Y_column: str = ""
        # Phase 3 outputs
        self.model = None          # RuleKit model
        self.xgb_model = None      # XGBoost model
        self.rules: list[str] = []
        self.mse: float | None = None
        self.rulekit_mse: float | None = None
        self.xgb_mse: float | None = None
        self.cv_mse: float | None = None
        self.cv_std: float | None = None
        self.cv_mae: float | None = None
        self.feature_importance: dict = {}
        self.is_trained: bool = False
        self.train_accuracy: float | None = None
        self.train_balanced_accuracy: float | None = None
        self.train_f1_macro: float | None = None
        self.train_mcc: float | None = None
        self.cv_accuracy: float | None = None
        self.cv_balanced_accuracy: float | None = None
        self.cv_f1_macro: float | None = None
        self.cv_precision_macro: float | None = None
        self.cv_recall_macro: float | None = None
        self.cv_mcc: float | None = None
        self.cv_folds: int | None = None
        self.warnings: list[str] = []
        self._label_classes: list[str] = []
        # Phase 4 outputs
        self.testing_X: pd.DataFrame | None = None
        # Phase 5 outputs
        self.predictions: list[dict] | None = None
        self.prediction_metrics: dict | None = None
        # Internal
        self._training_columns: list[str] = []
        self._scaler_mean: list[float] = []
        self._scaler_scale: list[float] = []

    def _clear_model_outputs(self) -> None:
        self.model = None
        self.xgb_model = None
        self.rules = []
        self.mse = None
        self.rulekit_mse = None
        self.xgb_mse = None
        self.cv_mse = None
        self.cv_std = None
        self.cv_mae = None
        self.feature_importance = {}
        self.is_trained = False
        self.train_accuracy = None
        self.train_balanced_accuracy = None
        self.train_f1_macro = None
        self.train_mcc = None
        self.cv_accuracy = None
        self.cv_balanced_accuracy = None
        self.cv_f1_macro = None
        self.cv_precision_macro = None
        self.cv_recall_macro = None
        self.cv_mcc = None
        self.cv_folds = None
        self.warnings = []
        self._label_classes = []
        self._training_columns = []
        self._scaler_mean = []
        self._scaler_scale = []

    def _clear_training_data(self) -> None:
        self.training_X = None
        self.training_Y = None
        self.training_Y_df = None
        self.training_Y_column = ""

    def _clear_prediction_outputs(self) -> None:
        self.predictions = None
        self.prediction_metrics = None

    def invalidate_from_phase(self, start_phase: int) -> None:
        """Clear outputs invalidated by re-running *start_phase*.

        Phase mapping:
        1 = discovery
        2 = training extraction
        3 = training
        4 = testing extraction
        5 = prediction (not persisted, kept for API symmetry)
        """
        if start_phase <= 1:
            self.feature_spec = {}
            self._clear_training_data()
            self._clear_model_outputs()
            self.testing_X = None
            self._clear_prediction_outputs()
            return

        if start_phase == 2:
            self._clear_training_data()
            self._clear_model_outputs()
            self.testing_X = None
            self._clear_prediction_outputs()
            return

        if start_phase == 3:
            self._clear_model_outputs()
            self._clear_prediction_outputs()
            return

        if start_phase == 4:
            self.testing_X = None
            self._clear_prediction_outputs()
            return

        if start_phase == 5:
            self._clear_prediction_outputs()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist pipeline state to disk (JSON + CSV + pickle for models)."""
        try:
            # Save JSON-serialisable scalars
            state = {
                "feature_spec": self.feature_spec,
                "target_variable": self.target_variable,
                "target_mode": self.target_mode,
                "training_Y_column": self.training_Y_column,
                "rules": self.rules,
                "mse": self.mse,
                "rulekit_mse": self.rulekit_mse,
                "xgb_mse": self.xgb_mse,
                "cv_mse": self.cv_mse,
                "cv_std": self.cv_std,
                "cv_mae": self.cv_mae,
                "feature_importance": self.feature_importance,
                "is_trained": self.is_trained,
                "train_accuracy": self.train_accuracy,
                "train_balanced_accuracy": self.train_balanced_accuracy,
                "train_f1_macro": self.train_f1_macro,
                "train_mcc": self.train_mcc,
                "cv_accuracy": self.cv_accuracy,
                "cv_balanced_accuracy": self.cv_balanced_accuracy,
                "cv_f1_macro": self.cv_f1_macro,
                "cv_precision_macro": self.cv_precision_macro,
                "cv_recall_macro": self.cv_recall_macro,
                "cv_mcc": self.cv_mcc,
                "cv_folds": self.cv_folds,
                "warnings": self.warnings,
                "predictions": self.predictions,
                "prediction_metrics": self.prediction_metrics,
                "_label_classes": self._label_classes,
                "_training_columns": self._training_columns,
                "_scaler_mean": self._scaler_mean,
                "_scaler_scale": self._scaler_scale,
            }
            with open(self._state_json, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=_json_default)

            # Save DataFrames as CSV
            self._save_df(self.training_X, self._training_x_csv)
            self._save_series(self.training_Y, self._training_y_csv)
            self._save_df(self.training_Y_df, self._training_y_df_csv)
            self._save_df(self.testing_X, self._testing_x_csv)

            # RuleKit model (Java object — must be pickle)
            if self.model is not None:
                with open(self._model_pkl, "wb") as f:
                    pickle.dump(self.model, f)
            elif os.path.exists(self._model_pkl):
                os.remove(self._model_pkl)

            # XGBoost model
            if self.xgb_model is not None:
                with open(self._xgb_model_pkl, "wb") as f:
                    pickle.dump(self.xgb_model, f)
            elif os.path.exists(self._xgb_model_pkl):
                os.remove(self._xgb_model_pkl)

            logger.info("Pipeline state saved to disk.")
        except Exception as e:
            logger.warning("Cannot save pipeline state: %s", e)

    def load_state(self) -> bool:
        """Load pipeline state from disk if it exists. Returns True on success."""
        if os.path.exists(self._state_json):
            return self._load_json_state()
        if os.path.exists(self._legacy_state_file):
            ok = self._load_legacy_pickle()
            if ok:
                self.save_state()
                try:
                    os.remove(self._legacy_state_file)
                except OSError:
                    pass
            return ok
        return False

    def _load_json_state(self) -> bool:
        try:
            with open(self._state_json, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.feature_spec = normalize_feature_spec(state.get("feature_spec", {}))
            self.target_variable = state.get("target_variable", "")
            self.target_mode = state.get("target_mode", "regression")
            self.training_Y_column = state.get("training_Y_column", "")
            self.rules = state.get("rules", [])
            self.mse = state.get("mse")
            self.rulekit_mse = state.get("rulekit_mse")
            self.xgb_mse = state.get("xgb_mse")
            self.cv_mse = state.get("cv_mse")
            self.cv_std = state.get("cv_std")
            self.cv_mae = state.get("cv_mae")
            self.feature_importance = state.get("feature_importance", {})
            self.is_trained = state.get("is_trained", False)
            self.train_accuracy = state.get("train_accuracy")
            self.train_balanced_accuracy = state.get("train_balanced_accuracy")
            self.train_f1_macro = state.get("train_f1_macro")
            self.train_mcc = state.get("train_mcc")
            self.cv_accuracy = state.get("cv_accuracy")
            self.cv_balanced_accuracy = state.get("cv_balanced_accuracy")
            self.cv_f1_macro = state.get("cv_f1_macro")
            self.cv_precision_macro = state.get("cv_precision_macro")
            self.cv_recall_macro = state.get("cv_recall_macro")
            self.cv_mcc = state.get("cv_mcc")
            self.cv_folds = state.get("cv_folds")
            self.warnings = state.get("warnings", [])
            self.predictions = state.get("predictions")
            self.prediction_metrics = state.get("prediction_metrics")
            self._label_classes = state.get("_label_classes", [])
            self._training_columns = state.get("_training_columns", [])
            self._scaler_mean = state.get("_scaler_mean", [])
            self._scaler_scale = state.get("_scaler_scale", [])

            self.training_X = self._load_df(self._training_x_csv)
            self.training_Y = self._load_series(self._training_y_csv)
            self.training_Y_df = self._load_df(self._training_y_df_csv)
            self.testing_X = self._load_df(self._testing_x_csv)

            if os.path.exists(self._model_pkl):
                with open(self._model_pkl, "rb") as f:
                    self.model = pickle.load(f)

            if os.path.exists(self._xgb_model_pkl):
                with open(self._xgb_model_pkl, "rb") as f:
                    self.xgb_model = pickle.load(f)

            logger.info("Pipeline state loaded from disk.")
            return True
        except Exception as e:
            logger.warning("Cannot load pipeline state: %s", e)
            return False

    def _load_legacy_pickle(self) -> bool:
        try:
            with open(self._legacy_state_file, "rb") as f:
                state = pickle.load(f)
            self.feature_spec = normalize_feature_spec(state.get("feature_spec", {}))
            self.target_variable = state.get("target_variable", "")
            self.target_mode = state.get("target_mode", "regression")
            self.training_X = state.get("training_X")
            self.training_Y_df = state.get("training_Y_df")
            self.training_Y = state.get("training_Y")
            self.training_Y_column = state.get("training_Y_column", "")
            self.model = state.get("model")
            self.rules = state.get("rules", [])
            self.mse = state.get("mse")
            self.is_trained = state.get("is_trained", False)
            self.train_accuracy = state.get("train_accuracy")
            self.train_balanced_accuracy = state.get("train_balanced_accuracy")
            self.train_f1_macro = state.get("train_f1_macro")
            self.train_mcc = state.get("train_mcc")
            self.cv_accuracy = state.get("cv_accuracy")
            self.cv_balanced_accuracy = state.get("cv_balanced_accuracy")
            self.cv_f1_macro = state.get("cv_f1_macro")
            self.cv_precision_macro = state.get("cv_precision_macro")
            self.cv_recall_macro = state.get("cv_recall_macro")
            self.cv_mcc = state.get("cv_mcc")
            self.cv_folds = state.get("cv_folds")
            self.warnings = state.get("warnings", [])
            self.predictions = state.get("predictions")
            self.prediction_metrics = state.get("prediction_metrics")
            self._label_classes = state.get("_label_classes", [])
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

    def discover_features(self, media_paths, target_variable, model_name, labels_df=None, progress_cb=None):
        """Phase 1: Feature discovery."""
        return discover_features(
            self,
            media_paths,
            target_variable,
            model_name,
            labels_df,
            progress_cb=progress_cb,
        )

    def extract_features_async(
        self, media_files, feature_spec, job_id, model_name, dataset_type,
        csv_path=None, labels_df=None
    ):
        """Phase 2/4: Async feature extraction (called from a background thread)."""
        return extract_features_async(
            self, media_files, feature_spec, job_id, model_name,
            dataset_type, csv_path, labels_df
        )

    def train_model(self, target_column: str, progress_cb=None) -> dict:
        """Phase 3: Train RuleKit model."""
        return train_model(self, target_column, progress_cb=progress_cb)

    def predict_batch(self, testing_Y_df=None, progress_cb=None) -> dict:
        """Phase 5: Batch prediction."""
        return predict_batch(self, testing_Y_df, progress_cb=progress_cb)
