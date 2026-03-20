from flask import Flask, request, jsonify, send_from_directory
import os
import shutil
import logging
import zipfile
import threading
import uuid
import json
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from services.processing import process_single_media, _is_media_file
from flask_cors import CORS
from rulekit.regression import RuleRegressor
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Povolíme CORS pro tvou novou doménu i localhost
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

ALLOWED_EXTENSIONS = {
    "image": {"png", "jpg", "jpeg", "webp", "heic", "gif"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
MEDIA_EXTS = ALLOWED_EXTENSIONS["video"] | ALLOWED_EXTENSIONS["image"]
ALL_ALLOWED = MEDIA_EXTS | ALLOWED_EXTENSIONS["zip"]

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
CHECKPOINT_FOLDER = "checkpoints"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

JOBS = {}  # Async job tracking


def allowed_file(filename: str, allowed_exts: set | None = None) -> bool:
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in (allowed_exts or ALL_ALLOWED)


def _extract_zip_contents(zip_path: str, extract_path: str):
    """Rozbalí ZIP a vrátí (media_files, csv_file_path_or_None)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    csv_file = None
    media_files = []
    for root, _dirs, files in os.walk(extract_path):
        for f in files:
            if f.startswith("._") or "__MACOSX" in root:
                continue
            full = os.path.join(root, f)
            logger.info(f"ZIP obsah: {full}")
            if f.lower().endswith((".csv", ".xlsx")):
                csv_file = full
            elif _is_media_file(full):
                media_files.append(full)
    logger.info(f"Nalezeno médií: {len(media_files)}, CSV: {csv_file}")
    return media_files, csv_file


# ================================================================
#  PIPELINE STATE
# ================================================================

class MachineLearningPipeline:
    def __init__(self):
        self.feature_spec: dict = {}
        self.target_variable: str = ""
        # Phase 2 outputs
        self.training_X: pd.DataFrame | None = None
        self.training_Y: pd.Series | None = None
        self.training_Y_column: str = ""
        # Phase 3 outputs
        self.model = None
        self.rules: list[str] = []
        self.mse: float | None = None
        self.is_trained: bool = False
        # Phase 4 outputs
        self.testing_X: pd.DataFrame | None = None

    PIPELINE_STATE_FILE = os.path.join(CHECKPOINT_FOLDER, "pipeline_state.pkl")

    def save_state(self):
        """Uloží stav pipeline na disk."""
        try:
            with open(self.PIPELINE_STATE_FILE, "wb") as f:
                pickle.dump({
                    "feature_spec": self.feature_spec,
                    "target_variable": self.target_variable,
                    "training_X": self.training_X,
                    "training_Y_df": getattr(self, "training_Y_df", None),
                    "training_Y": self.training_Y,
                    "training_Y_column": self.training_Y_column,
                    "model": self.model,
                    "rules": self.rules,
                    "mse": self.mse,
                    "is_trained": self.is_trained,
                    "testing_X": self.testing_X,
                    "_training_columns": getattr(self, "_training_columns", []),
                }, f)
            logger.info("Pipeline stav uložen na disk.")
        except Exception as e:
            logger.warning(f"Nelze uložit pipeline stav: {e}")

    def load_state(self):
        """Načte stav pipeline z disku pokud existuje."""
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
            logger.info("Pipeline stav načten z disku.")
            return True
        except Exception as e:
            logger.warning(f"Nelze načíst pipeline stav: {e}")
            return False

    # ==========================================
    # FÁZE 1: Feature Discovery
    # ==========================================
    def discover_features(self, media_paths: list[str], target_variable: str,
                          model_name: str, labels_df: pd.DataFrame | None = None) -> dict:
        """Analyzuje ukázková média a navrhne feature definition spec."""
        self.target_variable = target_variable

        # Pokud máme labels, přidáme info o distribuci cílové proměnné
        labels_context = ""
        if labels_df is not None:
            # Najdi sloupec s cílovou proměnnou (fuzzy match)
            target_col = None
            for c in labels_df.columns:
                if c.lower().replace(" ", "_") == target_variable.lower().replace(" ", "_"):
                    target_col = c
                    break
            if target_col is None:
                # Vezmi poslední numerický sloupec jako target
                numeric_cols = labels_df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[-1]

            if target_col is not None:
                col_data = labels_df[target_col]
                labels_context = (
                    f"\n\nYou also have access to the target variable '{target_col}' from the training labels:\n"
                    f"- Min: {col_data.min()}, Max: {col_data.max()}, Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}\n"
                    f"- Sample values: {list(col_data.head(10).values)}\n"
                    f"Use this information to suggest features that would correlate well with these target values.\n"
                )

        prompt = (
            f"You are a machine learning feature engineer.\n"
            f"The goal is to build a model to predict: '{target_variable}'.\n"
            f"Analyze the provided media sample(s) and suggest 3 to 8 features "
            f"that are highly relevant for predicting the target.\n\n"
            f"For each feature, provide:\n"
            f"- A descriptive name (lowercase_with_underscores)\n"
            f"- Expected value range, units, or categories\n\n"
            f"{labels_context}"
            f"Output STRICTLY a JSON object where keys are feature names "
            f"and values are descriptions with ranges/units.\n"
            f"Example: {{\"movie_length\": \"duration in seconds (0-7200)\", "
            f"\"extreme_language\": \"score 0-10\"}}"
        )

        # Zpracuj první médium (nebo více vzorků)
        all_features = {}
        for path in media_paths[:3]:  # max 3 vzorky
            result = process_single_media(path, prompt=prompt, model_name=model_name)
            analysis = result.get("analysis")
            if isinstance(analysis, dict):
                # LLM může vrátit features přímo nebo v "attributes"
                features = analysis.get("attributes", analysis)
                # Odfiltruj meta-klíče z VIDEO_ANALYSIS_TEMPLATE (pokud LLM je přidá)
                for key in ("summary", "classification", "reasoning"):
                    features.pop(key, None)
                all_features.update(features)

        if all_features:
            self.feature_spec = all_features
            return all_features

        # Fallback
        return {"visual_complexity": "score 1-10", "action_intensity": "score 1-10"}

    # ==========================================
    # FÁZE 2: Feature Extraction (async)
    # ==========================================
    def extract_features_async(self, media_files: list[str], feature_spec: dict,
                                job_id: str, model_name: str, dataset_type: str,
                                csv_path: str | None = None,
                                labels_df: pd.DataFrame | None = None):
        """Extrahuje features z médií podle feature_spec. Běží v threadu."""
        try:
            self.feature_spec = feature_spec
            spec_string = json.dumps(feature_spec, ensure_ascii=False)
            total = len(media_files)

            # Volitelný kontext z labels
            labels_context = ""
            if labels_df is not None:
                numeric_cols = labels_df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[-1]
                    labels_context = (
                        f"\nThe target variable '{target_col}' has range "
                        f"[{labels_df[target_col].min()}, {labels_df[target_col].max()}]. "
                        f"Use this context to calibrate your feature value estimates.\n"
                    )

            prompt = (
                f"You are a feature extraction AI.\n"
                f"Extract EXACTLY these features from the provided media:\n{spec_string}\n\n"
                f"{labels_context}"
                f"Output STRICTLY a valid JSON object with these exact keys "
                f"and their corresponding numerical or categorical values. "
                f"No extra keys, no explanations."
            )

            # Checkpoint soubor pro resume
            checkpoint_file = os.path.join(CHECKPOINT_FOLDER, f"extract_{dataset_type}.json")
            features_data = []
            done_names = set()
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as cf:
                        features_data = json.load(cf)
                    done_names = {row["media_name"] for row in features_data}
                    logger.info(f"Resume: načteno {len(features_data)} záznamů z checkpointu.")
                except Exception as e:
                    logger.warning(f"Nelze načíst checkpoint: {e}")
                    features_data = []

            JOBS[job_id] = {"progress": 5, "stage": f"Zahajuji extrakci features... ({len(done_names)} již hotovo)", "done": False}

            for i, media_path in enumerate(media_files):
                file_name = os.path.basename(media_path)
                media_name = os.path.splitext(file_name)[0]

                # Přeskoč již zpracované
                if media_name in done_names:
                    progress = 5 + int((i / total) * 90)
                    JOBS[job_id] = {
                        "progress": progress,
                        "stage": f"Přeskakuji ({i+1}/{total}): {file_name} (již hotovo)",
                        "done": False,
                    }
                    continue

                progress = 5 + int((i / total) * 90)
                JOBS[job_id] = {
                    "progress": progress,
                    "stage": f"Extrakce ({i+1}/{total}): {file_name}",
                    "done": False,
                }

                result = process_single_media(media_path, prompt=prompt, model_name=model_name)
                analysis = result.get("analysis", {})
                if isinstance(analysis, dict):
                    attrs = analysis.get("attributes", analysis)
                    # Odfiltruj meta-klíče
                    for key in ("summary", "classification", "reasoning"):
                        attrs.pop(key, None)
                else:
                    attrs = {}

                row = {"media_name": media_name}
                missing = []
                for feat_key in feature_spec:
                    val = attrs.get(feat_key)
                    if val is None:
                        missing.append(feat_key)
                        val = 0
                    # Flatten lists/dicts to scalar values for DataFrame compatibility
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val) if val else ""
                    elif isinstance(val, dict):
                        val = json.dumps(val, ensure_ascii=False)
                    row[feat_key] = val
                if missing:
                    logger.warning(f"{file_name}: chybějící features {missing}")
                features_data.append(row)

                # Průběžně ulož checkpoint
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as cf:
                        json.dump(features_data, cf, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Nelze zapsat checkpoint: {e}")

            df_X = pd.DataFrame(features_data)

            # Načti dataset_Y (CSV z ZIPu) pokud existuje
            df_Y = None
            if csv_path:
                try:
                    df_Y = pd.read_csv(csv_path)
                except Exception as e:
                    logger.warning(f"Nelze načíst CSV labels: {e}")

            # Ulož do pipeline
            if dataset_type == "training":
                self.training_X = df_X
                if df_Y is not None:
                    self.training_Y_df = df_Y
            elif dataset_type == "testing":
                self.testing_X = df_X

            # Persistuj stav na disk + smaž checkpoint (extrakce dokončena)
            self.save_state()
            try:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
            except Exception:
                pass

            result_payload = {
                "progress": 100,
                "stage": "Extrakce dokončena!",
                "done": True,
                "details": {
                    "status": "success",
                    "dataset_type": dataset_type,
                    "dataset_X": df_X.to_dict(orient="records"),
                    "feature_spec": feature_spec,
                    "rows_count": len(df_X),
                }
            }
            if df_Y is not None:
                result_payload["details"]["dataset_Y_columns"] = list(df_Y.columns)
            JOBS[job_id] = result_payload

        except Exception as e:
            logger.exception(f"Chyba při extrakci job {job_id}")
            JOBS[job_id] = {"progress": 100, "stage": "Chyba", "done": True, "error": str(e)}

    # ==========================================
    # FÁZE 3: ML Training (RuleKit)
    # ==========================================
    def train_model(self, target_column: str) -> dict:
        """Natrénuje RuleKit model z uložených training_X + training_Y."""
        if self.training_X is None:
            raise Exception("Nejprve proveďte Fázi 2 (extrakce features).")

        if not hasattr(self, 'training_Y_df') or self.training_Y_df is None:
            raise Exception("Chybí dataset_Y (CSV s labels). Musí být součástí trénovacího ZIPu.")

        df_gt = self.training_Y_df
        # Sjednoť join sloupec (první sloupec = ID)
        join_col = df_gt.columns[0]
        df_gt = df_gt.rename(columns={join_col: "media_name"})

        # Strip přípony z media_name v obou tabulkách
        self.training_X["media_name"] = self.training_X["media_name"].astype(str).str.strip()
        df_gt["media_name"] = df_gt["media_name"].astype(str).str.strip()

        df_merged = pd.merge(self.training_X, df_gt, on="media_name", how="inner")

        if df_merged.empty:
            raise Exception(
                "Po spojení dataset_X s dataset_Y nezbyla žádná data. "
                "Zkontrolujte, že názvy souborů v CSV odpovídají názvům médií (bez přípony)."
            )

        feature_cols = [c for c in self.feature_spec if c in df_merged.columns]
        if not feature_cols:
            raise Exception("Žádná z features nebyla nalezena v datech.")

        X = df_merged[feature_cols].copy()
        # Flatten any list/dict values that slipped through extraction
        for col in X.columns:
            X[col] = X[col].apply(lambda v: ", ".join(str(i) for i in v) if isinstance(v, list) else v)
        X = pd.get_dummies(X)

        if target_column in df_merged.columns:
            y = df_merged[target_column]
        else:
            # Zkus najít sloupec (fuzzy - ignoruj case)
            found = [c for c in df_merged.columns if c.lower() == target_column.lower()]
            if found:
                y = df_merged[found[0]]
            else:
                raise Exception(
                    f"Sloupec '{target_column}' nenalezen v CSV. "
                    f"Dostupné sloupce: {list(df_gt.columns)}"
                )

        self.model = RuleRegressor()
        self.model.fit(X, y)

        self.rules = [str(rule) for rule in self.model.model.rules] if hasattr(self.model, 'model') else []
        self.mse = round(float(mean_squared_error(y, self.model.predict(X))), 4)
        self.is_trained = True
        self.target_variable = target_column
        # Store training columns order for prediction compatibility
        self._training_columns = list(X.columns)

        self.save_state()

        return {
            "status": "success",
            "mse": self.mse,
            "rules_count": len(self.rules),
            "rules": self.rules,
            "feature_spec": self.feature_spec,
            "training_data_X": self.training_X.to_dict(orient="records"),
        }

    # ==========================================
    # FÁZE 5: Predikce (batch)
    # ==========================================
    def predict_batch(self, testing_Y_df: pd.DataFrame | None = None) -> dict:
        """Predikuje pro všechny objekty v testing_X. Volitelně porovná s testing_Y."""
        if not self.is_trained:
            raise Exception("Model není natrénovaný. Proveďte Fázi 3.")
        if self.testing_X is None or self.testing_X.empty:
            raise Exception("Chybí testovací dataset_X. Proveďte Fázi 4.")

        feature_cols = [c for c in self.feature_spec if c in self.testing_X.columns]
        X_test = self.testing_X[feature_cols].copy()
        for col in X_test.columns:
            X_test[col] = X_test[col].apply(lambda v: ", ".join(str(i) for i in v) if isinstance(v, list) else v)
        X_test = pd.get_dummies(X_test)

        # Zajisti stejné sloupce jako při tréninku
        for col in self._training_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self._training_columns]

        predictions = self.model.predict(X_test)

        # Pokud máme testing_Y, spáruj s predikcemi
        actual_values = {}
        if testing_Y_df is not None:
            join_col = testing_Y_df.columns[0]
            testing_Y_df = testing_Y_df.rename(columns={join_col: "media_name"})
            testing_Y_df["media_name"] = testing_Y_df["media_name"].astype(str).str.strip()
            # Najdi target sloupec
            target_col = None
            for c in testing_Y_df.columns:
                if c.lower() == self.target_variable.lower() or c == self.target_variable:
                    target_col = c
                    break
            if target_col is None:
                numeric_cols = [c for c in testing_Y_df.columns if c != "media_name"
                                and pd.api.types.is_numeric_dtype(testing_Y_df[c])]
                if numeric_cols:
                    target_col = numeric_cols[-1]
            if target_col:
                for _, row in testing_Y_df.iterrows():
                    actual_values[str(row["media_name"]).strip()] = float(row[target_col])

        results = []
        pred_list = []
        actual_list = []

        for i, row in self.testing_X.iterrows():
            pred_score = float(predictions[i])
            media_name = str(row.get("media_name", f"object_{i}"))
            # Najdi odpovídající pravidlo (zjednodušeně: první matching)
            rule = self.rules[0] if self.rules else "Default rule"

            item = {
                "media_name": media_name,
                "predicted_score": round(pred_score, 4),
                "rule_applied": rule,
                "extracted_features": {k: row[k] for k in self.feature_spec if k in row},
            }

            # Přidej actual score pokud máme testing_Y
            if media_name in actual_values:
                item["actual_score"] = round(actual_values[media_name], 4)
                pred_list.append(pred_score)
                actual_list.append(actual_values[media_name])

            results.append(item)

        # Spočítej metriky pokud máme párované hodnoty
        metrics = None
        if pred_list and actual_list:
            import numpy as np
            pred_arr = np.array(pred_list)
            actual_arr = np.array(actual_list)
            mse = float(np.mean((pred_arr - actual_arr) ** 2))
            mae = float(np.mean(np.abs(pred_arr - actual_arr)))
            # Pearson korelace
            if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
                correlation = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
            else:
                correlation = None
            metrics = {
                "mse": round(mse, 6),
                "mae": round(mae, 6),
                "correlation": round(correlation, 4) if correlation is not None else None,
                "matched_count": len(pred_list),
                "total_count": len(results),
            }

        return {"predictions": results, "metrics": metrics}


pipeline = MachineLearningPipeline()
pipeline.load_state()  # Obnov stav po restartu serveru


# ================================================================
#  API ENDPOINTS
# ================================================================

@app.route("/discover", methods=["POST"])
def api_discover():
    """Fáze 1: Feature Discovery z ukázkových médií (multiple files nebo ZIP)."""
    # Accept both "files" (multiple) and legacy "file" (single)
    uploaded = request.files.getlist("files")
    if not uploaded:
        uploaded = request.files.getlist("file")
    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    target_var = request.form.get("target_variable", "target value")
    model_name = request.form.get("model", "qwen2.5vl:7b")

    # Volitelný labels CSV
    labels_df = None
    if "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(UPLOAD_FOLDER, f"labels_{secure_filename(lf.filename)}")
            lf.save(labels_path)
            try:
                labels_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning(f"Nelze načíst labels CSV: {e}")
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    media_paths = []
    extract_path = None
    saved_paths = []

    try:
        for f in uploaded:
            if not allowed_file(f.filename):
                continue
            path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(path)
            saved_paths.append(path)

        if not saved_paths:
            return jsonify({"error": "Žádné podporované soubory."}), 400

        for path in saved_paths:
            if path.lower().endswith(".zip"):
                # Rozbal ZIP, vezmi vzorky
                extract_path = os.path.join(UPLOAD_FOLDER, f"discover_{uuid.uuid4().hex[:8]}")
                os.makedirs(extract_path, exist_ok=True)
                media_files, csv_in_zip = _extract_zip_contents(path, extract_path)
                media_paths.extend(media_files)
                if labels_df is None and csv_in_zip:
                    try:
                        labels_df = pd.read_csv(csv_in_zip)
                    except Exception:
                        pass
            else:
                media_paths.append(path)

        if not media_paths:
            return jsonify({"error": "Žádná média k analýze."}), 400

        features = pipeline.discover_features(media_paths, target_var, model_name, labels_df)
        return jsonify({"suggested_features": features})

    finally:
        if extract_path:
            shutil.rmtree(extract_path, ignore_errors=True)


@app.route("/extract", methods=["POST"])
def api_extract():
    """Fáze 2 & 4: Extrakce features z ZIP datasetu (async)."""
    if "file" not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS["zip"]):
        return jsonify({"error": "Povolený formát: .zip"}), 400

    model_name = request.form.get("model", "qwen2.5vl:7b")
    feature_spec = json.loads(request.form.get("feature_spec", "{}"))
    dataset_type = request.form.get("dataset_type", "training")  # "training" nebo "testing"

    if not feature_spec:
        return jsonify({"error": "Chybí feature_spec."}), 400

    # Volitelný labels CSV (separate upload)
    labels_df = None
    if "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(DATASET_FOLDER, f"labels_{secure_filename(lf.filename)}")
            lf.save(labels_path)
            try:
                labels_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning(f"Nelze načíst labels CSV: {e}")
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    zip_path = os.path.join(DATASET_FOLDER, secure_filename(file.filename))
    file.save(zip_path)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "stage": "Startuji extrakci...", "done": False}

    # Rozbal ZIP v hlavním threadu, spusť extrakci v threadu
    extract_path = os.path.join(DATASET_FOLDER, f"extract_{job_id}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = _extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Nelze rozbalit ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP neobsahuje žádná média."}), 400

    def _run():
        try:
            pipeline.extract_features_async(
                media_files, feature_spec, job_id, model_name, dataset_type, csv_path, labels_df
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    thread = threading.Thread(target=_run)
    thread.start()

    return jsonify({"job_id": job_id, "media_count": len(media_files)})


@app.route("/extract-local", methods=["POST"])
def api_extract_local():
    """Fáze 2 & 4: Extrakce features ze ZIP souboru již uloženého na serveru."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Očekáváno JSON tělo."}), 400

    zip_path = data.get("zip_path")
    model_name = data.get("model", "qwen2.5vl:7b")
    feature_spec = data.get("feature_spec", {})
    dataset_type = data.get("dataset_type", "training")
    labels_df = None

    if not zip_path or not os.path.exists(zip_path):
        return jsonify({"error": f"Soubor nenalezen: {zip_path}"}), 400
    if not feature_spec:
        return jsonify({"error": "Chybí feature_spec."}), 400

    labels_path = data.get("labels_path")
    if labels_path and os.path.exists(labels_path):
        try:
            labels_df = pd.read_csv(labels_path)
        except Exception as e:
            logger.warning(f"Nelze načíst labels CSV: {e}")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"progress": 0, "stage": "Startuji extrakci...", "done": False}

    extract_path = os.path.join(DATASET_FOLDER, f"extract_{job_id}")
    os.makedirs(extract_path, exist_ok=True)

    try:
        media_files, csv_path = _extract_zip_contents(zip_path, extract_path)
    except Exception as e:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": f"Nelze rozbalit ZIP: {e}"}), 400

    if not media_files:
        shutil.rmtree(extract_path, ignore_errors=True)
        return jsonify({"error": "ZIP neobsahuje žádná média."}), 400

    def _run():
        try:
            pipeline.extract_features_async(
                media_files, feature_spec, job_id, model_name, dataset_type, csv_path, labels_df
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)

    thread = threading.Thread(target=_run)
    thread.start()

    return jsonify({"job_id": job_id, "media_count": len(media_files)})


@app.route("/train", methods=["POST"])
def api_train():
    """Fáze 3: Trénink RuleKit modelu z uložených dataset_X + dataset_Y."""
    target_col = request.json.get("target_column", "") if request.is_json else request.form.get("target_column", "")

    if not target_col:
        return jsonify({"error": "Chybí target_column (název sloupce s cílovou proměnnou v CSV)."}), 400

    try:
        result = pipeline.train_model(target_col)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def api_predict():
    """Fáze 5: Predikce pro všechny objekty v testovacím datasetu."""
    # Volitelný testing_Y CSV
    testing_Y_df = None
    if request.files and "labels_file" in request.files:
        lf = request.files["labels_file"]
        if lf.filename:
            labels_path = os.path.join(UPLOAD_FOLDER, f"test_labels_{secure_filename(lf.filename)}")
            lf.save(labels_path)
            try:
                testing_Y_df = pd.read_csv(labels_path)
            except Exception as e:
                logger.warning(f"Nelze načíst testing labels CSV: {e}")
            finally:
                if os.path.exists(labels_path):
                    os.remove(labels_path)

    try:
        result = pipeline.predict_batch(testing_Y_df)
        return jsonify({
            "status": "success",
            "predictions": result["predictions"],
            "metrics": result["metrics"],
            "count": len(result["predictions"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/analyze", methods=["POST"])
def api_analyze():
    """Standalone LLM analýza médií – bez ML modelu."""
    uploaded = request.files.getlist("files")
    if not uploaded:
        uploaded = request.files.getlist("file")
    if not uploaded:
        return jsonify({"error": "Žádný soubor nebyl nahrán."}), 400

    description = request.form.get("description", "Analyze this media and describe its key visual and audio properties.")
    model_name = request.form.get("model", "qwen2.5vl:7b")

    # Ulož soubory a rozbal ZIP
    saved_paths = []
    for f in uploaded:
        fname = secure_filename(f.filename)
        dest = os.path.join(UPLOAD_FOLDER, fname)
        f.save(dest)
        if fname.lower().endswith(".zip"):
            extract_dir = os.path.join(UPLOAD_FOLDER, f"analyze_{uuid.uuid4().hex}")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(dest)
            for root, _, files in os.walk(extract_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    if _is_media_file(fp):
                        saved_paths.append(fp)
        else:
            if _is_media_file(dest):
                saved_paths.append(dest)

    if not saved_paths:
        return jsonify({"error": "Žádné podporované mediální soubory nebyly nalezeny."}), 400

    results = []
    for media_path in saved_paths:
        media_name = os.path.splitext(os.path.basename(media_path))[0]
        try:
            result = process_single_media(media_path, prompt=description, model_name=model_name)
            analysis = result.get("analysis")
            if isinstance(analysis, dict):
                attrs = analysis.get("attributes", analysis)
                for key in ("summary", "classification", "reasoning"):
                    attrs.pop(key, None)
            else:
                attrs = {"response": str(analysis)} if analysis else {}
            results.append({
                "media_name": media_name,
                "analysis": attrs,
                "transcript": result.get("transcript", ""),
            })
        except Exception as e:
            results.append({"media_name": media_name, "error": str(e), "analysis": {}})
        finally:
            try:
                os.remove(media_path)
            except Exception:
                pass

    return jsonify({"status": "success", "results": results, "count": len(results)})


@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    return jsonify(JOBS.get(job_id, {"error": "Job not found"}))


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug, use_reloader=debug)
