from flask import Flask, request, jsonify
import os
import shutil
import logging
import zipfile
import threading
import uuid
import json
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

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "image": {"png", "jpg", "jpeg", "webp", "heic", "gif"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
MEDIA_EXTS = ALLOWED_EXTENSIONS["video"] | ALLOWED_EXTENSIONS["image"]
ALL_ALLOWED = MEDIA_EXTS | ALLOWED_EXTENSIONS["zip"]

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

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
            if f.startswith("._") or f.startswith("__MACOSX"):
                continue
            full = os.path.join(root, f)
            if f.lower().endswith((".csv", ".xlsx")):
                csv_file = full
            elif _is_media_file(full):
                media_files.append(full)
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

    # ==========================================
    # FÁZE 1: Feature Discovery
    # ==========================================
    def discover_features(self, media_paths: list[str], target_variable: str, model_name: str) -> dict:
        """Analyzuje ukázková média a navrhne feature definition spec."""
        self.target_variable = target_variable

        prompt = (
            f"You are a machine learning feature engineer.\n"
            f"The goal is to build a model to predict: '{target_variable}'.\n"
            f"Analyze the provided media sample(s) and suggest 3 to 8 features "
            f"that are highly relevant for predicting the target.\n\n"
            f"For each feature, provide:\n"
            f"- A descriptive name (lowercase_with_underscores)\n"
            f"- Expected value range, units, or categories\n\n"
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
                                csv_path: str | None = None):
        """Extrahuje features z médií podle feature_spec. Běží v threadu."""
        try:
            self.feature_spec = feature_spec
            spec_string = json.dumps(feature_spec, ensure_ascii=False)
            total = len(media_files)

            prompt = (
                f"You are a feature extraction AI.\n"
                f"Extract EXACTLY these features from the provided media:\n{spec_string}\n\n"
                f"Output STRICTLY a valid JSON object with these exact keys "
                f"and their corresponding numerical or categorical values. "
                f"No extra keys, no explanations."
            )

            JOBS[job_id] = {"progress": 5, "stage": "Zahajuji extrakci features...", "done": False}
            features_data = []

            for i, media_path in enumerate(media_files):
                file_name = os.path.basename(media_path)
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

                row = {"media_name": os.path.splitext(file_name)[0]}
                missing = []
                for feat_key in feature_spec:
                    val = attrs.get(feat_key)
                    if val is None:
                        missing.append(feat_key)
                        val = 0
                    row[feat_key] = val
                if missing:
                    logger.warning(f"{file_name}: chybějící features {missing}")
                features_data.append(row)

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

        X = df_merged[feature_cols]
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
    def predict_batch(self) -> list[dict]:
        """Predikuje pro všechny objekty v testing_X."""
        if not self.is_trained:
            raise Exception("Model není natrénovaný. Proveďte Fázi 3.")
        if self.testing_X is None or self.testing_X.empty:
            raise Exception("Chybí testovací dataset_X. Proveďte Fázi 4.")

        feature_cols = [c for c in self.feature_spec if c in self.testing_X.columns]
        X_test = self.testing_X[feature_cols]
        X_test = pd.get_dummies(X_test)

        # Zajisti stejné sloupce jako při tréninku
        for col in self._training_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self._training_columns]

        predictions = self.model.predict(X_test)

        results = []
        for i, row in self.testing_X.iterrows():
            pred_score = float(predictions[i])
            # Najdi odpovídající pravidlo (zjednodušeně: první matching)
            rule = self.rules[0] if self.rules else "Default rule"
            results.append({
                "media_name": row.get("media_name", f"object_{i}"),
                "predicted_score": round(pred_score, 4),
                "rule_applied": rule,
                "extracted_features": {k: row[k] for k in self.feature_spec if k in row},
            })

        return results


pipeline = MachineLearningPipeline()


# ================================================================
#  API ENDPOINTS
# ================================================================

@app.route("/discover", methods=["POST"])
def api_discover():
    """Fáze 1: Feature Discovery z ukázkových médií (ZIP nebo single file)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Nepodporovaný typ souboru."}), 400

    target_var = request.form.get("target_variable", "target value")
    model_name = request.form.get("model", "qwen2.5vl:7b")

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    media_paths = []
    extract_path = None

    try:
        if path.lower().endswith(".zip"):
            # Rozbal ZIP, vezmi vzorky
            extract_path = os.path.join(UPLOAD_FOLDER, f"discover_{uuid.uuid4().hex[:8]}")
            os.makedirs(extract_path, exist_ok=True)
            media_files, _ = _extract_zip_contents(path, extract_path)
            media_paths = media_files[:3]  # max 3 vzorky
            if not media_paths:
                return jsonify({"error": "ZIP neobsahuje žádná média."}), 400
        else:
            media_paths = [path]

        features = pipeline.discover_features(media_paths, target_var, model_name)
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
                media_files, feature_spec, job_id, model_name, dataset_type, csv_path
            )
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)
            if os.path.exists(zip_path):
                os.remove(zip_path)

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
    try:
        predictions = pipeline.predict_batch()
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "count": len(predictions),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    return jsonify(JOBS.get(job_id, {"error": "Job not found"}))


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug, use_reloader=debug)
