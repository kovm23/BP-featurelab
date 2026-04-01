"""Session transfer endpoints: export/import pipeline checkpoint bundle."""
import glob
import io
import logging
import os
import zipfile

from flask import Blueprint, jsonify, request, send_file

logger = logging.getLogger(__name__)

session_transfer_bp = Blueprint("session_transfer", __name__)

_ALLOWED_IMPORT_EXT = {".json", ".csv", ".pkl"}


def _reset_pipeline_memory(pipeline) -> None:
    """Reset in-memory pipeline fields to defaults before loading imported state."""
    pipeline.feature_spec = {}
    pipeline.target_variable = ""
    pipeline.target_mode = "regression"
    pipeline.training_X = None
    pipeline.training_Y = None
    pipeline.training_Y_df = None
    pipeline.training_Y_column = ""
    pipeline.model = None
    pipeline.xgb_model = None
    pipeline.scaler = None
    pipeline.rules = []
    pipeline.mse = None
    pipeline.rulekit_mse = None
    pipeline.xgb_mse = None
    pipeline.cv_mse = None
    pipeline.cv_std = None
    pipeline.cv_mae = None
    pipeline.feature_importance = {}
    pipeline.is_trained = False
    pipeline.train_accuracy = None
    pipeline.train_f1_macro = None
    pipeline.cv_accuracy = None
    pipeline.cv_f1_macro = None
    pipeline._label_classes = []
    pipeline.testing_X = None
    pipeline._training_columns = []
    pipeline._scaler_mean = []
    pipeline._scaler_scale = []


def _clear_checkpoint_folder(checkpoint_folder: str) -> None:
    for pattern in ("*.json", "*.pkl", "*.csv"):
        for fpath in glob.glob(os.path.join(checkpoint_folder, pattern)):
            try:
                os.remove(fpath)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", fpath, e)


@session_transfer_bp.route("/export-session", methods=["GET"])
def export_session():
    """Export current session checkpoint files as a ZIP archive."""
    from app import get_pipeline

    pipeline = get_pipeline()
    checkpoint_folder = pipeline._checkpoint_folder

    files = []
    for pattern in ("*.json", "*.pkl", "*.csv"):
        files.extend(glob.glob(os.path.join(checkpoint_folder, pattern)))

    if not files:
        return jsonify({"error": "No session checkpoint data available to export."}), 400

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            zf.write(fpath, arcname=os.path.basename(fpath))

    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name="mfl_session_export.zip",
    )


@session_transfer_bp.route("/import-session", methods=["POST"])
def import_session():
    """Import a previously exported session ZIP into current session."""
    from app import get_pipeline

    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No session ZIP was uploaded."}), 400

    pipeline = get_pipeline()

    try:
        archive_bytes = uploaded.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if not members:
                return jsonify({"error": "Uploaded ZIP is empty."}), 400

            _clear_checkpoint_folder(pipeline._checkpoint_folder)

            imported = []
            for member in members:
                base_name = os.path.basename(member.filename)
                if not base_name:
                    continue
                ext = os.path.splitext(base_name)[1].lower()
                if ext not in _ALLOWED_IMPORT_EXT:
                    continue

                out_path = os.path.join(pipeline._checkpoint_folder, base_name)
                with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                imported.append(base_name)

        if not imported:
            return jsonify({"error": "ZIP does not contain supported checkpoint files (.json/.csv/.pkl)."}), 400

        _reset_pipeline_memory(pipeline)
        loaded = pipeline.load_state()
        if not loaded:
            return jsonify({"error": "Imported files were copied, but pipeline state could not be loaded."}), 400

        return jsonify({"ok": True, "imported_files": imported})
    except zipfile.BadZipFile:
        return jsonify({"error": "Uploaded file is not a valid ZIP archive."}), 400
    except Exception as e:
        logger.exception("Session import failed")
        return jsonify({"error": str(e)}), 500
