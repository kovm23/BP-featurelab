"""Standalone media analysis: /analyze endpoint."""
import logging
import os
import uuid
import zipfile

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from services.processing import process_single_media, _is_media_file

logger = logging.getLogger(__name__)

analyze_bp = Blueprint("analyze", __name__)

_META_KEYS = {"summary", "classification", "reasoning"}


@analyze_bp.route("/analyze", methods=["POST"])
def api_analyze():
    """Standalone LLM analysis of media files – no ML model involved."""
    uploaded = request.files.getlist("files")
    if not uploaded:
        uploaded = request.files.getlist("file")
    if not uploaded:
        return jsonify({"error": "No file was uploaded."}), 400

    description = request.form.get(
        "description",
        "Analyze this media and describe its key visual and audio properties.",
    )
    model_name = request.form.get("model", "qwen2.5vl:7b")

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
        return jsonify({"error": "No supported media files were found."}), 400

    results = []
    for media_path in saved_paths:
        media_name = os.path.splitext(os.path.basename(media_path))[0]
        try:
            result = process_single_media(media_path, prompt=description, model_name=model_name)
            analysis = result.get("analysis")
            if isinstance(analysis, dict):
                attrs = analysis.get("attributes", analysis)
                for key in _META_KEYS:
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
