"""Flask application factory."""
import logging
import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from pipeline import MachineLearningPipeline
from routes import (
    analyze_bp,
    discover_bp,
    extract_bp,
    health_bp,
    predict_bp,
    reset_bp,
    state_bp,
    status_bp,
    train_bp,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

app.register_blueprint(discover_bp)
app.register_blueprint(extract_bp)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(analyze_bp)
app.register_blueprint(status_bp)
app.register_blueprint(reset_bp)
app.register_blueprint(state_bp)
app.register_blueprint(health_bp)

pipeline = MachineLearningPipeline()
pipeline.load_state()  # Restore state after server restart

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=debug,
        use_reloader=debug,
    )
