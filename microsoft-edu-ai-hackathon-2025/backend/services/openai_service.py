import os
import openai
import httpx
import numpy as np
from PIL import Image
import base64
import io
import json
import time
import threading
import fcntl
import tempfile
from pathlib import Path
from env_loader import load_backend_env

load_backend_env()


def _ollama_api_base_url() -> str:
    raw_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return raw_base if raw_base.endswith("/v1") else f"{raw_base}/v1"


def get_ollama_healthcheck_url() -> str:
    raw_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if raw_base.endswith("/v1"):
        raw_base = raw_base[:-3]
    return f"{raw_base}/api/tags"

# --- Client configuration (local Ollama only) ---
local_client = openai.OpenAI(
    base_url=_ollama_api_base_url(),
    api_key="ollama",
    timeout=httpx.Timeout(120.0, connect=5.0),
)

# Ollama cannot handle concurrent requests — serialise via a global file-based lock
# (shared across worker processes, unlike threading.Semaphore)
_OLLAMA_LOCK_FILE = os.path.join(tempfile.gettempdir(), "ollama_model_load.lock")
Path(_OLLAMA_LOCK_FILE).touch(exist_ok=True)

# Counter of threads currently waiting to acquire the lock (for /queue-info).
_ollama_waiting = 0
_waiting_lock = threading.Lock()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
_OLLAMA_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX}
OLLAMA_CPU_FALLBACK = os.getenv("OLLAMA_CPU_FALLBACK", "1").strip().lower() in ("1", "true", "yes")


def ollama_request_options() -> dict:
    return dict(_OLLAMA_OPTIONS)


def _is_gpu_load_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "unable to allocate cuda",
            "cuda0 buffer",
            "do load request",
            "/load\": eof",
            "connection reset",
        )
    )


from contextlib import contextmanager

@contextmanager
def _tracked_ollama_lock():
    """Acquire global file-based Ollama lock while tracking waiting threads."""
    global _ollama_waiting
    with _waiting_lock:
        _ollama_waiting += 1
    try:
        # File-based lock: works across all worker processes
        with open(_OLLAMA_LOCK_FILE, 'w') as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
    finally:
        with _waiting_lock:
            _ollama_waiting -= 1


def get_ollama_queue_info() -> dict:
    """Return current Ollama queue status for the /queue-info endpoint."""
    with _waiting_lock:
        waiting = _ollama_waiting
    busy = waiting > 0
    queued = max(0, waiting - 1)
    return {"busy": busy, "queued": queued}


def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def _clean_json_response(content):
    """Strip the markdown wrapper from a JSON response (common with local models)."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1]
        elif len(parts) >= 2:
            content = parts[1]
    return content.strip()


def extract_image_features_with_llm(image_base64_list, prompt=None, deployment_name=None, feature_gen=False) -> list:
    features_list = []
    model_name = deployment_name or DEFAULT_MODEL

    for img_b64 in image_base64_list:
        prompt_text = prompt or "Extract meaningful features from this image for tabular dataset construction."

        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]

        max_retries = 3
        backoff = 2
        use_cpu_fallback = False

        for attempt in range(max_retries):
            try:
                options = ollama_request_options()
                if use_cpu_fallback:
                    options["num_gpu"] = 0
                with _tracked_ollama_lock():
                    response = local_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a feature extraction assistant. You MUST output valid JSON only. No text, no markdown, just JSON."},
                            {"role": "user", "content": user_content}
                        ],
                        max_tokens=2048,
                        temperature=0.1,
                        extra_body={"options": options},
                    )
                content = response.choices[0].message.content
                clean_content = _clean_json_response(content)

                try:
                    features = json.loads(clean_content)
                except Exception:
                    features = {"features": clean_content, "error": "JSON parse error", "raw": content}

                features_list.append(features)
                break

            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    features_list.append({"error": "Rate limit exceeded."})
            except Exception as e:
                msg = str(e).lower()
                is_transient = any(
                    t in msg for t in ("eof", "load request", "connection reset", "timed out", "connection refused")
                )
                if OLLAMA_CPU_FALLBACK and not use_cpu_fallback and _is_gpu_load_error(e):
                    use_cpu_fallback = True
                    logger.warning("GPU model load failed, retrying on CPU fallback: %s", e)
                    time.sleep(2)
                    continue
                if is_transient and attempt < max_retries - 1:
                    wait = backoff * (attempt + 1)
                    logger.warning("Transient Ollama error on image extraction attempt %s, retrying in %ss: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    features_list.append({"error": f"Model error ({model_name}): {str(e)}"})
                    break

    return features_list


def extract_text_features_with_llm(text_list, prompt=None, deployment_name=None, feature_gen=False) -> list:
    features_list = []
    model_name = deployment_name or DEFAULT_MODEL

    for text in text_list:
        prompt_text = prompt or "Extract meaningful features from this text."

        system_prompt = prompt_text
        if feature_gen:
            system_prompt += "\nIMPORTANT: Return ONLY valid JSON."

        max_retries = 3
        backoff = 2
        use_cpu_fallback = False

        for attempt in range(max_retries):
            try:
                options = ollama_request_options()
                if use_cpu_fallback:
                    options["num_gpu"] = 0
                with _tracked_ollama_lock():
                    response = local_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                        ],
                        max_tokens=2048,
                        temperature=0.1,
                        extra_body={"options": options},
                    )
                content = response.choices[0].message.content
                clean_content = _clean_json_response(content)

                try:
                    features = json.loads(clean_content)
                except Exception:
                    features = {"features": clean_content}
                features_list.append(features)
                break
            except Exception as e:
                if OLLAMA_CPU_FALLBACK and not use_cpu_fallback and _is_gpu_load_error(e):
                    use_cpu_fallback = True
                    logger.warning("GPU model load failed, retrying text extraction on CPU fallback: %s", e)
                    time.sleep(2)
                    continue
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                else:
                    features_list.append({"error": str(e)})
                    break

    return features_list
