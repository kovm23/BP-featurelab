import base64
import fcntl
import io
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import numpy as np
import openai
from PIL import Image

from env_loader import load_backend_env

load_backend_env()

from config import OLLAMA_CONNECT_TIMEOUT, OLLAMA_REQUEST_TIMEOUT  # noqa: E402
from utils.ollama_errors import is_gpu_load_error, is_transient_ollama_error  # noqa: E402

logger = logging.getLogger(__name__)


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
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    timeout=httpx.Timeout(OLLAMA_REQUEST_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT),
)

# Ollama cannot handle concurrent requests — serialise via a global file-based lock
# (shared across worker processes, unlike threading.Semaphore)
_OLLAMA_LOCK_FILE = os.path.join(tempfile.gettempdir(), "ollama_model_load.lock")
Path(_OLLAMA_LOCK_FILE).touch(exist_ok=True)

# Counter of threads currently waiting to acquire the lock (for /queue-info).
_ollama_waiting = 0
_waiting_lock = threading.Lock()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:32b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
_OLLAMA_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX}
OLLAMA_CPU_FALLBACK = os.getenv("OLLAMA_CPU_FALLBACK", "1").strip().lower() in ("1", "true", "yes")
OLLAMA_MAX_COMPLETION_TOKENS = int(os.getenv("OLLAMA_MAX_COMPLETION_TOKENS", "2048"))


def _read_custom_llm_max_completion_tokens() -> int | None:
    raw_value = os.getenv("CUSTOM_LLM_MAX_COMPLETION_TOKENS", "8192").strip()
    if not raw_value or raw_value == "0":
        return None
    return int(raw_value)


CUSTOM_LLM_MAX_COMPLETION_TOKENS = _read_custom_llm_max_completion_tokens()


def ollama_request_options() -> dict:
    return dict(_OLLAMA_OPTIONS)


def get_completion_token_limit(is_custom: bool) -> int | None:
    """Return the default output-token budget for the selected LLM endpoint."""
    return CUSTOM_LLM_MAX_COMPLETION_TOKENS if is_custom else OLLAMA_MAX_COMPLETION_TOKENS


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


def _make_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client for a custom endpoint."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return openai.OpenAI(
        base_url=url,
        api_key=api_key,
        timeout=httpx.Timeout(OLLAMA_REQUEST_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT),
    )


def get_client(custom_base_url: str = "", custom_api_key: str = "") -> "tuple[openai.OpenAI, bool]":
    """Return (client, is_custom). is_custom=True means skip file lock and extra_body."""
    if custom_base_url and custom_api_key:
        return _make_client(custom_base_url, custom_api_key), True
    return local_client, False


def _is_unsupported_parameter_error(exc: BaseException, parameter_name: str) -> bool:
    msg = str(exc).lower()
    parameter = parameter_name.lower()
    unsupported_markers = (
        "unsupported parameter",
        "unrecognized parameter",
        "unknown parameter",
        "unknown field",
        "extra_forbidden",
        "not permitted",
        "not supported",
    )
    return parameter in msg and any(marker in msg for marker in unsupported_markers)


def create_chat_completion_with_token_limit(
    client,
    *,
    is_custom: bool,
    token_limit: int | None,
    **kwargs,
):
    """Create a chat completion with endpoint-specific token-limit naming."""
    if is_custom:
        request_kwargs = dict(kwargs)
        if token_limit is None:
            return client.chat.completions.create(**request_kwargs)
        request_kwargs["max_completion_tokens"] = token_limit
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if not _is_unsupported_parameter_error(exc, "max_completion_tokens"):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["max_tokens"] = token_limit
            logger.info(
                "Custom endpoint does not support max_completion_tokens; retrying with max_tokens."
            )
            return client.chat.completions.create(**fallback_kwargs)

    request_kwargs = dict(kwargs)
    request_kwargs["max_tokens"] = token_limit
    with _tracked_ollama_lock():
        return client.chat.completions.create(**request_kwargs)


def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def _clean_json_response(content):
    """Strip the markdown wrapper or leading prose from a JSON response.

    Handles:
    - ```json ... ``` blocks
    - plain ``` ... ``` blocks
    - responses where the model outputs observation text first, then a JSON object
      (e.g. "I see a road...\n\n{\"key\": \"val\"}")
    """
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1]
        elif len(parts) >= 2:
            content = parts[1]
    content = content.strip()

    # If the cleaned content doesn't start with '{', try to find the first JSON
    # object in the text (handles "observation paragraph\n\n{...}" pattern).
    if not content.startswith("{"):
        brace_start = content.find("{")
        if brace_start != -1:
            content = content[brace_start:]

    return content.strip()


def _parse_json_or_raw(content):
    clean_content = _clean_json_response(content or "")
    try:
        return json.loads(clean_content)
    except (json.JSONDecodeError, ValueError):
        return {"features": clean_content, "error": "JSON parse error", "raw": content}


def extract_multimodal_features_with_llm(
    image_base64_list,
    prompt=None,
    deployment_name=None,
    feature_gen=False,
    custom_base_url: str = "",
    custom_api_key: str = "",
    custom_temperature: float | None = None,
) -> dict:
    """Send one multimodal request containing all provided images."""
    model_name = deployment_name or DEFAULT_MODEL
    client, is_custom = get_client(custom_base_url, custom_api_key)
    prompt_text = prompt or "Extract meaningful features from these media frames."
    user_content = [{"type": "text", "text": prompt_text}]
    user_content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        for img_b64 in image_base64_list
    )

    max_retries = 3
    backoff = 2
    use_cpu_fallback = False

    for attempt in range(max_retries):
        try:
            temperature = custom_temperature if (is_custom and custom_temperature is not None) else 0.1
            kwargs: dict = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a feature extraction assistant. You MUST output valid JSON only. No text, no markdown, just JSON."},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
            )
            if not is_custom:
                options = ollama_request_options()
                if use_cpu_fallback:
                    options["num_gpu"] = 0
                kwargs["extra_body"] = {"options": options}

            response = create_chat_completion_with_token_limit(
                client,
                is_custom=is_custom,
                token_limit=get_completion_token_limit(is_custom),
                **kwargs,
            )
            return _parse_json_or_raw(response.choices[0].message.content)
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return {"error": "Rate limit exceeded."}
        except Exception as e:
            if not is_custom and OLLAMA_CPU_FALLBACK and not use_cpu_fallback and is_gpu_load_error(e):
                use_cpu_fallback = True
                logger.warning("GPU model load failed, retrying multimodal extraction on CPU fallback: %s", e)
                time.sleep(2)
                continue
            if not is_custom and is_transient_ollama_error(e) and attempt < max_retries - 1:
                wait = backoff * (attempt + 1)
                logger.warning("Transient Ollama error on multimodal extraction attempt %s, retrying in %ss: %s", attempt + 1, wait, e)
                time.sleep(wait)
                continue
            return {"error": f"Model error ({model_name}): {str(e)}"}

    return {"error": f"Model error ({model_name}): no response"}


def extract_image_features_with_llm(
    image_base64_list,
    prompt=None,
    deployment_name=None,
    feature_gen=False,
    custom_base_url: str = "",
    custom_api_key: str = "",
    custom_temperature: float | None = None,
) -> list:
    features_list = []
    model_name = deployment_name or DEFAULT_MODEL
    client, is_custom = get_client(custom_base_url, custom_api_key)

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
                temperature = custom_temperature if (is_custom and custom_temperature is not None) else 0.1
                kwargs: dict = dict(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a feature extraction assistant. You MUST output valid JSON only. No text, no markdown, just JSON."},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                )
                if not is_custom:
                    options = ollama_request_options()
                    if use_cpu_fallback:
                        options["num_gpu"] = 0
                    kwargs["extra_body"] = {"options": options}

                response = create_chat_completion_with_token_limit(
                    client,
                    is_custom=is_custom,
                    token_limit=get_completion_token_limit(is_custom),
                    **kwargs,
                )

                features_list.append(_parse_json_or_raw(response.choices[0].message.content))
                break

            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    features_list.append({"error": "Rate limit exceeded."})
            except Exception as e:
                if not is_custom and OLLAMA_CPU_FALLBACK and not use_cpu_fallback and is_gpu_load_error(e):
                    use_cpu_fallback = True
                    logger.warning("GPU model load failed, retrying on CPU fallback: %s", e)
                    time.sleep(2)
                    continue
                if not is_custom and is_transient_ollama_error(e) and attempt < max_retries - 1:
                    wait = backoff * (attempt + 1)
                    logger.warning("Transient Ollama error on image extraction attempt %s, retrying in %ss: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    features_list.append({"error": f"Model error ({model_name}): {str(e)}"})
                    break

    return features_list


def extract_text_features_with_llm(
    text_list,
    prompt=None,
    deployment_name=None,
    feature_gen=False,
    custom_base_url: str = "",
    custom_api_key: str = "",
    custom_temperature: float | None = None,
) -> list:
    features_list = []
    model_name = deployment_name or DEFAULT_MODEL
    client, is_custom = get_client(custom_base_url, custom_api_key)

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
                temperature = custom_temperature if (is_custom and custom_temperature is not None) else 0.1
                kwargs: dict = dict(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=temperature,
                )
                if not is_custom:
                    options = ollama_request_options()
                    if use_cpu_fallback:
                        options["num_gpu"] = 0
                    kwargs["extra_body"] = {"options": options}

                response = create_chat_completion_with_token_limit(
                    client,
                    is_custom=is_custom,
                    token_limit=get_completion_token_limit(is_custom),
                    **kwargs,
                )

                content = response.choices[0].message.content
                clean_content = _clean_json_response(content)

                try:
                    features = json.loads(clean_content)
                except (json.JSONDecodeError, ValueError):
                    features = {"features": clean_content}
                features_list.append(features)
                break
            except Exception as e:
                if not is_custom and OLLAMA_CPU_FALLBACK and not use_cpu_fallback and is_gpu_load_error(e):
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
