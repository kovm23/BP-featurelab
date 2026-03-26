import os
import openai
import httpx
import numpy as np
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import json
import time

load_dotenv()

# --- Konfigurace klienta (pouze lokální Ollama) ---
local_client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=httpx.Timeout(120.0, connect=5.0),
)

DEFAULT_MODEL = "qwen2.5vl:7b"


def image_to_base64(img_arr):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def _clean_json_response(content):
    """Vyčistí markdown obal z JSON odpovědi (časté u lokálních modelů)."""
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

        for attempt in range(max_retries):
            try:
                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a feature extraction assistant. You MUST output valid JSON only. No text, no markdown, just JSON."},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=2048,
                    temperature=0.1,
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

        for attempt in range(max_retries):
            try:
                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=2048,
                    temperature=0.1,
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
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                else:
                    features_list.append({"error": str(e)})
                    break

    return features_list
