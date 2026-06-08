"""Tests for media processing LLM dispatch."""

from services import processing


def test_video_processing_sends_all_keyframes_in_one_multimodal_request(monkeypatch):
    captured = {}

    monkeypatch.setattr(processing, "_is_image_file", lambda _path: False)
    monkeypatch.setattr(processing, "extract_audio_from_video", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        processing,
        "extract_key_frames_with_timestamps",
        lambda *_args, **_kwargs: [("frame1", 0.0), ("frame2", 1.0), ("frame3", 2.0)],
    )
    monkeypatch.setattr(processing, "_convert_frame_to_base64", lambda frame: f"b64-{frame}")

    def fake_multimodal(image_base64_list, **kwargs):
        captured["images"] = image_base64_list
        captured["kwargs"] = kwargs
        return {"feature": 7}

    monkeypatch.setattr(processing, "extract_multimodal_features_with_llm", fake_multimodal)

    result = processing.process_single_media(
        "sample.mp4",
        "Extract features.",
        model_name="gpt-test",
        custom_base_url="https://api.openai.com/v1",
        custom_api_key="sk-test",
        custom_temperature=0.2,
    )

    assert result["analysis"] == {"feature": 7}
    assert captured["images"] == ["b64-frame1", "b64-frame2", "b64-frame3"]
    assert captured["kwargs"]["deployment_name"] == "gpt-test"
    assert captured["kwargs"]["custom_temperature"] == 0.2
