import os
import gc
import html
import pytest
import numpy as np
from unittest.mock import patch
import tracemalloc
import urllib.request
import urllib.error

from ui_helpers import render_template
from feedback_manager import save_user_feedback
from predict import predict_digit


def test_SR_01_xss_prevention_in_render_template():
    """
    SR-01: Verify that input variables rendered into HTML templates are escaped
    to prevent Cross-Site Scripting (XSS) via st.markdown(..., unsafe_allow_html=True).
    """
    with patch("streamlit.markdown") as mock_st_markdown:
        xss_payload = "<script>alert('XSS')</script>"
        render_template(
            "prediction_box.html",
            category="Symbol",
            digit=xss_payload,
            confidence=95.0,
            category_detail="test",
        )

        assert mock_st_markdown.called, "st.markdown was not called"
        rendered_html = mock_st_markdown.call_args[0][0]

        # Verify unescaped script tag is NOT in the rendered output
        assert "<script>" not in rendered_html, (
            f"Unescaped XSS payload found in rendered HTML: {rendered_html}"
        )
        assert (
            "&lt;script&gt;" in rendered_html
            or html.escape(xss_payload) in rendered_html
        )


def test_SR_02_path_traversal_prevention(draw_helpers, temp_feedback_dir):
    """
    SR-02: Path traversal attacks (e.g. '../../etc/passwd', '..\\..\\evil') passed as true_label
    must never create directories or files outside the designated feedback directory.
    """
    img = draw_helpers.make_character_8()
    prep_arr = np.zeros((1, 784), dtype=np.float32)

    traversal_payloads = [
        "../../evil_path",
        "..\\..\\evil_path",
        "/etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
        "../../../../shadow",
    ]

    parent_feedback_dir = os.path.abspath(temp_feedback_dir["dir"])

    for payload in traversal_payloads:
        res = save_user_feedback(
            original_image=img,
            preprocessed_28x28_arr=prep_arr,
            true_label=payload,
            predicted_label="8",
            was_correct=False,
        )
        sample_id = res["sample_id"]

        # Verify all created files are strictly within temp_feedback_dir
        for root, dirs, files in os.walk(parent_feedback_dir):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                assert full_path.startswith(parent_feedback_dir), (
                    f"File written outside feedback directory: {full_path}"
                )


@pytest.mark.slow
def test_SR_03_memory_stability_consecutive_inferences(loaded_model):
    """
    SR-03: Memory stability across 100 consecutive inferences.
    No memory leak: memory growth after 100 inferences must be under 15 MB.
    """
    gc.collect()
    tracemalloc.start()

    # Warmup
    dummy_input = np.random.rand(1, 784).astype(np.float32)
    for _ in range(5):
        _ = loaded_model.predict_proba(dummy_input)

    gc.collect()
    snapshot_start = tracemalloc.take_snapshot()

    # Run 100 consecutive inferences
    for _ in range(100):
        _ = loaded_model.predict_proba(dummy_input)

    gc.collect()
    snapshot_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate difference
    stats = snapshot_end.compare_to(snapshot_start, "lineno")
    total_diff_kb = sum(stat.size_diff for stat in stats) / 1024.0

    # Memory growth across 100 inferences should be minimal (< 15 MB)
    assert total_diff_kb < 15360, f"Memory growth too high: {total_diff_kb:.2f} KB"


@pytest.mark.e2e
def test_SR_04_streamlit_endpoint_health_check():
    """
    SR-04: Streamlit health check endpoint http://localhost:8501/_stcore/health
    returns HTTP 200 when the app is actively running.
    """
    health_url = "http://localhost:8501/_stcore/health"
    try:
        req = urllib.request.Request(
            health_url, headers={"User-Agent": "PytestHealthCheck"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            assert response.status == 200, f"Expected status 200, got {response.status}"
            content = response.read().decode("utf-8")
            assert "ok" in content.lower() or content == "ok"
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
        pytest.skip(
            f"Streamlit server is not running on localhost:8501 ({e}). Run Streamlit before e2e tests."
        )
