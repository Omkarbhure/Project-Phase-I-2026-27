import os
import io
import base64
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENT_DIR = os.path.join(CURRENT_DIR, "components", "canvas")

_custom_canvas_func = components.declare_component(
    "responsive_handwritten_canvas",
    path=COMPONENT_DIR
)


class CanvasResult:
    """Standardized result object matching previous canvas contracts with extended stroke metadata."""
    def __init__(self, pil_image=None, has_drawing=False, stroke_count=0, strokes=None, raw_data=None):
        self.image = pil_image
        self.has_drawing = has_drawing
        self.stroke_count = stroke_count
        self.strokes = strokes or []
        self.raw_data = raw_data or {}
        
        # Legacy compatibility attribute: image_data as numpy array
        if pil_image is not None:
            self.image_data = np.array(pil_image)
        else:
            self.image_data = None


def st_responsive_canvas(
    stroke_width: int = 20,
    canvas_size: int = 320,
    auto_predict: bool = True,
    key: str = "handwritten_canvas"
) -> CanvasResult:
    """
    Renders a responsive HTML5 handwritten canvas with:
      - Stroke-level Undo, Redo, Clear, and Eraser tools
      - Brush presets (Thin 12px, Medium 20px matching training data, Bold 28px)
      - Faint background guide crosshairs & 20x20 box (pure CSS, not rendered into output raster)
      - Touch & Stylus drawing support via PointerEvents
      - Debounced pen-up auto-prediction (~400 ms)
      - Stroke coordinate recording
    """
    component_value = _custom_canvas_func(
        stroke_width=stroke_width,
        canvas_size=canvas_size,
        auto_predict=auto_predict,
        key=key,
        default=None
    )

    if component_value is None or not isinstance(component_value, dict):
        return CanvasResult()

    data_url = component_value.get("dataUrl")
    has_drawing = bool(component_value.get("hasDrawing", False))
    stroke_count = int(component_value.get("strokeCount", 0))
    strokes = component_value.get("strokes", [])

    pil_image = None
    if has_drawing and data_url and "," in data_url:
        try:
            _, b64_str = data_url.split(",", 1)
            img_bytes = base64.b64decode(b64_str)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        except Exception:
            pil_image = None

    return CanvasResult(
        pil_image=pil_image,
        has_drawing=has_drawing and (pil_image is not None),
        stroke_count=stroke_count,
        strokes=strokes,
        raw_data=component_value
    )

