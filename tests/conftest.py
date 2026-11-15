import os
import sys
import pytest
import numpy as np
from PIL import Image, ImageDraw

# Add project root and src to Python path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

for p in [SRC_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import predict
import feedback_manager


# ---------------------------------------------------------------------------
# Synthetic Drawing Helper Fixtures (Pure PIL/NumPy - Zero External File Deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def draw_helpers():
    class Helpers:
        @staticmethod
        def make_blank(size=(280, 280), mode="RGBA", bg_color=(0, 0, 0, 0)):
            return Image.new(mode, size, bg_color)

        @staticmethod
        def make_exact_pixels(n_pixels: int, size=(280, 280), intensity: int = 255):
            """Creates a black canvas with exactly n_pixels of specified intensity."""
            arr = np.zeros(size, dtype=np.uint8)
            coords = np.unravel_index(np.arange(n_pixels), size)
            arr[coords] = intensity
            return Image.fromarray(arr, mode="L")

        @staticmethod
        def make_horizontal_line(size=(280, 280), width=200, height=10, intensity=255):
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            x0 = (size[0] - width) // 2
            y0 = (size[1] - height) // 2
            draw.rectangle([x0, y0, x0 + width - 1, y0 + height - 1], fill=intensity)
            return img

        @staticmethod
        def make_vertical_line(size=(280, 280), width=10, height=200, intensity=255):
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            x0 = (size[0] - width) // 2
            y0 = (size[1] - height) // 2
            draw.rectangle([x0, y0, x0 + width - 1, y0 + height - 1], fill=intensity)
            return img

        @staticmethod
        def make_character_8(size=(280, 280), offset=(0, 0), stroke_width=16):
            """Generates figure-8 using two stacked geometric ellipses."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            ox, oy = offset
            cx, cy = size[0] // 2 + ox, size[1] // 2 + oy
            # Top loop
            draw.ellipse([cx - 40, cy - 70, cx + 40, cy - 2], outline=255, width=stroke_width)
            # Bottom loop
            draw.ellipse([cx - 48, cy - 5, cx + 48, cy + 75], outline=255, width=stroke_width)
            return img

        @staticmethod
        def make_character_E(size=(280, 280), stroke_width=24):
            """Generates bold 'E' with large ink area but hollow interior (fill ratio < 0.82)."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            x0, y0 = 60, 40
            x1, y1 = 220, 240
            # Vertical spine
            draw.rectangle([x0, y0, x0 + stroke_width, y1], fill=255)
            # Top arm
            draw.rectangle([x0, y0, x1, y0 + stroke_width], fill=255)
            # Middle arm
            mid_y = (y0 + y1) // 2 - stroke_width // 2
            draw.rectangle([x0, mid_y, x0 + 110, mid_y + stroke_width], fill=255)
            # Bottom arm
            draw.rectangle([x0, y1 - stroke_width, x1, y1], fill=255)
            return img

        @staticmethod
        def make_multipart_i(size=(280, 280)):
            """Legitimate multi-part character 'i': stem + dot vertically aligned."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            cx = size[0] // 2
            # Dot
            draw.ellipse([cx - 7, 60, cx + 7, 74], fill=255)
            # Stem
            draw.rectangle([cx - 7, 100, cx + 7, 210], fill=255)
            return img

        @staticmethod
        def make_multipart_exclamation(size=(280, 280)):
            """Legitimate multi-part character '!': stem + dot below."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            cx = size[0] // 2
            # Stem
            draw.rectangle([cx - 8, 50, cx + 8, 175], fill=255)
            # Dot
            draw.ellipse([cx - 8, 205, cx + 8, 221], fill=255)
            return img

        @staticmethod
        def make_multipart_equals(size=(280, 280)):
            """Legitimate multi-part character '=': two horizontal parallel bars."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            cx, cy = size[0] // 2, size[1] // 2
            # Top bar
            draw.rectangle([cx - 65, cy - 35, cx + 65, cy - 15], fill=255)
            # Bottom bar
            draw.rectangle([cx - 65, cy + 15, cx + 65, cy + 35], fill=255)
            return img

        @staticmethod
        def make_multipart_percent(size=(280, 280)):
            """Legitimate multi-part character '%': slash + two circles."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            # Diagonal slash
            draw.line([(60, 220), (220, 60)], fill=255, width=14)
            # Top circle
            draw.ellipse([70, 70, 115, 115], outline=255, width=8)
            # Bottom circle
            draw.ellipse([165, 165, 210, 210], outline=255, width=8)
            return img

        @staticmethod
        def make_multipart_question(size=(280, 280)):
            """Legitimate multi-part character '?': curved top hook + bottom dot."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            cx = size[0] // 2
            # Hook arc
            draw.arc([cx - 45, 50, cx + 45, 130], start=180, end=0, fill=255, width=14)
            draw.line([(cx + 45, 90), (cx, 130), (cx, 165)], fill=255, width=14)
            # Dot
            draw.ellipse([cx - 7, 200, cx + 7, 214], fill=255)
            return img

        @staticmethod
        def make_multi_character(size=(280, 280)):
            """Two distinct side-by-side digits ('1' and '0') with horizontal aspect ratio > 1.6."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            # '1' on left (x: 40..60, y: 90..190, h=100)
            draw.rectangle([40, 90, 60, 190], fill=255)
            # '0' on right (x: 125..225, y: 90..190, w=100, h=100). Total w=185, h=100 -> aspect ratio=1.85 > 1.6
            draw.ellipse([125, 90, 225, 190], outline=255, width=16)
            return img

        @staticmethod
        def make_solid_block(size=(280, 280), block_size=(100, 100)):
            """Solid filled rectangle (>4,000 pixels and fill_ratio > 0.82)."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            cx, cy = size[0] // 2, size[1] // 2
            bw, bh = block_size
            draw.rectangle([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2], fill=255)
            return img

        @staticmethod
        def make_complex_scribble(size=(280, 280), n_fragments=8):
            """Scattered scribble with > 6 disconnected components."""
            img = Image.new("L", size, 0)
            draw = ImageDraw.Draw(img)
            for i in range(n_fragments):
                x = 40 + (i % 4) * 55
                y = 50 + (i // 4) * 80
                draw.line([(x, y), (x + 25, y + 25)], fill=255, width=6)
            return img

    return Helpers


# ---------------------------------------------------------------------------
# Loaded Ensemble Model Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def loaded_model():
    """Returns the cached production ensemble model."""
    return predict.load_model()


# ---------------------------------------------------------------------------
# Isolated Temporary Feedback Environment Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_feedback_dir(tmp_path, monkeypatch):
    """
    Redirects feedback directories and log CSV to tmp_path to ensure tests
    NEVER write to or taint data/user_feedback/.
    """
    fb_dir = tmp_path / "user_feedback"
    fb_dir.mkdir(parents=True, exist_ok=True)
    log_csv = fb_dir / "feedback_log.csv"

    monkeypatch.setattr(feedback_manager, "FEEDBACK_DIR", str(fb_dir))
    monkeypatch.setattr(feedback_manager, "FEEDBACK_LOG_CSV", str(log_csv))

    return {
        "dir": str(fb_dir),
        "csv": str(log_csv),
    }
