import pytest
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import MagicMock

from disambiguation import disambiguate_homoglyphs, is_five_pointed_star
from predict import LABEL_MAP, predict_digit
from eval_protocol import get_dataset, get_stratified_split, compute_top_confusions


# ---------------------------------------------------------------------------
# Synthetic Drawing Helpers for Disambiguation Pairs
# ---------------------------------------------------------------------------

def make_letter_P():
    """Clear 'P': top loop + left vertical stem extending downwards. Lower-right empty."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Left vertical stem
    draw.rectangle([40, 30, 52, 140], fill=255)
    # Upper loop on right
    draw.ellipse([45, 30, 95, 85], outline=255, width=12)
    return img


def make_digit_9():
    """Clear '9': top loop + right vertical stem extending downwards. Lower-left empty."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Right vertical stem
    draw.rectangle([80, 30, 92, 140], fill=255)
    # Upper loop on left
    draw.ellipse([35, 30, 85, 85], outline=255, width=12)
    return img


def make_letter_B():
    """Clear 'B': straight vertical left spine + two loops on right."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Straight left spine
    draw.rectangle([40, 30, 52, 140], fill=255)
    # Top loop
    draw.ellipse([45, 30, 90, 85], outline=255, width=12)
    # Bottom loop
    draw.ellipse([45, 80, 95, 140], outline=255, width=12)
    return img


def make_digit_8():
    """Clear '8': figure-8 with curved waist on both left and right."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 30, 95, 85], outline=255, width=12)
    draw.ellipse([36, 80, 99, 140], outline=255, width=12)
    return img


def make_letter_D():
    """Clear capital 'D': straight vertical left spine + single large right loop."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Straight vertical left spine
    draw.rectangle([40, 30, 52, 140], fill=255)
    # Large loop
    draw.ellipse([45, 30, 100, 140], outline=255, width=14)
    return img


def make_digit_0():
    """Clear '0': smooth convex oval loop, curved on left (no flat straight spine)."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([38, 30, 98, 140], outline=255, width=14)
    return img


def make_digit_6():
    """Clear '6': lower closed loop + top-left stem; top-right is completely open."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Lower loop (y: 70..140, x: 38..98)
    draw.ellipse([38, 70, 98, 140], outline=255, width=14)
    # Stem going up to top-left (PIL arc angles 180 to 270)
    draw.arc([38, 25, 98, 90], start=180, end=270, fill=255, width=14)
    return img


def make_symbol_exclamation():
    """Clear '!': vertical stem + blank horizontal gap + bottom dot."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    # Stem
    draw.rectangle([64, 30, 76, 100], fill=255)
    # Gap: y=101..120 is empty (20px gap)
    # Dot
    draw.ellipse([64, 122, 76, 134], fill=255)
    return img


def make_digit_1():
    """Clear '1': continuous vertical stroke with no gap."""
    img = Image.new("L", (140, 160), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([64, 30, 76, 140], fill=255)
    return img


def make_five_pointed_star(size=(200, 200)):
    """Synthetic 5-pointed star (★)."""
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    r_outer = 70
    r_inner = 28
    points = []
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        angle = i * np.pi / 5 - np.pi / 2
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=255)
    return img


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_DS_01_clear_P_and_clear_9_resolved():
    """DS-01: Disambiguation correctly distinguishes clear 'P' (class 25) vs clear '9' (class 9)."""
    img_P = make_letter_P()
    img_9 = make_digit_9()

    # Ambiguous probability vector where 9 and 25 are tied as top-2
    probs = np.zeros(48)
    probs[9] = 0.40
    probs[25] = 0.38

    # For P image, must resolve to 25 ('P / p')
    res_P = disambiguate_homoglyphs(raw_pred_idx=9, probabilities=probs, image=img_P, label_map=LABEL_MAP)
    assert res_P == 25, f"Expected 25 ('P / p'), got {res_P}"

    # For 9 image, must resolve to 9 ('9')
    res_9 = disambiguate_homoglyphs(raw_pred_idx=25, probabilities=probs, image=img_9, label_map=LABEL_MAP)
    assert res_9 == 9, f"Expected 9 ('9'), got {res_9}"


def test_DS_02_twin_pairs_resolved():
    """DS-02: Resolves ambiguous homoglyph twins: B/8, D/0, 6/0, !/1."""
    # 1. B/8 (class 11 vs class 8)
    img_B = make_letter_B()
    probs_B8 = np.zeros(48)
    probs_B8[8] = 0.40
    probs_B8[11] = 0.38
    res_B = disambiguate_homoglyphs(raw_pred_idx=8, probabilities=probs_B8, image=img_B, label_map=LABEL_MAP)
    assert res_B == 11, f"Expected 11 ('B / b'), got {res_B}"

    # 2. D/0 (class 13 vs class 0)
    img_D = make_letter_D()
    probs_D0 = np.zeros(48)
    probs_D0[0] = 0.42
    probs_D0[13] = 0.39
    res_D = disambiguate_homoglyphs(raw_pred_idx=0, probabilities=probs_D0, image=img_D, label_map=LABEL_MAP)
    assert res_D == 13, f"Expected 13 ('D / d'), got {res_D}"

    # 3. 6/0 (class 6 vs class 0)
    img_6 = make_digit_6()
    probs_60 = np.zeros(48)
    probs_60[0] = 0.41
    probs_60[6] = 0.39
    res_6 = disambiguate_homoglyphs(raw_pred_idx=0, probabilities=probs_60, image=img_6, label_map=LABEL_MAP)
    assert res_6 == 6, f"Expected 6 ('6'), got {res_6}"

    # 4. !/1 (class 42 vs class 1)
    img_excl = make_symbol_exclamation()
    probs_1_excl = np.zeros(48)
    probs_1_excl[1] = 0.41
    probs_1_excl[42] = 0.39
    res_excl = disambiguate_homoglyphs(raw_pred_idx=1, probabilities=probs_1_excl, image=img_excl, label_map=LABEL_MAP)
    assert res_excl == 42, f"Expected 42 ('!'), got {res_excl}"


@pytest.mark.smoke
def test_DS_03_consensus_guard_heuristics_not_invoked(draw_helpers, monkeypatch):
    """
    DS-03: When base models agree (majority >= 2) OR margin >= 12% OR max probability >= 35%,
    heuristics are NOT invoked (spy check).
    """
    import disambiguation
    spy_disambiguate = MagicMock(side_effect=disambiguation.disambiguate_homoglyphs)
    monkeypatch.setattr(disambiguation, "disambiguate_homoglyphs", spy_disambiguate)

    img = draw_helpers.make_character_8()
    # High-confidence prediction (clear '8' with strong consensus)
    pred_label, conf, probs, _, _, _ = predict_digit(img)

    # Heuristic disambiguation must NOT be called on decisive consensus
    assert spy_disambiguate.call_count == 0


def test_DS_04_five_pointed_star_detected_as_asterisk():
    """DS-04: 5-pointed star is detected by is_five_pointed_star and maps to '*' (class 38)."""
    star_img = make_five_pointed_star()
    assert is_five_pointed_star(star_img) is True

    # Non-star (circle / plain oval)
    circle_img = make_digit_0()
    assert is_five_pointed_star(circle_img) is False

    # In predict_digit, 5-pointed star predicts '*'
    pred_label, conf, probs, _, _, cat = predict_digit(star_img)
    assert pred_label == "*"
    assert cat == "Symbol"


@pytest.mark.slow
def test_DS_05_disambiguation_accuracy_improvement():
    """
    DS-05: Verifies disambiguation preserves or improves accuracy on ambiguous twin samples.
    """
    # Evaluate across synthetic twin images
    twins = [
        (make_letter_P(), 25, 9),
        (make_digit_9(), 9, 25),
        (make_letter_B(), 11, 8),
        (make_letter_D(), 13, 0),
        (make_digit_6(), 6, 0),
        (make_symbol_exclamation(), 42, 1),
    ]

    for img, true_cls, competing_cls in twins:
        probs = np.zeros(48)
        probs[competing_cls] = 0.40
        probs[true_cls] = 0.38
        resolved = disambiguate_homoglyphs(competing_cls, probs, img, LABEL_MAP)
        assert resolved == true_cls, f"Disambiguation failed for true class {true_cls} vs {competing_cls}"


@pytest.mark.slow
def test_DS_06_known_confusion_regression_ceiling(loaded_model):
    """
    DS-06: Known-confusion regression test on the fixed test split (6,720 samples):
    Top errors must not exceed the audited baseline ceilings:
      / -> 1: <= 45
      I -> L: <= 44
      - -> I: <= 35
      Z -> L: <= 24
      F -> L: <= 22
    """
    X, y = get_dataset()
    _, X_test, _, y_test = get_stratified_split(X, y, test_size=0.2, random_state=42)

    y_pred = loaded_model.predict(X_test)
    confused_pairs = compute_top_confusions(y_test, y_pred, top_n=20)

    pair_dict = {f"{p['true_label']}->{p['predicted_label']}": p['count'] for p in confused_pairs}

    # Baseline limits
    assert pair_dict.get("/->1", 0) <= 45, f"Slash to 1 confusion exceeded baseline: {pair_dict.get('/->1')}"
    assert pair_dict.get("I / i->L / l", 0) <= 44, f"I to L confusion exceeded baseline: {pair_dict.get('I / i->L / l')}"
    assert pair_dict.get("-->I / i", 0) <= 35, f"Dash to I confusion exceeded baseline: {pair_dict.get('-->I / i')}"
    assert pair_dict.get("Z / z->L / l", 0) <= 24, f"Z to L confusion exceeded baseline: {pair_dict.get('Z / z->L / l')}"
    assert pair_dict.get("F / f->L / l", 0) <= 22, f"F to L confusion exceeded baseline: {pair_dict.get('F / f->L / l')}"
