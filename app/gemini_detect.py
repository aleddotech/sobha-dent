from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "dent": (232, 93, 62),
    "scratch": (47, 128, 237),
    "crack": (242, 201, 76),
    "glass_shatter": (111, 207, 151),
    "lamp_broken": (155, 81, 224),
    "tire_flat": (130, 130, 130),
}

PROMPT = """You are inspecting a vehicle photo for body damage.

Detect every dent and scratch that is actually visible (including faint / small ones).
For each damage, return a polygon that tightly follows the damage outline.
Use as many vertices as needed (minimum 4, typically 6–16). Do not force a rectangle
if the dent is irregular.

Return JSON only:
{
  "damages": [
    {
      "label": "dent" | "scratch",
      "confidence": 0.0-1.0,
      "polygon_xy": [x1, y1, x2, y2, ..., xn, yn]
    }
  ]
}

Coordinates are normalized 0–1000 relative to the image (x right, y down).
If there is no damage, return {"damages": []}.
"""


@dataclass
class Detection:
    label: str
    confidence: float
    polygon: list[tuple[float, float]]
    box: list[float]
    color: tuple[int, int, int]
    area_frac: float
    n_points: int


def _norm_label(raw: str) -> str:
    n = (raw or "dent").lower().replace(" ", "_")
    if "scratch" in n:
        return "scratch"
    if "crack" in n:
        return "crack"
    return "dent" if "dent" in n else n


def _scale(vals: list[float], w: int, h: int) -> list[float]:
    if not vals:
        return vals
    mx = max(abs(v) for v in vals)
    span = 1.0 if mx <= 1.5 else 1000.0
    out = []
    for i, v in enumerate(vals):
        axis = w if i % 2 == 0 else h
        out.append(max(0.0, min(axis, (v / span) * axis)))
    return out


def _pairs(flat: list[float]) -> list[tuple[float, float]]:
    pts = []
    for i in range(0, len(flat) - 1, 2):
        pts.append((flat[i], flat[i + 1]))
    return pts


def _from_box_2d(box: list[float], w: int, h: int) -> list[tuple[float, float]]:
    """Gemini box_2d is often [ymin, xmin, ymax, xmax] in 0–1000."""
    if len(box) != 4:
        return []
    y1, x1, y2, x2 = box
    xs = _scale([x1, y1, x2, y2], w, h)
    # xs is x,y interleaved from [x1,y1,x2,y2]
    xa, ya, xb, yb = xs
    return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]


def _polygon_from_item(item: dict[str, Any], w: int, h: int) -> list[tuple[float, float]]:
    if "polygon_xy" in item and item["polygon_xy"]:
        flat = [float(v) for v in item["polygon_xy"]]
        if len(flat) == 4:
            x1, y1, x2, y2 = _scale(flat, w, h)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return _pairs(_scale(flat, w, h))
    if "box_2d" in item and item["box_2d"]:
        return _from_box_2d([float(v) for v in item["box_2d"]], w, h)
    if "box" in item and item["box"]:
        flat = [float(v) for v in item["box"]]
        if len(flat) == 4:
            x1, y1, x2, y2 = _scale(flat, w, h)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return _pairs(_scale(flat, w, h))
    return []


def _area(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def parse_damages(text: str, w: int, h: int, focus_only: bool) -> list[Detection]:
    blob = text.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?", "", blob).strip()
        blob = re.sub(r"```$", "", blob).strip()
    match = re.search(r"\{.*\}", blob, re.S)
    if match:
        blob = match.group(0)
    data = json.loads(blob)
    items = data.get("damages", data if isinstance(data, list) else [])
    out: list[Detection] = []
    img_area = float(w * h) or 1.0
    for item in items:
        if not isinstance(item, dict):
            continue
        label = _norm_label(str(item.get("label", "dent")))
        if focus_only and label not in {"dent", "scratch"}:
            continue
        pts = _polygon_from_item(item, w, h)
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [min(xs), min(ys), max(xs), max(ys)]
        conf = float(item.get("confidence", 0.7) or 0.7)
        area = _area(pts)
        out.append(
            Detection(
                label=label,
                confidence=conf,
                polygon=pts,
                box=box,
                color=COLORS.get(label, (200, 200, 200)),
                area_frac=area / img_area,
                n_points=len(pts),
            )
        )
    return out


def annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    im = image.convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for d in detections:
        fill = (*d.color, 70)
        outline = (*d.color, 255)
        draw.polygon(d.polygon, fill=fill, outline=outline)
        x, y = d.polygon[0]
        tag = f"{d.label} {d.confidence:.0%} n={d.n_points}"
        draw.rectangle((x, y - 18, x + 8 * len(tag), y), fill=outline)
        draw.text((x + 3, y - 16), tag, fill=(0, 0, 0, 255), font=font)
    return Image.alpha_composite(im, overlay).convert("RGB")


def detect_image(image: Image.Image, focus_only: bool = False) -> tuple[list[Detection], str]:
    from google import genai
    from google.genai import types

    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    client = genai.Client(api_key=key)

    rgb = image.convert("RGB")
    w, h = rgb.size
    long_side = max(w, h)
    if long_side > 1024:
        scale = 1024 / long_side
        rgb_in = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    else:
        rgb_in = rgb

    config = types.GenerateContentConfig(
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.2")),
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            )
        ],
    )
    response = client.models.generate_content(
        model=model,
        contents=[rgb_in, PROMPT],
        config=config,
    )
    text = response.text or '{"damages":[]}'
    return parse_damages(text, w, h, focus_only), text
