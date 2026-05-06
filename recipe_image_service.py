from __future__ import annotations

import base64
from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
IMAGE_CACHE_DIR = BASE_DIR / "assets" / "pinterest_images"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = "".join(character if character.isalnum() else "-" for character in value)
    value = "-".join(part for part in value.split("-") if part)
    return value[:80] or "recipe"


def _clean_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default

    text = str(value).strip()
    if not text or text == "-1" or text.lower() == "nan":
        return default

    return text


def _normalize_ingredient_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def recipe_image_query(row: pd.Series) -> str:
    ingredients = row.get("ingredient_list", [])
    ingredients = [ingredient for ingredient in ingredients if ingredient]
    top_ingredients = ", ".join(ingredients[:3]) if ingredients else "indian recipe"
    parts = [row.get("name", ""), row.get("course", ""), row.get("region_clean", ""), top_ingredients]
    return ", ".join(part for part in parts if part)


def style_image(file_path: Path, size: tuple[int, int] = (900, 700), border: int = 10) -> Path:
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        fitted = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        framed = ImageOps.expand(fitted, border=border, fill=(236, 215, 189))
        framed.save(file_path, format="JPEG", quality=90)
    return file_path


def create_neutral_placeholder(file_path: Path, size: tuple[int, int] = (900, 700), border: int = 10) -> Path:
    canvas = Image.new("RGB", size, color=(243, 237, 232))
    framed = ImageOps.expand(canvas, border=border, fill=(236, 215, 189))
    framed.save(file_path, format="JPEG", quality=90)
    return file_path


def ensure_recipe_image(row: pd.Series, force_refresh: bool = False) -> Path:
    slug = slugify(str(row.get("name", "recipe")))
    file_path = IMAGE_CACHE_DIR / f"{slug}.jpg"
    if file_path.exists() and not force_refresh and file_path.stat().st_size > 0:
        return file_path

    return create_neutral_placeholder(file_path)


def image_to_data_uri(file_path: Path) -> str:
    encoded_bytes = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_bytes}"


def preload_recipe_images(frame: pd.DataFrame, limit: int | None = None) -> list[Path]:
    image_paths: list[Path] = []
    rows = frame.head(limit) if limit is not None else frame
    for _, row in rows.iterrows():
        image_paths.append(ensure_recipe_image(row))
    return image_paths
