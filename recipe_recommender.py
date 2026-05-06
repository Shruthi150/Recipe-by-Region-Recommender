from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "Data" / "indian_food.csv"


def load_recipe_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the recipe dataset from disk."""

    return pd.read_csv(csv_path)


def _clean_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default

    text = str(value).strip()
    if not text or text == "-1" or text.lower() == "nan":
        return default

    return text


def _split_ingredients(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []

    parts = re.split(r",|/|;|\band\b|&", text, flags=re.IGNORECASE)
    cleaned = [re.sub(r"\s+", " ", part).strip().lower() for part in parts]
    return [part for part in cleaned if part]


def clean_recipe_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing values and build a searchable feature column."""

    df = raw_df.copy()

    # Stable unique id prevents collisions when recipe names repeat.
    df = df.reset_index(drop=True)
    df["recipe_id"] = df.index.map(lambda index: f"recipe_{index:04d}")

    for column in ["name", "ingredients", "diet", "flavor_profile", "course", "state", "region"]:
        if column in df.columns:
            df[column] = df[column].map(_clean_text)

    for column in ["prep_time", "cook_time"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[df[column] < 0, column] = np.nan

    df["ingredient_list"] = df.get("ingredients", "").map(_split_ingredients)

    df["region_clean"] = df["region"].where(df["region"].ne(""), "Unknown")
    df["region_clean"] = df["region_clean"].replace("", "Unknown")

    df["feature_text"] = (
        df[["diet", "flavor_profile", "course", "state"]]
        .fillna("")
        .astype(str)
        .assign(ingredients=df["ingredient_list"].map(" ".join))
        [["ingredients", "diet", "flavor_profile", "course", "state"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"[^a-z0-9\s,]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_ingredient_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


@dataclass
class RecipeRegionRecommender:
    data: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def fit(cls, raw_df: pd.DataFrame) -> "RecipeRegionRecommender":
        data = clean_recipe_data(raw_df)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(data["feature_text"])
        return cls(data=data, vectorizer=vectorizer, matrix=matrix)

    def available_regions(self) -> list[str]:
        regions = self.data.loc[self.data["region_clean"].ne("Unknown"), "region_clean"]
        return sorted(regions.dropna().unique().tolist())

    def available_ingredients(
        self,
        region: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
    ) -> list[str]:
        candidates = self._filtered_candidates(
            region=region,
            course=course,
            diet=diet,
            flavor_profile=flavor_profile,
        )
        ingredients = candidates["ingredient_list"].explode().dropna().map(_normalize_ingredient_name)
        unique_ingredients = sorted({ingredient for ingredient in ingredients if ingredient})
        return unique_ingredients

    def _filtered_candidates(
        self,
        region: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
    ) -> pd.DataFrame:
        candidates = self.data.copy()

        if region and region != "All regions":
            candidates = candidates.loc[candidates["region_clean"].str.casefold() == region.casefold()]

        if course:
            candidates = candidates.loc[candidates["course"].str.casefold() == course.casefold()]

        if diet:
            candidates = candidates.loc[candidates["diet"].str.casefold() == diet.casefold()]

        if flavor_profile:
            candidates = candidates.loc[candidates["flavor_profile"].str.casefold() == flavor_profile.casefold()]

        return candidates

    def _query_text(
        self,
        region: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
        query: str | None = None,
        fallback_frame: pd.DataFrame | None = None,
    ) -> str:
        parts = [region, course, diet, flavor_profile, query]
        text = " ".join(part for part in parts if part).strip()
        if text:
            return text

        frame = fallback_frame if fallback_frame is not None and not fallback_frame.empty else self.data
        if region and region != "All regions":
            frame = frame.loc[frame["region_clean"].str.casefold() == region.casefold()]

        if frame.empty:
            frame = self.data

        return " ".join(frame["feature_text"].head(min(25, len(frame))).tolist())

    def recommend(
        self,
        region: str | None = None,
        query: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        candidates = self._filtered_candidates(region=region, course=course, diet=diet, flavor_profile=flavor_profile)
        if candidates.empty:
            return candidates.copy()

        query_text = self._query_text(
            region=region,
            course=course,
            diet=diet,
            flavor_profile=flavor_profile,
            query=query,
            fallback_frame=candidates,
        )

        query_vector = self.vectorizer.transform([query_text.lower()])
        candidate_positions = candidates.index.to_list()
        scores = cosine_similarity(query_vector, self.matrix[candidate_positions]).ravel()

        result = candidates.copy()
        result["score"] = scores
        result["matched_terms"] = result["feature_text"].map(
            lambda text: ", ".join(sorted(tokenize(text) & tokenize(query_text))[:5]) if tokenize(text) & tokenize(query_text) else "region fit"
        )

        ordered = result.sort_values(by=["score", "prep_time", "cook_time", "name"], ascending=[False, True, True, True], na_position="last")
        return ordered.head(top_n).reset_index(drop=True)

    def recommend_by_pantry(
        self,
        pantry_ingredients: list[str],
        region: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
        query: str | None = None,
        top_n: int = 5,
        require_all: bool = False,
    ) -> pd.DataFrame:
        pantry_set = {_normalize_ingredient_name(item) for item in pantry_ingredients if _normalize_ingredient_name(item)}
        if not pantry_set:
            return self.data.head(0).copy()

        candidates = self._filtered_candidates(region=region, course=course, diet=diet, flavor_profile=flavor_profile)
        if candidates.empty:
            return candidates.copy()

        result = candidates.copy()
        ingredient_sets = result["ingredient_list"].map(lambda items: {_normalize_ingredient_name(item) for item in items if _normalize_ingredient_name(item)})
        result["available_ingredients"] = ingredient_sets.map(lambda items: sorted(items & pantry_set))
        result["missing_ingredients"] = ingredient_sets.map(lambda items: sorted(items - pantry_set))
        result["total_ingredients"] = ingredient_sets.map(len)
        result["available_count"] = result["available_ingredients"].map(len)
        result["coverage"] = np.where(result["total_ingredients"] > 0, result["available_count"] / result["total_ingredients"], 0.0)
        result["can_make"] = result["missing_ingredients"].map(len).eq(0)
        result = result.loc[result["available_count"] > 0]

        if require_all:
            result = result.loc[result["can_make"]]

        if result.empty:
            return result.copy()

        if query:
            query_vector = self.vectorizer.transform([query.lower()])
            similarity_scores = cosine_similarity(query_vector, self.matrix[result.index.to_list()]).ravel()
            result["query_score"] = similarity_scores
        else:
            result["query_score"] = 0.0

        result["score"] = 0.7 * result["coverage"] + 0.3 * result["query_score"]
        result["pantry_match"] = result["coverage"].map(lambda value: f"{int(round(value * 100))}%")

        ordered = result.sort_values(
            by=["can_make", "coverage", "query_score", "prep_time", "cook_time", "name"],
            ascending=[False, False, False, True, True, True],
            na_position="last",
        )
        return ordered.head(top_n).reset_index(drop=True)

    def recommend_pantry_fallback(
        self,
        region: str | None = None,
        course: str | None = None,
        diet: str | None = None,
        flavor_profile: str | None = None,
        query: str | None = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        return self.recommend(
            region=region,
            query=query,
            course=None if course else None,
            diet=diet,
            flavor_profile=flavor_profile,
            top_n=top_n,
        )


def summarize_frame(df: pd.DataFrame) -> dict[str, object]:
    known_regions = df.loc[df["region_clean"].ne("Unknown"), "region_clean"]
    courses = df["course"].replace("", pd.NA).dropna()
    diets = df["diet"].replace("", pd.NA).dropna()
    return {
        "rows": int(len(df)),
        "known_regions": int(known_regions.nunique()),
        "missing_region_rows": int((df["region_clean"] == "Unknown").sum()),
        "regions": sorted(known_regions.dropna().unique().tolist()),
        "courses": sorted(courses.unique().tolist()),
        "diet": sorted(diets.unique().tolist()),
    }
