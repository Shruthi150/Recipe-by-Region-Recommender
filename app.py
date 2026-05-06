from __future__ import annotations

import hashlib
from pathlib import Path
import sys

# --- FIX: Prevent Asyncio ConnectionResetError on Windows ---
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ------------------------------------------------------------

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from recipe_recommender import RecipeRegionRecommender, load_recipe_data, summarize_frame
from recipe_image_service import ensure_recipe_image, image_to_data_uri

def _is_streamlit_runtime() -> bool:
    if st is None:
        return False

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if not _is_streamlit_runtime():
    print("Run this project with: .\\.venv\\Scripts\\python.exe -m streamlit run app.py")
    sys.exit(0)


st.set_page_config(
    page_title="Recipe by Region Recommender",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #64707d;
        --accent: #c86f2c;
        --accent-2: #216b6a;
        --accent-soft: #f6ddc3;
        --bg: #f3efe8;
        --panel: rgba(255, 255, 255, 0.9);
        --line: rgba(23, 33, 43, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(200, 111, 44, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(33, 107, 106, 0.12), transparent 24%),
            linear-gradient(180deg, #fffdf9 0%, var(--bg) 100%);
        color: var(--ink);
    }

    .hero {
        padding: 2rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255, 250, 241, 0.92), rgba(255, 255, 255, 0.86));
        box-shadow: 0 16px 40px rgba(31, 41, 51, 0.08);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        color: var(--ink);
        font-size: 2.45rem;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: var(--muted);
        margin-bottom: 0;
        font-size: 1.01rem;
    }

    .section-panel {
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        padding: 1rem 1.05rem;
        box-shadow: 0 12px 30px rgba(31, 41, 51, 0.05);
        margin-bottom: 1rem;
    }

    .recipe-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.96);
        padding: 0.9rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 12px 30px rgba(31, 41, 51, 0.06);
    }

    .recipe-card h3 {
        margin-bottom: 0.15rem;
        color: var(--ink);
        font-size: 1.15rem;
    }

    .recipe-subtle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }

    .recipe-grid {
        display: grid;
        grid-template-columns: 34% 1fr;
        gap: 1rem;
        align-items: stretch;
    }

    .recipe-media {
        border-radius: 18px;
        overflow: hidden;
        min-height: 100%;
        border: 1px solid rgba(23, 33, 43, 0.08);
        box-shadow: 0 8px 22px rgba(31, 41, 51, 0.08);
    }

    .recipe-media img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .recipe-body {
        padding: 0.2rem 0.15rem 0.2rem 0;
    }

    .recipe-body p {
        margin-bottom: 0.45rem;
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: #7b3f06;
        font-size: 0.8rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(31, 41, 51, 0.08);
    }

    .tag-good {
        background: rgba(33, 107, 106, 0.12);
        color: #145e5d;
    }

    .tag-warn {
        background: rgba(200, 111, 44, 0.12);
        color: #8b4a16;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.45);
        padding: 0.35rem;
        border-radius: 999px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(200, 111, 44, 0.18), rgba(33, 107, 106, 0.14));
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_model() -> RecipeRegionRecommender:
    raw = load_recipe_data()
    return RecipeRegionRecommender.fit(raw)


@st.cache_data(show_spinner=False)
def get_ingredient_options() -> list[str]:
    return load_model().available_ingredients()


def display_name(value: str) -> str:
    return value.replace("-", " ").replace("/", " / ").title()


def format_choice(value: str) -> str:
    return display_name(value) if value else "Any"


def format_region(value: str) -> str:
    if value == "All regions":
        return value
    return display_name(value)


def render_recipe_cards(frame: pd.DataFrame, pantry_mode: bool = False) -> None:
    if frame.empty:
        st.warning("No recipes match the current filters. Try loosening region, course, diet, flavor, or pantry constraints.")
        return

    for _, row in frame.iterrows():
        prep_time = "n/a" if pd.isna(row["prep_time"]) else int(row["prep_time"])
        cook_time = "n/a" if pd.isna(row["cook_time"]) else int(row["cook_time"])
        ingredients = row["ingredients"] if row["ingredients"] else "Not listed"
        if pantry_mode:
            available_count = int(row.get("available_count", 0))
            total_ingredients = int(row.get("total_ingredients", 0))
            coverage = row.get("coverage", 0.0)
            missing = row.get("missing_ingredients", [])
            available = row.get("available_ingredients", [])
            missing_text = ", ".join(missing[:8]) if missing else "None"
            available_text = ", ".join(available[:8]) if available else "None"
            status_tag = "tag-good" if bool(row.get("can_make", False)) else "tag-warn"
            status_text = "You can make this now" if bool(row.get("can_make", False)) else "Missing a few ingredients"
        else:
            available_count = 0
            total_ingredients = 0
            coverage = 0.0
            missing_text = ""
            available_text = ""
            status_tag = "tag-good"
            status_text = "Region match"

        image_path = ensure_recipe_image(row)
        image_uri = image_to_data_uri(image_path)
        card_html = f"""
        <div class="recipe-card">
            <div class="recipe-grid">
                <div class="recipe-media">
                    <img src="{image_uri}" alt="{row['name']}" />
                </div>
                <div class="recipe-body">
                    <h3>{row['name']}</h3>
                    <div class="recipe-subtle">{display_name(row['course'])} • {display_name(row['diet'])} • {display_name(row['region_clean'])}</div>
                    <div>
                        <span class="pill">Region: {display_name(row['region_clean'])}</span>
                        <span class="pill">Course: {display_name(row['course'])}</span>
                        <span class="pill">Diet: {display_name(row['diet'])}</span>
                        <span class="pill">Time: {prep_time} + {cook_time} min</span>
                        <span class="pill">Score: {row.get('score', 0.0):.2f}</span>
                        <span class="tag {status_tag}">{status_text}</span>
                    </div>
                    <p><strong>Ingredients:</strong> {ingredients}</p>
                    <p><strong>Flavor Profile:</strong> {display_name(row['flavor_profile'])} | <strong>State:</strong> {display_name(row['state'])}</p>
                    {f'<p><strong>Pantry Coverage:</strong> {available_count}/{total_ingredients} ingredients matched ({coverage:.0%})</p>' if pantry_mode else ''}
                    {f'<p><strong>Available Ingredients:</strong> {available_text}</p>' if pantry_mode else ''}
                    {f'<p><strong>Missing Ingredients:</strong> {missing_text}</p>' if pantry_mode else ''}
                    <p><strong>Why It Fits:</strong> {row.get('matched_terms', 'region fit')}</p>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


model = load_model()
summary = summarize_frame(model.data)
regions = model.available_regions()
region_options = ["All regions"] + regions
course_options = ["", "dessert", "main course", "snack", "starter"]
diet_options = ["", "vegetarian", "non vegetarian"]
flavor_options = ["", "sweet", "spicy", "bitter", "sour"]


st.markdown(
    """
    <div class="hero">
        <h1>Recipe by Region Recommender</h1>
        <p>Find region-authentic recipes, then switch to the pantry checker to see what you can cook from the ingredients at home.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Recipes", summary["rows"])
metric_b.metric("Known Regions", summary["known_regions"])
metric_c.metric("Ingredients", len(model.available_ingredients()))
metric_d.metric("Missing Region Rows", summary["missing_region_rows"])


with st.sidebar:
    st.header("Discovery Filters")
    selected_region = st.selectbox("Region", region_options, index=0, format_func=format_region)
    selected_course = st.selectbox("Course", course_options, index=0, format_func=format_choice)
    selected_diet = st.selectbox("Diet", diet_options, index=0, format_func=format_choice)
    selected_flavor = st.selectbox("Flavor", flavor_options, index=0, format_func=format_choice)
    query_text = st.text_input("Ingredient Or Craving Keywords", placeholder="cardamom milk dessert")
    top_n = st.slider("Recipes To Show", min_value=3, max_value=8, value=5)

    st.divider()
    st.header("Pantry Checker")
    pantry_enabled = st.checkbox("Check What I Can Cook From My Pantry", value=True)
    pantry_search = st.text_input("Search Pantry Ingredients", placeholder="milk, ghee, rice") if pantry_enabled else ""
    pantry_search_lower = pantry_search.strip().casefold()
    pantry_ingredient_options = model.available_ingredients(
        region=selected_region,
        course=selected_course or None,
        diet=selected_diet or None,
        flavor_profile=selected_flavor or None,
    )
    filtered_ingredients = [
        ingredient for ingredient in pantry_ingredient_options if not pantry_search_lower or pantry_search_lower in ingredient.casefold()
    ]
    select_all_filtered = st.checkbox("Select All Visible Ingredients", value=False, disabled=not pantry_enabled)
    selected_pantry: list[str] = []
    if pantry_enabled:
        st.caption(f"{len(filtered_ingredients)} ingredients available for the current filters.")
        if select_all_filtered:
            selected_pantry = filtered_ingredients
        else:
            with st.expander(f"Choose ingredients ({len(filtered_ingredients)})", expanded=False):
                cols = st.columns(2)
                for index, ingredient in enumerate(filtered_ingredients):
                    key = f"pantry_{ingredient.replace(' ', '_').replace('-', '_')}"
                    if cols[index % 2].checkbox(display_name(ingredient), key=key):
                        selected_pantry.append(ingredient)


st.subheader("Regional distribution")
region_counts = model.data.loc[model.data["region_clean"].ne("Unknown"), "region_clean"].value_counts().reset_index()
region_counts.columns = ["region", "count"]
st.bar_chart(region_counts.set_index("region"), height=260)


recipe_tab, pantry_tab = st.tabs(["Recipe finder", "Pantry checker"])

with recipe_tab:
    st.markdown(
        f"<div class='section-panel'>Use the filters on the left to find recipes for {format_choice(selected_course).lower()} dishes and the region you want.</div>",
        unsafe_allow_html=True,
    )
    recommendations = model.recommend(
        region=selected_region,
        query=query_text,
        course=selected_course or None,
        diet=selected_diet or None,
        flavor_profile=selected_flavor or None,
        top_n=top_n,
    )
    render_recipe_cards(recommendations, pantry_mode=False)

with pantry_tab:
    st.markdown(
        "<div class='section-panel'>Tick the ingredients you already have. The ingredient list comes directly from the filtered dataset, so it updates automatically if new dishes or ingredients are added.</div>",
        unsafe_allow_html=True,
    )

    if not pantry_enabled:
        st.info("Enable the pantry checker in the sidebar to start selecting ingredients.")
    elif not selected_pantry:
        st.warning("Select at least one pantry ingredient to see matching recipes.")
    else:
        pantry_results = model.recommend_by_pantry(
            pantry_ingredients=selected_pantry,
            region=selected_region,
            course=selected_course or None,
            diet=selected_diet or None,
            flavor_profile=selected_flavor or None,
            query=query_text or None,
            top_n=top_n,
        )
        st.caption(f"Selected pantry ingredients: {len(selected_pantry)}")

        if pantry_results.empty:
            st.warning("No such recipe. Consider this instead.")
            fallback_results = model.recommend_pantry_fallback(
                region=selected_region,
                course=selected_course or None,
                diet=selected_diet or None,
                flavor_profile=selected_flavor or None,
                query=query_text or None,
                top_n=top_n,
            )
            if fallback_results.empty:
                st.info("No fallback suggestions were available for the current filters.")
            else:
                st.markdown("<div class='section-panel'>Same-region alternatives with a looser course filter.</div>", unsafe_allow_html=True)
                render_recipe_cards(fallback_results, pantry_mode=False)
        else:
            render_recipe_cards(pantry_results, pantry_mode=True)


st.caption("Built from the Indian Food dataset. The recipe finder uses TF-IDF similarity over ingredients and recipe metadata, while the pantry checker scores recipes by ingredient coverage and optionally blends in the same similarity signal.")