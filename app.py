from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import pandas as pd
import html

# --- FIX: Prevent Asyncio ConnectionResetError on Windows ---
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ------------------------------------------------------------

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
    print("Run this project with: streamlit run app.py")
    sys.exit(0)

st.set_page_config(
    page_title="Recipe by Region Recommender",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL LOADING ---
@st.cache_data(show_spinner=False)
def load_model() -> RecipeRegionRecommender:
    raw = load_recipe_data()
    return RecipeRegionRecommender.fit(raw)

model = load_model()
summary = summarize_frame(model.data)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Discovery Filters")
    regions = model.available_regions()
    region_options = ["All regions"] + regions
    selected_region = st.selectbox("Region", region_options, index=0, 
                                   format_func=lambda x: x.replace("-", " ").title())
    
    course_options = ["", "dessert", "main course", "snack", "starter"]
    selected_course = st.selectbox("Course", course_options, index=0, 
                                    format_func=lambda x: x.title() if x else "Any")
    
    diet_options = ["", "vegetarian", "non vegetarian"]
    selected_diet = st.selectbox("Diet", diet_options, index=0, 
                                  format_func=lambda x: x.title() if x else "Any")
    
    flavor_options = ["", "sweet", "spicy", "bitter", "sour"]
    selected_flavor = st.selectbox("Flavor", flavor_options, index=0, 
                                    format_func=lambda x: x.title() if x else "Any")
    
    query_text = st.text_input("Keywords", placeholder="cardamom milk dessert")
    top_n = st.slider("Recipes To Show", 3, 8, 5)

    st.divider()
    st.header("Pantry Checker")
    pantry_enabled = st.toggle("Enable Pantry Checker", value=True)
    
    selected_pantry: list[str] = []
    if pantry_enabled:
        pantry_search = st.text_input("Search Pantry", placeholder="milk, ghee, rice")
        pantry_search_lower = pantry_search.strip().casefold()
        
        pantry_ingredient_options = model.available_ingredients(
            region=selected_region,
            course=selected_course or None,
            diet=selected_diet or None,
            flavor_profile=selected_flavor or None,
        )
        
        filtered_ingredients = [
            i for i in pantry_ingredient_options 
            if not pantry_search_lower or pantry_search_lower in i.casefold()
        ]
        
        select_all = st.checkbox("Select All Visible", value=False)
        if select_all:
            selected_pantry = filtered_ingredients
        else:
            with st.expander(f"Choose ingredients ({len(filtered_ingredients)})"):
                for ingredient in filtered_ingredients:
                    if st.checkbox(ingredient.replace("-", " ").title(), key=f"p_{ingredient}"):
                        selected_pantry.append(ingredient)

# --- CLEAN LIGHT THEME CSS ---
theme_css = """
<style>
    :root {
        --ink: #17212b;
        --muted: #64707d;
        --accent: #c86f2c;
        --bg: #f3efe8;
        --panel: rgba(255, 255, 255, 0.9);
        --line: rgba(23, 33, 43, 0.08);
        --card-bg: #ffffff;
        --pill-bg: #f6ddc3;
        --pill-text: #7b3f06;
    }

    .stApp {
        background-color: var(--bg) !important;
    }

    .stApp, .stMarkdown, p, h1, h2, h3, h4, span, label, div {
        color: var(--ink) !important;
    }

    .hero {
        padding: 2rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(135deg, #fffaf1, #ffffff);
        box-shadow: 0 16px 40px rgba(31, 41, 51, 0.05);
        margin-bottom: 1.25rem;
    }

    .section-panel {
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .recipe-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: var(--card-bg) !important;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(31, 41, 51, 0.08);
    }

    .recipe-grid {
        display: flex;
        flex-direction: row;
        gap: 1.5rem;
        align-items: flex-start;
    }

    .recipe-media-container {
        flex: 0 0 300px;
    }

    .recipe-media-container img {
        width: 100%;
        height: 220px;
        object-fit: cover;
        border-radius: 16px;
        border: 1px solid var(--line);
    }

    .recipe-content-container {
        flex: 1;
    }

    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: var(--pill-bg);
        color: var(--pill-text) !important;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .reason-box {
        margin-top: 15px;
        padding: 12px 16px;
        background: rgba(200, 111, 44, 0.08);
        border-left: 4px solid var(--accent);
        border-radius: 4px;
        font-style: italic;
    }

    .tag-good { background: rgba(74, 222, 128, 0.2) !important; color: #2d6a4f !important; }
    .tag-warn { background: rgba(251, 146, 60, 0.2) !important; color: #8b4a16 !important; }
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- HELPER: CARD RENDERING ---
def render_recipe_cards(frame: pd.DataFrame, pantry_mode: bool = False) -> None:
    if frame.empty:
        st.warning("No recipes match these filters.")
        return

    for _, row in frame.iterrows():
        p_time = row.get('prep_time', 0)
        c_time = row.get('cook_time', 0)
        total_time = int(p_time if pd.notna(p_time) else 0) + int(c_time if pd.notna(c_time) else 0)
        
        image_path = ensure_recipe_image(row)
        image_uri = image_to_data_uri(image_path)
        
        name = html.escape(str(row['name']))
        ingredients = html.escape(str(row['ingredients']))
        course = html.escape(str(row['course']).title())
        region = html.escape(str(row['region_clean']).title())
        reason = html.escape(str(row.get('matched_terms', 'Excellent regional fit')))
        
        pantry_html = ""
        status_tag = "tag-good"
        status_text = "Regional Match"

        if pantry_mode:
            available = int(row.get('available_count', 0))
            total = int(row.get('total_ingredients', 1))
            coverage = row.get('coverage', 0.0)
            missing = ", ".join(row.get('missing_ingredients', []))
            can_make = bool(row.get("can_make", False))
            status_tag = "tag-good" if can_make else "tag-warn"
            status_text = "Ready to cook" if can_make else "Missing items"
            
            pantry_html = f'<p style="margin: 8px 0;"><strong>Pantry Match:</strong> {available}/{total} ingredients ({coverage:.0%})</p>'
            pantry_html += f'<p style="margin: 8px 0;"><strong>Missing:</strong> {html.escape(missing) if missing else "None"}</p>'

        html_string = (
            f'<div class="recipe-card">'
                f'<div class="recipe-grid">'
                    f'<div class="recipe-media-container">'
                        f'<img src="{image_uri}" />'
                    f'</div>'
                    f'<div class="recipe-content-container">'
                        f'<h3 style="margin: 0 0 10px 0; font-size: 1.5rem;">{name}</h3>'
                        f'<div style="margin-bottom: 12px;">'
                            f'<span class="pill">{course}</span>'
                            f'<span class="pill">{region}</span>'
                            f'<span class="pill">{total_time} mins</span>'
                            f'<span class="pill {status_tag}">{status_text}</span>'
                        f'</div>'
                        f'<p style="margin: 8px 0;"><strong>Ingredients:</strong> {ingredients}</p>'
                        f'{pantry_html}'
                        f'<div class="reason-box">'
                            f'<strong>Why it fits:</strong> {reason}'
                        f'</div>'
                    f'</div>'
                f'</div>'
            f'</div>'
        )
        
        st.write(html_string, unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown('<div class="hero"><h1>Recipe by Region Recommender</h1><p>Discovery meets your pantry.</p></div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Recipes", summary["rows"])
m2.metric("Regions", summary["known_regions"])
m3.metric("Ingredients", len(model.available_ingredients()))

st.subheader("Regional Distribution")
region_counts = model.data[model.data["region_clean"] != "Unknown"]["region_clean"].value_counts().reset_index()
st.bar_chart(region_counts.set_index("region_clean"), height=200)

tab_rec, tab_pantry = st.tabs(["🔍 Find Recipes", "🧺 Pantry Checker"])

with tab_rec:
    st.markdown('<div class="section-panel">Discover authentic regional dishes.</div>', unsafe_allow_html=True)
    recs = model.recommend(
        region=selected_region,
        query=query_text,
        course=selected_course or None,
        diet=selected_diet or None,
        flavor_profile=selected_flavor or None,
        top_n=top_n
    )
    render_recipe_cards(recs, pantry_mode=False)

with tab_pantry:
    st.markdown('<div class="section-panel">See what you can cook with your ingredients.</div>', unsafe_allow_html=True)
    if not pantry_enabled:
        st.info("Enable the Pantry Checker in the sidebar.")
    elif not selected_pantry:
        st.warning("Please select ingredients in the sidebar.")
    else:
        pantry_results = model.recommend_by_pantry(
            pantry_ingredients=selected_pantry,
            region=selected_region,
            course=selected_course or None,
            diet=selected_diet or None,
            top_n=top_n
        )
        render_recipe_cards(pantry_results, pantry_mode=True)

st.caption("Built from the Indian Food dataset.")