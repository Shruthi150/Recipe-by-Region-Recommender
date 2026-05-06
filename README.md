# Recipe by Region Recommender

This project turns the Kaggle Indian Food dataset into a small, shareable recipe recommender focused on Indian regions. A user can either search for region-authentic dishes or switch to the pantry checker and tick the ingredients they already have at home. The app then returns recipes from the chosen region that are closest to that preference profile or pantry inventory.

## What is included

- `Data/indian_food.csv` - the dataset used in the project
- `recipe_recommender.py` - shared cleaning and recommendation logic
- `Training_Notebook.ipynb` - data exploration, model building, and sanity checks
- `app.py` - Streamlit demo
- `technical_report.md` - source for the written report
- `one_slide_pitch.md` - source for the one-slide pitch
- `build_deliverables.py` - script to generate the PDF report and PNG pitch

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notebook

Open `Training_Notebook.ipynb` in VS Code or Jupyter and run the cells from top to bottom. The notebook loads the dataset, cleans the missing region labels, builds the TF-IDF recommender, and demonstrates example recommendations.

## Generate deliverables

```bash
python build_deliverables.py
```

This writes `Deliverables/Technical_Report.pdf` and `Deliverables/One_Slide_Pitch.png`.

## Method summary

The recommender uses a content-based approach:

1. Clean the dataset and normalize the `-1` missing-value markers.
2. Build a searchable text field from ingredients, diet, flavor, course, and state.
3. Use TF-IDF to score similarity between the user profile and each candidate recipe.
4. Filter to the selected region first, then rank the best matches.
5. In pantry mode, compare the selected pantry ingredients against each recipe's ingredient list and rank by ingredient coverage.

## Dataset note

The dataset contains 255 recipes and 13 rows with missing region labels. Those rows are kept in the cleaned dataframe but excluded from region-specific recommendations.
