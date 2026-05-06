# One-Slide Pitch

## Recipe by Region Recommender

Find region-authentic Indian recipes in seconds.

### Problem

People often know the cuisine they want, but not the regional dishes that fit their taste, diet, or course preference.

### Solution

A region-aware recommender that filters by Indian region first, then ranks recipes by ingredient and metadata similarity.

### Why it works

- Uses the dataset's region, course, diet, and ingredient metadata
- Gives a simple, explainable ranking
- Runs instantly in a Streamlit demo

### Demo outcome

Pick North, South, East, West, Central, or North East, then add a craving like "cardamom milk dessert" to see ranked recipes.

### Stack

Pandas, scikit-learn TF-IDF, Streamlit, and a lightweight cleaning pipeline.
